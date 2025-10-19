# api/views.py
import os
import mimetypes
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from drf_yasg.utils import swagger_auto_schema
from django.conf import settings
from django.shortcuts import get_object_or_404
from drf_yasg import openapi
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import AllowAny  # keep if you want to expose some endpoints
from rest_framework.response import Response
from rest_framework import status
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

from api.models import ReIDResult, StoredImage
from .serializers import ReIDResultSerializer  # (kept if you use it elsewhere)
from reid.tasks.pipeline import pipeline_run


# ------------ Storage helpers ------------
def _default_storage_is_s3() -> bool:
    try:
        name = default_storage.__class__.__name__.lower()
        return "s3" in name or hasattr(default_storage, "bucket_name")
    except Exception:
        return False


def _get_s3_client():
    if not getattr(settings, "AWS_USE_S3", False):
        return None
    region = getattr(settings, "AWS_S3_REGION_NAME", None)
    bucket = getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
    if not (region and bucket):
        return None
    try:
        return boto3.client("s3", region_name=region)
    except Exception:
        return None


_S3_CLIENT = _get_s3_client()
DEFAULT_PRESIGN_EXPIRES = int(getattr(settings, "PRESIGN_EXPIRES", 300))


def _build_image_url(img: StoredImage, expires: Optional[int] = None) -> Optional[str]:
    """
    Return a browser-viewable URL for a StoredImage.
    - S3: presigned URL (expires in `expires` seconds)
    - Local: MEDIA_URL + key if file exists
    """
    if expires is None:
        expires = DEFAULT_PRESIGN_EXPIRES

    if img.storage == "s3":
        bucket = getattr(default_storage, "bucket_name", None) or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        client = getattr(getattr(getattr(default_storage, "connection", None), "meta", None), "client", None) or _S3_CLIENT
        if not (client and bucket):
            return None
        params = {"Bucket": bucket, "Key": img.s3_key()}
        if img.version_id:
            params["VersionId"] = img.version_id
        try:
            return client.generate_presigned_url("get_object", Params=params, ExpiresIn=int(expires))
        except Exception:
            return None

    # Local
    local_path = img.local_path
    if local_path and os.path.exists(local_path):
        return f"{settings.MEDIA_URL}{img.s3_key()}"
    return None


# ------------ Swagger bits ------------
file_param = openapi.Parameter(
    name="image",
    in_=openapi.IN_FORM,
    type=openapi.TYPE_FILE,
    description="Image file to upload (multipart/form-data).",
    required=True,
)

resp_201 = openapi.Response(
    description="Created",
    schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            "reid_id": openapi.Schema(type=openapi.TYPE_INTEGER),
            "image_id": openapi.Schema(type=openapi.TYPE_STRING, description="UUID of StoredImage"),
            "status": openapi.Schema(type=openapi.TYPE_STRING),
            "image_url": openapi.Schema(type=openapi.TYPE_STRING, description="Preview URL (presigned for S3 / direct for local)"),
        },
    ),
)

result_response = openapi.Response(
    description="Job result",
    schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            "id": openapi.Schema(type=openapi.TYPE_INTEGER),
            "status": openapi.Schema(type=openapi.TYPE_STRING),
            "predicted_animal": openapi.Schema(type=openapi.TYPE_STRING, description="Animal code (technical)"),
            "display_name": openapi.Schema(type=openapi.TYPE_STRING, description="User-facing name (pretty)"),
            "description": openapi.Schema(type=openapi.TYPE_STRING, description="User-facing description/bio"),
            "image_id": openapi.Schema(type=openapi.TYPE_STRING, description="UUID of StoredImage"),
            "image_url": openapi.Schema(type=openapi.TYPE_STRING, description="Preview URL (presigned/local)"),
            "votes": openapi.Schema(type=openapi.TYPE_OBJECT),
        },
    ),
)


# ------------ Upload API ------------
@swagger_auto_schema(
    method="post",
    manual_parameters=[file_param],
    responses={201: resp_201, 400: "Bad Request", 502: "S3/local save error"},
)
@api_view(["POST"])
# @permission_classes([AllowAny])  # uncomment if you intend to make this open; otherwise enforce your API-key auth globally
@parser_classes([MultiPartParser, FormParser])
def upload_and_reid_api(request):
    """
    Upload -> create StoredImage -> save to S3 or Local (not both) -> create ReIDResult -> enqueue pipeline.
    If AWS_USE_S3=True, upload is S3-or-error (no local fallback).

    Returns:
      - reid_id (int)
      - image_id (uuid string)
      - status (str)
      - image_url (str) preview link (presigned for S3 / direct for local)
    """
    img = request.FILES.get("image")
    if not img:
        return Response({"detail": "No image provided."}, status=status.HTTP_400_BAD_REQUEST)

    if img.size > 10 * 1024 * 1024:
        return Response({"detail": "File too large (max 10MB)."}, status=status.HTTP_400_BAD_REQUEST)

    ct = img.content_type or mimetypes.guess_type(img.name)[0] or "application/octet-stream"
    if not ct.startswith("image/"):
        return Response({"detail": "Only image/* files allowed."}, status=status.HTTP_400_BAD_REQUEST)

    # 1) DB row first (to compute storage key)
    _, ext = os.path.splitext(img.name or "")
    stored = StoredImage.objects.create(
        original_name=img.name or "",
        file_ext=(ext[:16] if ext else ""),
        content_type=ct,
        size=img.size,
    )
    key = stored.s3_key()

    # 2) Save: S3-only when AWS_USE_S3=True; Local-only otherwise
    want_s3 = bool(getattr(settings, "AWS_USE_S3", False))
    bucket = getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")

    if want_s3:
        # S3-or-error: NEVER write a local persistent copy when S3 is enabled
        if _default_storage_is_s3():
            # Save via Django S3 storage backend
            default_storage.save(key, ContentFile(img.read()))
            # Best-effort to fetch ETag/VersionId
            try:
                conn = getattr(default_storage, "connection", None)
                client = getattr(getattr(conn, "meta", None), "client", None)
                if client and bucket:
                    head = client.head_object(Bucket=bucket, Key=key)
                    stored.etag = (head.get("ETag") or "").strip('"')
                    stored.version_id = head.get("VersionId")
            except Exception:
                pass
            updates = ["storage"]
            stored.storage = "s3"
            if stored.etag:
                updates.append("etag")
            if stored.version_id:
                updates.append("version_id")
            stored.save(update_fields=updates)

        elif _S3_CLIENT and bucket:
            # Save via boto3 (default_storage is local, but we still target S3)
            try:
                put = _S3_CLIENT.put_object(Bucket=bucket, Key=key, Body=img, ContentType=ct)
                etag = (put.get("ETag") or "").strip('"')
                ver = put.get("VersionId")
                stored.storage = "s3"
                updates = ["storage"]
                if etag:
                    stored.etag = etag
                    updates.append("etag")
                if ver:
                    stored.version_id = ver
                    updates.append("version_id")
                stored.save(update_fields=updates)
            except ClientError as e:
                stored.delete()
                return Response({"detail": "Failed to upload to S3.", "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        else:
            stored.delete()
            return Response(
                {"detail": "S3 is enabled but not configured (no backend/client/bucket)."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

    else:
        # Local-only (S3 disabled)
        try:
            local_path = os.path.join(settings.MEDIA_ROOT, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as dest:
                for chunk in img.chunks():
                    dest.write(chunk)
            stored.local_path = local_path
            stored.storage = "local"
            stored.save(update_fields=["local_path", "storage"])
        except Exception as e:
            stored.delete()
            return Response({"detail": "Failed to save image locally.", "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)

    # 3) Create ReID job
    reid = ReIDResult.objects.create(image=stored, status="queued")

    # 4) Kick pipeline
    try:
        pipeline_run.delay(reid.pk, stored.s3_key())
    except Exception:
        reid.status = "error"
        reid.save(update_fields=["status"])

    # 5) Preview URL (so UI can show the uploaded image immediately)
    img_url = _build_image_url(stored, expires=DEFAULT_PRESIGN_EXPIRES)

    return Response(
        {
            "reid_id": reid.pk,
            "image_id": str(stored.object_id),
            "status": reid.status,
            "image_url": img_url,
        },
        status=status.HTTP_201_CREATED,
    )


# ------------ Result API ------------
@swagger_auto_schema(method="get", responses={200: result_response, 404: "Not found"})
@api_view(["GET"])
def api_result(request, job_id: int):
    """
    Return a friendly, user-facing payload for the result card:
    - display_name: pretty name for UI (falls back to code/predicted_id)
    - description: longer user-facing text (falls back to bio/about if present)
    - image_url: preview link (presigned for S3, direct for local)
    - predicted_animal: technical code (kept for compatibility)
    - votes: technical details (optional)
    """
    try:
        rec = ReIDResult.objects.get(pk=job_id)
    except ReIDResult.DoesNotExist:
        return Response({"detail": "Not found"}, status=404)

    # Optional override of expiry via query (?expires=600)
    expires_q = request.query_params.get("expires")
    try:
        expires = int(expires_q) if expires_q is not None else DEFAULT_PRESIGN_EXPIRES
    except ValueError:
        expires = DEFAULT_PRESIGN_EXPIRES

    # Predicted animal fields (defensive: support various schema shapes)
    code = None
    display_name = None
    description = None

    if hasattr(rec, "predicted_animal") and rec.predicted_animal_id:
        animal = rec.predicted_animal
        code = getattr(animal, "code", None) or getattr(animal, "slug", None) or getattr(animal, "id", None)
        display_name = (
            getattr(animal, "display_name", None)
            or getattr(animal, "name", None)
            or getattr(animal, "title", None)
            or (str(code) if code else None)
        )
        description = (
            getattr(animal, "description", None)
            or getattr(animal, "bio", None)
            or getattr(animal, "about", None)
        )
    elif hasattr(rec, "predicted_id"):
        code = rec.predicted_id
        display_name = str(code)

    # Build preview URL from the stored image
    img = rec.image
    img_url = _build_image_url(img, expires=expires)

    payload = {
        "id": rec.pk,
        "status": rec.status,
        "predicted_animal": code,
        "display_name": display_name,
        "description": description,
        "image_id": str(img.pk),
        "image_url": img_url,
        "votes": rec.votes_json,
    }

    return Response(payload, status=200)


# ------------ Download API ------------
expires_param = openapi.Parameter(
    name="expires",
    in_=openapi.IN_QUERY,
    description="Expiry in seconds (default 300).",
    type=openapi.TYPE_INTEGER,
    required=False,
)

response_200 = openapi.Response(
    description="Presigned URL",
    schema=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            "image_id": openapi.Schema(type=openapi.TYPE_STRING),
            "url": openapi.Schema(type=openapi.TYPE_STRING),
            "expires_in": openapi.Schema(type=openapi.TYPE_INTEGER),
        },
    ),
)

@swagger_auto_schema(method="get", manual_parameters=[expires_param], responses={200: response_200, 404: "Not found"})
@api_view(["GET"])
# @permission_classes([AllowAny])
def presigned_download_url_api(request, image_id):
    """
    For images stored in S3, return a presigned URL.
    For local images, return MEDIA_URL + key.
    """
    img = get_object_or_404(StoredImage, pk=image_id)

    expires = request.query_params.get("expires", DEFAULT_PRESIGN_EXPIRES)
    try:
        expires = int(expires)
    except ValueError:
        return Response({"detail": "expires must be an integer"}, status=status.HTTP_400_BAD_REQUEST)

    if img.storage == "s3":
        # Prefer bucket from storage backend; fall back to settings
        bucket = getattr(default_storage, "bucket_name", None) or getattr(settings, "AWS_STORAGE_BUCKET_NAME", "")
        # Try client from storage backend; else fall back to our boto client
        client = getattr(getattr(getattr(default_storage, "connection", None), "meta", None), "client", None) or _S3_CLIENT
        if not (client and bucket):
            return Response({"detail": "S3 not configured for presign."}, status=status.HTTP_502_BAD_GATEWAY)

        params = {"Bucket": bucket, "Key": img.s3_key()}
        if img.version_id:
            params["VersionId"] = img.version_id

        try:
            url = client.generate_presigned_url("get_object", Params=params, ExpiresIn=expires)
        except ClientError as e:
            return Response({"detail": "Failed to generate URL", "error": str(e)}, status=status.HTTP_502_BAD_GATEWAY)
        return Response({"image_id": str(img.pk), "url": url, "expires_in": expires})

    # Local
    local_path = img.local_path
    if not local_path or not os.path.exists(local_path):
        return Response({"detail": "Image not found locally."}, status=status.HTTP_404_NOT_FOUND)
    url = f"{settings.MEDIA_URL}{img.s3_key()}"
    return Response({"image_id": str(img.pk), "url": url, "expires_in": expires})
