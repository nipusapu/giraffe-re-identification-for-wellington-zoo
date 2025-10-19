# config/urls.py
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.shortcuts import redirect
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework.permissions import AllowAny

schema_view = get_schema_view(
    openapi.Info(
        title="ReID API",
        default_version="v1",
        description="Giraffe re-identification endpoints",
    ),
    public=True,
    permission_classes=[AllowAny],
)

def root(_req):
    return redirect("schema-swagger-ui")

urlpatterns = [
    path("", root),
    path("api/", include("api.urls")),  # <— only here
    path("swagger/", schema_view.with_ui("swagger", cache_timeout=0), name="schema-swagger-ui"),
    path("redoc/",   schema_view.with_ui("redoc",   cache_timeout=0), name="schema-redoc"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
