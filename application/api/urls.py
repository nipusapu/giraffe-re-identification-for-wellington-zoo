# api/urls.py
from django.urls import path
from .views import upload_and_reid_api, api_result,presigned_download_url_api

urlpatterns = [
    path('upload/', upload_and_reid_api, name='api_upload'),
    #path('status/<int:job_id>/', api_status, name='api_status'),
    path('result/<int:job_id>/', api_result, name='api_result'),
    path("images/<uuid:image_id>/download-url/", presigned_download_url_api, name="image_download_url"),
]