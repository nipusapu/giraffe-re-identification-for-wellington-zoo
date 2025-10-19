from django.conf import settings
def current_storage_label() -> str:
    return 's3' if getattr(settings, 'AWS_USE_S3', False) else 'local'