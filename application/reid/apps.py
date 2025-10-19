# reid/apps.py
from django.apps import AppConfig

class ReidConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "reid"
    
    def ready(self):
        from . import signals  # noqa: F401
