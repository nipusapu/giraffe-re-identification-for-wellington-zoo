from django.apps import AppConfig
from django.db.models.signals import post_migrate


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    
    def ready(self):
        from .signals import seed_after_migrate  # local import to avoid app registry issues
        post_migrate.connect(seed_after_migrate, sender=self)
