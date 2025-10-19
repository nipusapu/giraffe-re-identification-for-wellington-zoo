import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE','config.settings')
app = Celery("reid")           # any name is fine
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()       # finds tasks.py in your INSTALLED_APPS

# (optional) default config
app.conf.timezone = "UTC"
