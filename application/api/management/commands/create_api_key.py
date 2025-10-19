from django.core.management.base import BaseCommand
from api.models import ApiKey
import secrets, hashlib

class Command(BaseCommand):
    help = "Create an API key for server-to-server auth"

    def add_arguments(self, parser):
        parser.add_argument("name")

    def handle(self, *args, **opts):
        name = opts["name"]
        prefix = secrets.token_hex(6)          # 12 hex chars
        secret = secrets.token_urlsafe(32)     # long random
        salt   = secrets.token_hex(16)
        hashed = hashlib.sha256((secret + salt).encode()).hexdigest()
        obj = ApiKey.objects.create(name=name, prefix=prefix, salt=salt, hashed_key=hashed)
        self.stdout.write(self.style.SUCCESS(f"Created key for {name}"))
        self.stdout.write(self.style.WARNING("Store this securely now; it won't be shown again:"))
        self.stdout.write(f"X-API-Key: {prefix}.{secret}")
