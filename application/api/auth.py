from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from django.utils.crypto import constant_time_compare
from django.contrib.auth.models import AnonymousUser
from django.utils import timezone
from .models import ApiKey
import hashlib

class APIKeyAuthentication(BaseAuthentication):
    header_name = "HTTP_X_API_KEY"  # maps to 'X-API-Key' header

    def authenticate(self, request):
        raw = request.META.get(self.header_name)
        if not raw:  # no header -> let other auth backends try
            return None
        try:
            prefix, secret = raw.split(".", 1)
        except ValueError:
            raise AuthenticationFailed("Invalid API key format")

        try:
            rec = ApiKey.objects.get(prefix=prefix, is_active=True)
        except ApiKey.DoesNotExist:
            raise AuthenticationFailed("Invalid API key")

        expected = hashlib.sha256((secret + rec.salt).encode()).hexdigest()
        if not constant_time_compare(expected, rec.hashed_key):
            raise AuthenticationFailed("Invalid API key")

        rec.last_used_at = timezone.now()
        rec.save(update_fields=["last_used_at"])
        # Return a pseudo-user and attach the key record in request.auth
        return (AnonymousUser(), rec)

from rest_framework.permissions import BasePermission
class HasAPIKey(BasePermission):
    def has_permission(self, request, view):
        return getattr(request, "auth", None) is not None and \
               isinstance(request.auth, ApiKey) and request.auth.is_active

# Optional: throttle per key
from rest_framework.throttling import SimpleRateThrottle
class APIKeyRateThrottle(SimpleRateThrottle):
    scope = "api_key"
    def get_cache_key(self, request, view):
        rec = getattr(request, "auth", None)
        if not rec or not isinstance(rec, ApiKey):
            return None
        return self.cache_format % {"scope": self.scope, "ident": rec.prefix}
