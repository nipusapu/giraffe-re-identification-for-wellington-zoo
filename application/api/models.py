# api/models.py
import os
import uuid
from django.db import models
from django.utils import timezone


class StoredImage(models.Model):
    STORAGE_CHOICES = [('local', 'Local'), ('s3', 'S3')]
    # Use UUID as the public ID
    storage      = models.CharField(max_length=10, choices=STORAGE_CHOICES, default='local', db_index=True)
    object_id    = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_name= models.CharField(max_length=255)
    file_ext     = models.CharField(max_length=16, blank=True)     # ".jpg"
    content_type = models.CharField(max_length=100, blank=True)
    size         = models.PositiveBigIntegerField(null=True, blank=True)
    etag         = models.CharField(max_length=64, blank=True)      # from S3 (optional)
    version_id   = models.CharField(max_length=255, null=True, blank=True)
    local_path   = models.CharField(max_length=500, blank=True, null=True)  # for local file storage
    created_at   = models.DateTimeField(auto_now_add=True)
    storage_key  = models.CharField(max_length=512, null=True, blank=True)

    # Key is derived at runtime; not stored in DB
    def s3_key(self) -> str:
        return f"uploads/{self.object_id}{self.file_ext or ''}"

    def __str__(self):
        return f"{self.object_id} ({self.original_name})"


def upload_to(_instance, filename: str) -> str:
    # If you ever use ImageField again; otherwise unused
    date = timezone.now().strftime("%Y/%m/%d")
    ext = os.path.splitext(filename)[1].lower()
    return f"uploads/{date}/{uuid.uuid4().hex}{ext}"


class Animal(models.Model):
    code        = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.code


class ReIDResult(models.Model):
    STATUS_CHOICES = [
        ('queued',        'Queued'),
        ('detecting',     'Detecting'),
        ('no_detection',  'No detection'),
        ('reidentifying', 'Re-Identifying'),
        ('completed',     'Completed'),
        ('error',         'Error'),
    ]

    # Link to StoredImage, not a file field
    image            = models.ForeignKey(StoredImage, on_delete=models.CASCADE, related_name="reid_results")
    status           = models.CharField(max_length=20, choices=STATUS_CHOICES, default='queued')
    predicted_animal = models.ForeignKey(Animal, on_delete=models.SET_NULL, null=True, blank=True, related_name='reid_results')
    votes_json       = models.JSONField(default=dict, blank=True)
    created_at       = models.DateTimeField(auto_now_add=True)
    updated_at       = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['created_at']),
        ]

    def __str__(self):
        animal = self.predicted_animal.code if self.predicted_animal_id else "—"
        return f"ReIDResult #{self.pk} [{self.status}] → {animal}"

    @property
    def predicted_code(self):
        return self.predicted_animal.code if self.predicted_animal_id else None


class ImageTag(models.Model):
    # Point to *api* models (or use the class names directly, as below)
    image         = models.ForeignKey(StoredImage, on_delete=models.CASCADE, related_name="tags")
    animal        = models.ForeignKey(Animal, on_delete=models.CASCADE, related_name="image_tags")
    reid_result   = models.ForeignKey(ReIDResult, null=True, blank=True, on_delete=models.SET_NULL, related_name="tags")

    # extra metadata
    code_snapshot = models.CharField(max_length=100)
    confidence    = models.FloatField(null=True, blank=True)
    source        = models.CharField(max_length=20, choices=[("auto","Auto"),("manual","Manual")], default="auto")
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("image", "animal")]
        indexes = [
            models.Index(fields=["image"]),
            models.Index(fields=["animal"]),
            models.Index(fields=["code_snapshot"]),
        ]

    def __str__(self):
        return f"{self.image_id} → {self.code_snapshot}"
    

class ApiKey(models.Model):
    name = models.CharField(max_length=100, unique=True)
    prefix = models.CharField(max_length=16, unique=True, db_index=True)
    salt = models.CharField(max_length=32)
    hashed_key = models.CharField(max_length=64)  # hex sha256
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.name} ({self.prefix})"
