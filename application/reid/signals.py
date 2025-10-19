# reid/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from api.models import ReIDResult, ImageTag

@receiver(post_save, sender=ReIDResult)
def create_tag_on_completion(sender, instance: ReIDResult, created, **kwargs):
    if instance.status == "completed" and instance.predicted_animal_id and instance.image_id:
        # try to carry confidence from your pipeline output, if present
        confidence = None
        if instance.votes_json and isinstance(instance.votes_json, dict):
            confidence = instance.votes_json.get("confidence")

        ImageTag.objects.get_or_create(
            image=instance.image,
            animal=instance.predicted_animal,
            defaults={
                "code_snapshot": instance.predicted_animal.code,
                "reid_result": instance,
                "confidence": confidence,
                "source": "auto",
            },
        )
