# yourapp/seed.py
from django.db import transaction
from .models import Animal

ANIMALS = {
    "ZAHARA": (
        "Zahara was born at Wellington Zoo in 2004. You can easily identify her by her "
        "dark coat and the butterfly shaped spot on her neck. Zahara is very inquisitive "
        "and often leads the way for Zuri and Sunny. She really enjoys taking part in the "
        "training sessions with the keepers."
    ),
    "SUNNY": (
        "Sunny is Nia’s dad and the only male giraffe living at the zoo. He arrived from "
        "Australia in 2019. You’ll notice that he has just one eye. Vets in Australia "
        "tried unsuccessfully to save it after an injury – but it doesn’t hold him back. "
        "He is calm, friendly and even a bit cheeky at times."
    ),
    "ZURI": (
        "Zuri is a first time Mum to Nia and also Zahara’s niece. She was born at "
        "Auckland Zoo and made the move to Wellington in 2016. Zuri is a gentle "
        "individual but can be feisty when she wants to be! Zuri enjoys her independence "
        "but has a strong bond with each of the herd members."
    ),
    "NIA": (
        "Nia was born at Wellington Zoo on November 29, 2023 to Mum Zuri and Dad Sunny. "
        "She was nearly 6ft tall at birth and weighed 75kg! She is very confident and "
        "inquisitive, as long as Mum is nearby. Like the meaning of her name, Nia is "
        "shaping up to be a tenacious individual – much like Mum!"
    ),
}

def ensure_seed_animals() -> tuple[int, int]:
    """
    Create or update the default Animal rows.
    Returns: (created_count, updated_count)
    """
    created, updated = 0, 0
    with transaction.atomic():
        for code, description in ANIMALS.items():
            obj, is_created = Animal.objects.get_or_create(
                code=code,
                defaults={"description": description},
            )
            if is_created:
                created += 1
            else:
                if obj.description != description:
                    obj.description = description
                    obj.save(update_fields=["description"])
                    updated += 1
    return created, updated
