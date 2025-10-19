# yourapp/signals.py
import logging
from .seed import ensure_seed_animals

log = logging.getLogger(__name__)

def seed_after_migrate(**kwargs):
    created, updated = ensure_seed_animals()
    if created or updated:
        log.info("Animal seed: created=%s updated=%s", created, updated)
