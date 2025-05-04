import uuid

import numpy as np
from .PRI import PRIJitter, PRIStagger, PRIDwellSwitch
from .Frequency import (
    Frequency1,
    Frequency2,
    Frequency3_1,
    Frequency3_2,
    Frequency3_3,
    FrequencyBuilder,
    FrequencyConfig,
)
from .PW import PW
from .BW import BW
from .Emitter import EmitterConfig, Emitter, build_emitter
import json
import os


def save_emitter(emitter, SAVE_DIR="saved_emitters"):

    os.makedirs(SAVE_DIR, exist_ok=True)  # Skapa mappen om den inte finns
    emitter_data = emitter.model_dump()
    filename = str(emitter.id) + ".json"
    filepath = os.path.join(SAVE_DIR, filename)  # 🔥 Lägg i rätt mapp

    with open(filepath, "w") as f:
        json.dump(emitter_data, f, indent=4, default=str)

    return filepath  # (full sökväg som returneras)


def load_emitter(filename: str) -> Emitter:

    with open(filename, "r") as f:
        data = json.load(f)

    config = EmitterConfig(**data["config"])
    rng_state = data["rng_state"]
    emitter = build_emitter(config=config, rng_state=rng_state)

    return emitter
