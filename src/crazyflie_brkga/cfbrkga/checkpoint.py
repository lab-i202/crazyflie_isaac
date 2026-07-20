from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import Individual


def save_checkpoint(path: Path, generation_index: int, population: list[Individual], rng_state: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = np.stack([item.random_keys for item in population])
    fitness = np.array([np.nan if item.fitness is None else item.fitness for item in population], dtype=np.float64)
    ids = np.array([item.individual_id for item in population], dtype=np.int64)
    np.savez_compressed(path, generation=generation_index, random_keys=keys, fitness=fitness, individual_ids=ids)
    path.with_suffix(".rng.json").write_text(json.dumps(rng_state, indent=2), encoding="utf-8")
    return path
