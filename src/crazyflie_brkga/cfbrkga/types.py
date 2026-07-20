from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Individual:
    individual_id: int
    generation: int
    index_in_generation: int
    random_keys: np.ndarray
    decoded_weights: np.ndarray
    origin: str
    elite_parent_id: int | None = None
    non_elite_parent_id: int | None = None
    fitness: float | None = None


@dataclass(slots=True)
class BlobObservation:
    visible: bool
    centroid_x: float | None
    centroid_y: float | None
    euclidean_px: float | None
    euclidean_norm: float
    area_px: float
    circularity: float
    mean_value: float
    mask_fraction: float


@dataclass(slots=True)
class StepOutcome:
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    reason: str
    front_norm: float
    step_penalty: float
    lost_landmark_streak: int


@dataclass(slots=True)
class EvaluationResult:
    individual_id: int
    generation: int
    env_index: int
    episode_seed: int
    reward: float
    status: str
    reason: str
    success: bool
    steps: int
    simulated_time_s: float
    wall_time_s: float
    collision_count: int
    visible_steps: int
    invalid_camera_frames: int
    mean_euclid_norm: float
    min_front_tof_m: float
    final_front_tof_m: float
    final_euclid_norm: float
    debug_dir: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
