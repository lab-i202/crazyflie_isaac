from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from cfbrkga.config import load_config, save_config
from cfbrkga.image_processing import LuminousBlobDetector


def _synthetic_landmark() -> np.ndarray:
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    cv2.circle(image, (150, 90), 16, (255, 230, 35), -1)
    cv2.circle(image, (150, 90), 8, (255, 255, 255), -1)
    return image


def test_configuration_round_trip_keeps_gui_sections(tmp_path: Path):
    cfg = load_config()
    cfg["environment"]["room"]["has_roof"] = True
    cfg["environment"]["landmark"]["light_intensity"] = 4321.0
    cfg["scenario_preview"]["camera"]["azimuth_deg"] = 61.0
    cfg["vision"]["threshold_mode"] = "hsv_or_brightness"
    output = save_config(cfg, tmp_path / "gui_round_trip.json")

    loaded = load_config(output)
    assert loaded["environment"]["room"]["has_roof"] is True
    assert loaded["environment"]["landmark"]["light_intensity"] == 4321.0
    assert loaded["scenario_preview"]["camera"]["azimuth_deg"] == 61.0
    assert loaded["vision"]["threshold_mode"] == "hsv_or_brightness"


def test_detector_debug_views_for_configurable_modes():
    image = _synthetic_landmark()
    for mode in ("hsv", "brightness", "hsv_or_brightness", "grayscale"):
        cfg = load_config()
        cfg["vision"]["threshold_mode"] = mode
        cfg["vision"]["brightness_threshold"] = 200
        cfg["vision"]["grayscale_threshold"] = 180
        cfg["vision"]["hsv_lower"] = [10, 40, 160]
        cfg["vision"]["hsv_upper"] = [60, 255, 255]
        detector = LuminousBlobDetector.from_config(cfg)
        views = detector.debug_views(image)
        assert views.raw_mask.shape == image.shape[:2]
        assert views.processed_mask.shape == image.shape[:2]
        assert views.overlay.shape == image.shape
        assert views.observation.visible
