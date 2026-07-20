from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .types import BlobObservation


@dataclass(slots=True)
class VisionSettings:
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    min_area_px: float
    max_area_fraction: float
    min_circularity: float
    morphology_kernel: int


class LuminousBlobDetector:
    """In-memory luminous landmark detector.

    The original Gazebo implementation wrote frames to disk and used a black-hat
    transform. This detector processes the Isaac RGB array directly and targets a
    bright configurable color range. The network still receives exactly the original
    two values: front ToF and normalized Euclidean center error.
    """

    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, cfg: dict) -> "LuminousBlobDetector":
        vision = cfg["vision"]
        return cls(
            VisionSettings(
                hsv_lower=tuple(int(v) for v in vision["hsv_lower"]),
                hsv_upper=tuple(int(v) for v in vision["hsv_upper"]),
                min_area_px=float(vision["min_area_px"]),
                max_area_fraction=float(vision["max_area_fraction"]),
                min_circularity=float(vision["min_circularity"]),
                morphology_kernel=max(1, int(vision["morphology_kernel"])),
            )
        )

    def detect(self, rgb: np.ndarray) -> BlobObservation:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            return self._missing()
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                scale = 255.0 if float(np.nanmax(image)) <= 1.5 else 1.0
                image = np.clip(image * scale, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        rgb3 = np.ascontiguousarray(image[:, :, :3])
        hsv = cv2.cvtColor(rgb3, cv2.COLOR_RGB2HSV)
        lower = np.array(self.settings.hsv_lower, dtype=np.uint8)
        upper = np.array(self.settings.hsv_upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        k = self.settings.morphology_kernel
        if k > 1:
            if k % 2 == 0:
                k += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = rgb3.shape[:2]
        max_area = width * height * self.settings.max_area_fraction
        candidates: list[tuple[float, np.ndarray, float, float, float]] = []
        value_channel = hsv[:, :, 2]
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.settings.min_area_px or area > max_area:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-9:
                continue
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
            if circularity < self.settings.min_circularity:
                continue
            local_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(local_mask, [contour], -1, 255, thickness=-1)
            mean_value = float(cv2.mean(value_channel, mask=local_mask)[0])
            score = area * max(circularity, 0.05) * (0.5 + mean_value / 255.0)
            candidates.append((score, contour, area, circularity, mean_value))
        if not candidates:
            return self._missing(mask_fraction=float(np.count_nonzero(mask)) / float(mask.size))
        _, contour, area, circularity, mean_value = max(candidates, key=lambda item: item[0])
        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-9:
            return self._missing(mask_fraction=float(np.count_nonzero(mask)) / float(mask.size))
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])
        center_x = width / 2.0
        center_y = height / 2.0
        euclidean = math.hypot(cx - center_x, cy - center_y)
        # Preserve the executable Gazebo baseline: normalize by the full diagonal.
        diagonal = math.hypot(width, height)
        euclidean_norm = min(1.0, max(0.0, euclidean / diagonal if diagonal > 0 else 1.0))
        return BlobObservation(
            visible=True,
            centroid_x=cx,
            centroid_y=cy,
            euclidean_px=euclidean,
            euclidean_norm=euclidean_norm,
            area_px=area,
            circularity=circularity,
            mean_value=mean_value,
            mask_fraction=float(np.count_nonzero(mask)) / float(mask.size),
        )

    def overlay(self, rgb: np.ndarray, observation: BlobObservation) -> np.ndarray:
        output = np.ascontiguousarray(np.asarray(rgb)[:, :, :3].copy())
        height, width = output.shape[:2]
        cv2.drawMarker(output, (width // 2, height // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if observation.visible and observation.centroid_x is not None and observation.centroid_y is not None:
            point = (int(round(observation.centroid_x)), int(round(observation.centroid_y)))
            cv2.circle(output, point, 8, (255, 0, 0), 2)
            cv2.line(output, (width // 2, height // 2), point, (255, 255, 255), 1)
        return output

    @staticmethod
    def _missing(mask_fraction: float = 0.0) -> BlobObservation:
        return BlobObservation(False, None, None, None, 1.0, 0.0, 0.0, 0.0, mask_fraction)
