from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .types import BlobObservation


@dataclass(slots=True)
class VisionSettings:
    threshold_mode: str
    hsv_lower: tuple[int, int, int]
    hsv_upper: tuple[int, int, int]
    brightness_threshold: int
    grayscale_threshold: int
    adaptive_block_size: int
    adaptive_c: float
    gaussian_blur_kernel: int
    morph_open_kernel: int
    morph_close_kernel: int
    erode_iterations: int
    dilate_iterations: int
    invert_mask: bool
    min_area_px: float
    max_area_fraction: float
    min_circularity: float
    prefer_brightest_blob: bool


@dataclass(slots=True)
class DetectionViews:
    rgb: np.ndarray
    hsv_visualization: np.ndarray
    grayscale: np.ndarray
    raw_mask: np.ndarray
    processed_mask: np.ndarray
    overlay: np.ndarray
    observation: BlobObservation


class LuminousBlobDetector:
    """In-memory luminous landmark detector used by training and the configurator.

    The network contract is unchanged: the detector returns the normalized Euclidean
    distance from the selected blob centroid to the image centre. This module only
    expands the preprocessing choices and exposes intermediate images for validation.
    """

    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LuminousBlobDetector":
        vision = cfg["vision"]
        legacy_kernel = max(0, int(vision.get("morphology_kernel", 3)))
        return cls(
            VisionSettings(
                threshold_mode=str(vision.get("threshold_mode", "hsv")),
                hsv_lower=tuple(int(v) for v in vision["hsv_lower"]),
                hsv_upper=tuple(int(v) for v in vision["hsv_upper"]),
                brightness_threshold=int(vision.get("brightness_threshold", 220)),
                grayscale_threshold=int(vision.get("grayscale_threshold", 220)),
                adaptive_block_size=int(vision.get("adaptive_block_size", 21)),
                adaptive_c=float(vision.get("adaptive_c", 3.0)),
                gaussian_blur_kernel=int(vision.get("gaussian_blur_kernel", 0)),
                morph_open_kernel=int(vision.get("morph_open_kernel", legacy_kernel)),
                morph_close_kernel=int(vision.get("morph_close_kernel", legacy_kernel)),
                erode_iterations=int(vision.get("erode_iterations", 0)),
                dilate_iterations=int(vision.get("dilate_iterations", 0)),
                invert_mask=bool(vision.get("invert_mask", False)),
                min_area_px=float(vision["min_area_px"]),
                max_area_fraction=float(vision["max_area_fraction"]),
                min_circularity=float(vision["min_circularity"]),
                prefer_brightest_blob=bool(vision.get("prefer_brightest_blob", True)),
            )
        )

    @staticmethod
    def _odd_kernel(value: int, *, minimum: int = 0) -> int:
        value = max(minimum, int(value))
        if value <= 1:
            return value
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            raise ValueError("RGB image must have shape [height, width, channels>=3].")
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                finite = image[np.isfinite(image)]
                maximum = float(np.max(finite)) if finite.size else 0.0
                scale = 255.0 if maximum <= 1.5 else 1.0
                image = np.clip(image * scale, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image[:, :, :3])

    def _hsv_mask(self, hsv: np.ndarray) -> np.ndarray:
        lower = np.array(self.settings.hsv_lower, dtype=np.uint8)
        upper = np.array(self.settings.hsv_upper, dtype=np.uint8)
        if int(lower[0]) <= int(upper[0]):
            return cv2.inRange(hsv, lower, upper)
        # Hue wrap is useful for red landmarks. S and V bounds remain unchanged.
        lower_a = np.array([0, lower[1], lower[2]], dtype=np.uint8)
        upper_a = np.array([upper[0], upper[1], upper[2]], dtype=np.uint8)
        lower_b = np.array([lower[0], lower[1], lower[2]], dtype=np.uint8)
        upper_b = np.array([179, upper[1], upper[2]], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower_a, upper_a), cv2.inRange(hsv, lower_b, upper_b))

    def build_masks(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rgb3 = self.normalize_rgb(rgb)
        blur_kernel = self._odd_kernel(self.settings.gaussian_blur_kernel)
        working = cv2.GaussianBlur(rgb3, (blur_kernel, blur_kernel), 0) if blur_kernel > 1 else rgb3
        hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
        mode = self.settings.threshold_mode
        if mode == "hsv":
            raw_mask = self._hsv_mask(hsv)
        elif mode == "brightness":
            raw_mask = cv2.inRange(hsv[:, :, 2], self.settings.brightness_threshold, 255)
        elif mode == "hsv_or_brightness":
            hsv_mask = self._hsv_mask(hsv)
            bright_mask = cv2.inRange(hsv[:, :, 2], self.settings.brightness_threshold, 255)
            raw_mask = cv2.bitwise_or(hsv_mask, bright_mask)
        elif mode == "grayscale":
            _, raw_mask = cv2.threshold(gray, self.settings.grayscale_threshold, 255, cv2.THRESH_BINARY)
        elif mode == "adaptive":
            block = self._odd_kernel(self.settings.adaptive_block_size, minimum=3)
            raw_mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                self.settings.adaptive_c,
            )
        else:
            raise ValueError(f"Unsupported threshold mode: {mode}")
        if self.settings.invert_mask:
            raw_mask = cv2.bitwise_not(raw_mask)

        processed = raw_mask.copy()
        open_kernel = self._odd_kernel(self.settings.morph_open_kernel)
        close_kernel = self._odd_kernel(self.settings.morph_close_kernel)
        if open_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        if close_kernel > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        if self.settings.erode_iterations > 0:
            processed = cv2.erode(processed, None, iterations=self.settings.erode_iterations)
        if self.settings.dilate_iterations > 0:
            processed = cv2.dilate(processed, None, iterations=self.settings.dilate_iterations)
        return rgb3, hsv, gray, processed if processed is not None else raw_mask

    def detect(self, rgb: np.ndarray) -> BlobObservation:
        try:
            rgb3, hsv, _gray, mask = self.build_masks(rgb)
        except (ValueError, cv2.error):
            return self._missing()
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
            if self.settings.prefer_brightest_blob:
                score = area * max(circularity, 0.05) * (0.25 + mean_value / 255.0)
            else:
                score = area * max(circularity, 0.05)
            candidates.append((score, contour, area, circularity, mean_value))
        mask_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        if not candidates:
            return self._missing(mask_fraction=mask_fraction)
        _, contour, area, circularity, mean_value = max(candidates, key=lambda item: item[0])
        moments = cv2.moments(contour)
        if abs(moments["m00"]) <= 1e-9:
            return self._missing(mask_fraction=mask_fraction)
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
            mask_fraction=mask_fraction,
        )

    def overlay(self, rgb: np.ndarray, observation: BlobObservation) -> np.ndarray:
        output = self.normalize_rgb(rgb).copy()
        height, width = output.shape[:2]
        cv2.drawMarker(output, (width // 2, height // 2), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if observation.visible and observation.centroid_x is not None and observation.centroid_y is not None:
            point = (int(round(observation.centroid_x)), int(round(observation.centroid_y)))
            cv2.circle(output, point, 8, (255, 0, 0), 2)
            cv2.line(output, (width // 2, height // 2), point, (255, 255, 255), 1)
        return output

    def debug_views(self, rgb: np.ndarray) -> DetectionViews:
        rgb3 = self.normalize_rgb(rgb)
        working = rgb3
        blur_kernel = self._odd_kernel(self.settings.gaussian_blur_kernel)
        if blur_kernel > 1:
            working = cv2.GaussianBlur(rgb3, (blur_kernel, blur_kernel), 0)
        hsv = cv2.cvtColor(working, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
        # Recreate the raw mask before morphology so the configurator can compare it.
        mode = self.settings.threshold_mode
        if mode == "hsv":
            raw_mask = self._hsv_mask(hsv)
        elif mode == "brightness":
            raw_mask = cv2.inRange(hsv[:, :, 2], self.settings.brightness_threshold, 255)
        elif mode == "hsv_or_brightness":
            raw_mask = cv2.bitwise_or(
                self._hsv_mask(hsv), cv2.inRange(hsv[:, :, 2], self.settings.brightness_threshold, 255)
            )
        elif mode == "grayscale":
            _, raw_mask = cv2.threshold(gray, self.settings.grayscale_threshold, 255, cv2.THRESH_BINARY)
        else:
            block = self._odd_kernel(self.settings.adaptive_block_size, minimum=3)
            raw_mask = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                self.settings.adaptive_c,
            )
        if self.settings.invert_mask:
            raw_mask = cv2.bitwise_not(raw_mask)
        _rgb, _hsv, _gray, processed = self.build_masks(rgb3)
        observation = self.detect(rgb3)
        overlay = self.overlay(rgb3, observation)
        hsv_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return DetectionViews(
            rgb=rgb3,
            hsv_visualization=hsv_visualization,
            grayscale=gray,
            raw_mask=raw_mask,
            processed_mask=processed,
            overlay=overlay,
            observation=observation,
        )

    @staticmethod
    def _missing(mask_fraction: float = 0.0) -> BlobObservation:
        return BlobObservation(False, None, None, None, 1.0, 0.0, 0.0, 0.0, mask_fraction)
