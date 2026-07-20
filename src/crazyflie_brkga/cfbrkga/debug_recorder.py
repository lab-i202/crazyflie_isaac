from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


class DebugEpisodeRecorder:
    def __init__(self, base_dir: Path, enabled: bool, save_frames: bool, frame_stride: int) -> None:
        self.base_dir = base_dir
        self.enabled = enabled
        self.save_frames = save_frames
        self.frame_stride = max(1, int(frame_stride))
        self._csv_file = None
        self._writer = None
        if enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._csv_file = (self.base_dir / "steps.csv").open("w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._csv_file,
                fieldnames=[
                    "step_index", "simulation_time_s", "camera_frame_index", "action", "reward",
                    "front_tof_m", "euclid_norm", "visible", "blob_x", "blob_y", "x_m", "y_m",
                    "z_m", "yaw_rad", "collision", "reason",
                ],
            )
            self._writer.writeheader()
            if save_frames:
                (self.base_dir / "camera").mkdir(exist_ok=True)

    def write_step(self, row: dict, rgb: np.ndarray | None = None, overlay: np.ndarray | None = None) -> None:
        if not self.enabled:
            return
        assert self._writer is not None
        self._writer.writerow({key: row.get(key) for key in self._writer.fieldnames})
        self._csv_file.flush()
        if self.save_frames and rgb is not None:
            camera_dir = self.base_dir / "camera"
            bgr = cv2.cvtColor(np.asarray(rgb)[:, :, :3], cv2.COLOR_RGB2BGR)
            # last_frame.png is always the most recent processed camera frame. The
            # archive stride controls only numbered rgb_*.png files.
            cv2.imwrite(str(camera_dir / "last_frame.png"), bgr)
            if row["step_index"] % self.frame_stride == 0:
                cv2.imwrite(str(camera_dir / f"rgb_{row['step_index']:06d}.png"), bgr)
            if overlay is not None:
                overlay_bgr = cv2.cvtColor(np.asarray(overlay)[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(camera_dir / "last_detection.png"), overlay_bgr)

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
