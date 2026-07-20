from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from .debug_recorder import DebugEpisodeRecorder
from .geometry import Pose, RoomGeometry
from .image_processing import LuminousBlobDetector
from .model import BatchedQNetwork
from .reward import BaselineReward, MissionState
from .types import EvaluationResult, Individual


class MockBatchEvaluator:
    """Deterministic software-only evaluator used to validate the full training pipeline.

    It generates a synthetic onboard camera image, runs the real blob detector, uses
    the same two network inputs and executes the same five actions. It is not a
    replacement for Isaac Sim; it is the mandatory smoke test for BRKGA, SQLite,
    CSV, image processing and dashboard data flow.
    """

    backend_name = "mock"

    def __init__(self, cfg: dict, run_output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = run_output_dir
        self.geometry = RoomGeometry.from_config(cfg)
        self.detector = LuminousBlobDetector.from_config(cfg)
        self.reward_model = BaselineReward(cfg)

    def evaluate_batch(
        self,
        population: list[Individual],
        episode_seed: int,
        evaluation_ids: list[int],
        generation_index: int,
        step_callback=None,
    ) -> list[EvaluationResult]:
        weights = np.stack([item.decoded_weights for item in population])
        network = BatchedQNetwork(weights, self.cfg["policy"].get("device", "cuda"))
        return [
            self._evaluate_one(item, episode_seed, evaluation_ids[index], generation_index, network, index, step_callback)
            for index, item in enumerate(population)
        ]

    def _evaluate_one(self, item, seed, evaluation_id, generation_index, network, network_row, step_callback):
        started = time.perf_counter()
        pose = self.geometry.initial_pose(seed + item.index_in_generation * 1009)
        mission_state = MissionState()
        episode_reward = 0.0
        visible_steps = 0
        collision_count = 0
        invalid_frames = 0
        euclidean_values: list[float] = []
        front_values: list[float] = []
        final_blob = self.detector._missing()
        action_counts = np.zeros(5, dtype=np.int64)
        debug_cfg = self.cfg["data"]["integrity"]
        debug_enabled = self.cfg["data"]["execution_mode"] == "integrity" and bool(debug_cfg["save_raw_data"])
        debug_dir = self.output_dir / "debug" / f"generation_{generation_index:04d}" / f"evaluation_{evaluation_id:08d}"
        recorder = DebugEpisodeRecorder(debug_dir, debug_enabled, bool(debug_cfg["save_frames"]), int(debug_cfg["frame_stride"]))
        reason = "timeout"
        success = False
        steps = 0
        try:
            for step_index in range(int(self.cfg["mission"]["max_steps"])):
                rgb = self._render_camera(pose)
                blob = self.detector.detect(rgb)
                final_blob = blob
                readings = self.geometry.ranges(pose)
                front = float(readings["front"])
                observation = np.array([[front, blob.euclidean_norm]], dtype=np.float32)
                # Select only the correct network row while keeping the batched implementation tested.
                full_obs = np.zeros((network.batch_size, 2), dtype=np.float32)
                full_obs[network_row] = observation[0]
                action = int(network.actions(full_obs)[network_row])
                action_counts[action] += 1
                next_pose, geometric_collision = self.geometry.integrate_action(pose, action, self.cfg)
                next_readings = self.geometry.ranges(next_pose)
                colliding = geometric_collision or self.geometry.range_collision(next_readings, next_pose, self.cfg)
                next_rgb = self._render_camera(next_pose)
                next_blob = self.detector.detect(next_rgb)
                next_front = float(next_readings["front"])
                outcome = self.reward_model.step(mission_state, step_index, next_front, next_blob.euclidean_norm, next_blob.visible, colliding)
                episode_reward += outcome.reward
                pose = next_pose
                visible_steps += int(next_blob.visible)
                collision_count += int(colliding)
                euclidean_values.append(next_blob.euclidean_norm)
                front_values.append(next_front)
                steps = step_index + 1
                reason = outcome.reason
                success = outcome.success
                row = {
                    "step_index": step_index,
                    "simulation_time_s": steps * float(self.cfg["mission"]["control_step_s"]),
                    "camera_frame_index": step_index,
                    "action": action,
                    "reward": outcome.reward,
                    "front_tof_m": next_front,
                    "euclid_norm": next_blob.euclidean_norm,
                    "visible": next_blob.visible,
                    "blob_x": next_blob.centroid_x,
                    "blob_y": next_blob.centroid_y,
                    "x_m": pose.x,
                    "y_m": pose.y,
                    "z_m": pose.z,
                    "yaw_rad": pose.yaw_rad,
                    "collision": colliding,
                    "reason": reason,
                }
                if step_callback is not None:
                    step_callback(evaluation_id, row)
                recorder.write_step(row, next_rgb, self.detector.overlay(next_rgb, next_blob))
                final_blob = next_blob
                if outcome.terminated or outcome.truncated:
                    break
        finally:
            recorder.close()
        wall_time = time.perf_counter() - started
        return EvaluationResult(
            individual_id=item.individual_id,
            generation=item.generation,
            env_index=network_row,
            episode_seed=seed,
            reward=float(episode_reward),
            status="finished",
            reason=reason,
            success=success,
            steps=steps,
            simulated_time_s=steps * float(self.cfg["mission"]["control_step_s"]),
            wall_time_s=wall_time,
            collision_count=collision_count,
            visible_steps=visible_steps,
            invalid_camera_frames=invalid_frames,
            mean_euclid_norm=float(np.mean(euclidean_values)) if euclidean_values else 1.0,
            min_front_tof_m=float(np.min(front_values)) if front_values else self.geometry.max_range,
            final_front_tof_m=float(front_values[-1]) if front_values else self.geometry.max_range,
            final_euclid_norm=float(final_blob.euclidean_norm),
            debug_dir=str(debug_dir) if debug_enabled else "",
            extra={"backend": "mock", "action_counts": action_counts.tolist()},
        )

    def _render_camera(self, pose: Pose) -> np.ndarray:
        camera = self.cfg["camera"]
        width = int(camera["width"])
        height = int(camera["height"])
        image = np.zeros((height, width, 3), dtype=np.uint8)
        background = self.cfg["environment"]["room"].get("camera_background_rgb", [35, 35, 35])
        image[:] = np.array(background, dtype=np.uint8)
        landmark = self.cfg["environment"]["landmark"]
        lx, ly, lz = (float(v) for v in landmark["position_m"])
        dx = lx - pose.x
        dy = ly - pose.y
        dz = lz - pose.z
        forward = math.cos(pose.yaw_rad) * dx + math.sin(pose.yaw_rad) * dy
        lateral = -math.sin(pose.yaw_rad) * dx + math.cos(pose.yaw_rad) * dy
        if forward <= 0.02:
            return image
        hfov = math.radians(float(camera["horizontal_fov_deg"]))
        vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * height / width)
        angle_x = math.atan2(lateral, forward)
        angle_y = math.atan2(dz, forward)
        if abs(angle_x) > hfov / 2 or abs(angle_y) > vfov / 2:
            return image
        px = width / 2 - angle_x / (hfov / 2) * width / 2
        py = height / 2 - angle_y / (vfov / 2) * height / 2
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        radius = max(3, int(float(landmark["radius_m"]) * width / max(distance, 0.05) * 1.5))
        color = tuple(int(v) for v in landmark["rgb_255"])
        cv2.circle(image, (int(round(px)), int(round(py))), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return image
