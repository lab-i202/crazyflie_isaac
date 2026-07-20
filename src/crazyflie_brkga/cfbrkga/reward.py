from __future__ import annotations

from dataclasses import dataclass

from .types import StepOutcome


@dataclass(slots=True)
class MissionState:
    lost_landmark_streak: int = 0


class BaselineReward:
    """Reward and termination parity with the executable Gazebo BRKGA environment."""

    def __init__(self, cfg: dict) -> None:
        mission = cfg["mission"]
        reward = mission["reward"]
        self.front_min = float(mission["front_tof_min_m"])
        self.front_max = float(mission["front_tof_max_m"])
        self.success_front = float(mission["success_front_tof_m"])
        self.lost_max = int(mission["lost_landmark_max_steps"])
        self.max_steps = int(mission["max_steps"])
        self.time_penalty = float(reward["time_penalty"])
        self.step_scale = float(reward["step_penalty_scale"])
        self.collision_penalty = float(reward["collision_penalty"])
        self.lost_penalty = float(reward["lost_landmark_penalty"])
        self.timeout_penalty = float(reward["timeout_penalty"])
        self.failure_terminal_penalty = float(reward["failure_terminal_penalty"])
        self.success_reward = float(reward["success_reward"])
        self.near_target_partial_reward = float(reward["near_target_partial_reward"])

    def normalize_tof(self, value: float) -> float:
        if self.front_max == self.front_min:
            raise ValueError("front_tof_max_m and front_tof_min_m cannot be equal")
        return max(0.0, min(1.0, (float(value) - self.front_min) / (self.front_max - self.front_min)))

    def step(
        self,
        state: MissionState,
        step_index: int,
        front_tof_m: float,
        euclid_norm: float,
        visible: bool,
        colliding: bool,
    ) -> StepOutcome:
        state.lost_landmark_streak = 0 if visible else state.lost_landmark_streak + 1
        terminated = False
        truncated = False
        success = False
        reason = "running"
        reward = 0.0

        if colliding:
            terminated = True
            reason = "collision"
            reward += self.collision_penalty
        elif state.lost_landmark_streak >= self.lost_max:
            terminated = True
            reason = "lost_landmark"
            reward += self.near_target_partial_reward if front_tof_m < self.success_front else self.lost_penalty
        elif visible and front_tof_m < self.success_front:
            terminated = True
            success = True
            reason = "success"
        elif step_index + 1 >= self.max_steps:
            truncated = True
            reason = "timeout"
            reward += self.near_target_partial_reward if front_tof_m < self.success_front else self.timeout_penalty

        front_norm = self.normalize_tof(front_tof_m)
        step_penalty = -self.step_scale * (front_norm + float(euclid_norm))
        reward += step_penalty - self.time_penalty
        if terminated or truncated:
            reward += self.success_reward if success else self.failure_terminal_penalty

        return StepOutcome(
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            success=success,
            reason=reason,
            front_norm=front_norm,
            step_penalty=step_penalty,
            lost_landmark_streak=state.lost_landmark_streak,
        )
