from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class Pose:
    x: float
    y: float
    z: float
    yaw_rad: float


@dataclass(slots=True)
class AABB:
    name: str
    bmin: tuple[float, float, float]
    bmax: tuple[float, float, float]

    def contains_inflated(self, point: tuple[float, float, float], radius: float) -> bool:
        return all(self.bmin[i] - radius <= point[i] <= self.bmax[i] + radius for i in range(3))


class RoomGeometry:
    def __init__(self, cfg: dict) -> None:
        room = cfg["environment"]["room"]
        self.sx, self.sy, self.sz = (float(v) for v in room["size_m"])
        self.radius = float(cfg["environment"]["crazyflie"]["collision_radius_m"])
        self.max_range = float(cfg["environment"]["sensors"]["max_range_m"])
        self.obstacles: list[AABB] = []
        for item in cfg["environment"].get("obstacles", []):
            if not item.get("enabled", True) or item.get("kind", "box") not in {"box", "wall"}:
                continue
            px, py, pz = (float(v) for v in item["position_m"])
            ox, oy, oz = (float(v) for v in item["size_m"])
            self.obstacles.append(
                AABB(str(item["name"]), (px - ox / 2, py - oy / 2, pz - oz / 2), (px + ox / 2, py + oy / 2, pz + oz / 2))
            )
        for item in cfg["environment"].get("custom_assets", []):
            if not item.get("enabled", True) or not item.get("collision_aabb_size_m"):
                continue
            px, py, pz = (float(v) for v in item.get("position_m", [0.0, 0.0, 0.0]))
            sx, sy, sz = (float(v) for v in item["collision_aabb_size_m"])
            self.obstacles.append(
                AABB(
                    f"asset:{item.get('name', 'custom')}",
                    (px - sx / 2, py - sy / 2, pz - sz / 2),
                    (px + sx / 2, py + sy / 2, pz + sz / 2),
                )
            )

    def initial_pose(self, seed: int) -> Pose:
        import numpy as np

        agent = self._agent_cfg
        base = agent["initial_pose"]["position_m"]
        yaw_deg = float(agent["initial_pose"]["yaw_deg"])
        randomization = agent.get("spawn_randomization_m", [0.0, 0.0, 0.0])
        rng = np.random.default_rng(seed)
        offsets = [rng.uniform(-float(v), float(v)) for v in randomization]
        return Pose(float(base[0]) + offsets[0], float(base[1]) + offsets[1], float(base[2]) + offsets[2], math.radians(yaw_deg))

    @property
    def _agent_cfg(self) -> dict:
        # Assigned by from_config to keep the constructor compact.
        return self.__dict__["agent_cfg"]

    @classmethod
    def from_config(cls, cfg: dict) -> "RoomGeometry":
        instance = cls(cfg)
        instance.__dict__["agent_cfg"] = cfg["environment"]["crazyflie"]
        return instance

    def integrate_action(self, pose: Pose, action: int, cfg: dict) -> tuple[Pose, bool]:
        actions = cfg["mission"]["actions"]
        dt = float(cfg["mission"]["control_step_s"])
        forward_speed = float(actions["forward_speed_m_s"])
        vertical_speed = float(actions["vertical_speed_m_s"])
        yaw_rate = float(actions["yaw_rate_rad_s"])
        x, y, z, yaw = pose.x, pose.y, pose.z, pose.yaw_rad
        if action == 0:
            x += math.cos(yaw) * forward_speed * dt
            y += math.sin(yaw) * forward_speed * dt
        elif action == 1:
            yaw += yaw_rate * dt
        elif action == 2:
            yaw -= yaw_rate * dt
        elif action == 3:
            z += vertical_speed * dt
        elif action == 4:
            z -= vertical_speed * dt
        else:
            raise ValueError(f"Invalid baseline action {action}")
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        proposed = Pose(x, y, z, yaw)
        collision = self.is_collision(proposed)
        return (Pose(pose.x, pose.y, pose.z, yaw) if collision else proposed), collision

    def is_collision(self, pose: Pose) -> bool:
        r = self.radius
        if pose.x < -self.sx / 2 + r or pose.x > self.sx / 2 - r:
            return True
        if pose.y < -self.sy / 2 + r or pose.y > self.sy / 2 - r:
            return True
        if pose.z < r or pose.z > self.sz - r:
            return True
        return any(item.contains_inflated((pose.x, pose.y, pose.z), r) for item in self.obstacles)

    def ranges(self, pose: Pose) -> dict[str, float]:
        directions = {
            "front": (math.cos(pose.yaw_rad), math.sin(pose.yaw_rad), 0.0),
            "back": (-math.cos(pose.yaw_rad), -math.sin(pose.yaw_rad), 0.0),
            "left": (-math.sin(pose.yaw_rad), math.cos(pose.yaw_rad), 0.0),
            "right": (math.sin(pose.yaw_rad), -math.cos(pose.yaw_rad), 0.0),
            "up": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
        }
        origin = (pose.x, pose.y, pose.z)
        solids = list(self.obstacles) + self._wall_aabbs()
        output: dict[str, float] = {}
        for name, direction in directions.items():
            distances = [self._ray_aabb(origin, direction, box) for box in solids]
            valid = [value for value in distances if value is not None]
            output[name] = min(valid) if valid else self.max_range
        return output

    def range_collision(self, readings: dict[str, float], pose: Pose, cfg: dict) -> bool:
        sensors = cfg["environment"]["sensors"]
        horizontal_threshold = float(sensors["horizontal_collision_threshold_m"])
        if any(readings[name] < horizontal_threshold for name in ("front", "back", "left", "right")):
            return True
        return pose.z >= float(sensors["top_collision_altitude_m"])

    def _wall_aabbs(self) -> list[AABB]:
        t = 0.02
        return [
            AABB("wall_x_pos", (self.sx / 2, -self.sy / 2, 0), (self.sx / 2 + t, self.sy / 2, self.sz)),
            AABB("wall_x_neg", (-self.sx / 2 - t, -self.sy / 2, 0), (-self.sx / 2, self.sy / 2, self.sz)),
            AABB("wall_y_pos", (-self.sx / 2, self.sy / 2, 0), (self.sx / 2, self.sy / 2 + t, self.sz)),
            AABB("wall_y_neg", (-self.sx / 2, -self.sy / 2 - t, 0), (self.sx / 2, -self.sy / 2, self.sz)),
            AABB("floor", (-self.sx / 2, -self.sy / 2, -t), (self.sx / 2, self.sy / 2, 0)),
            AABB("ceiling", (-self.sx / 2, -self.sy / 2, self.sz), (self.sx / 2, self.sy / 2, self.sz + t)),
        ]

    def _ray_aabb(self, origin: tuple[float, float, float], direction: tuple[float, float, float], box: AABB) -> float | None:
        tmin = -float("inf")
        tmax = float("inf")
        for index in range(3):
            o = origin[index]
            d = direction[index]
            lower = box.bmin[index]
            upper = box.bmax[index]
            if abs(d) < 1e-12:
                if o < lower or o > upper:
                    return None
                continue
            t1 = (lower - o) / d
            t2 = (upper - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
        if tmax < 0:
            return None
        hit = tmin if tmin >= 0 else tmax
        return hit if 0 <= hit <= self.max_range else None
