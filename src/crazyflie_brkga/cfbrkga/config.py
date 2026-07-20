from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"


class ConfigError(ValueError):
    pass


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a JSON object.")
    data = copy.deepcopy(data)
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(PROJECT_ROOT)
    validate_config(data)
    return data


def resolve_config_path(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("CRAZYFLIE_BRKGA_CONFIG", "")
    if not raw:
        return DEFAULT_CONFIG_PATH
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2)
        handle.write("\n")
    return destination


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(config.get("_project_root", PROJECT_ROOT)) / candidate).resolve()


def validate_config(cfg: dict[str, Any]) -> None:
    required = ["project", "app", "environment", "camera", "vision", "policy", "brkga", "mission", "data"]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    env = cfg["environment"]
    room = env["room"]
    if len(room["size_m"]) != 3 or any(float(v) <= 0 for v in room["size_m"]):
        raise ConfigError("environment.room.size_m must contain three positive values.")
    if int(env["num_parallel_envs"]) < 1:
        raise ConfigError("environment.num_parallel_envs must be at least 1.")
    for collection_name in ("obstacles", "custom_assets"):
        collection = env.get(collection_name, [])
        if not isinstance(collection, list):
            raise ConfigError(f"environment.{collection_name} must be a JSON list.")
    for index, obstacle in enumerate(env.get("obstacles", [])):
        if not isinstance(obstacle, dict):
            raise ConfigError(f"environment.obstacles[{index}] must be an object.")
        if len(obstacle.get("position_m", [])) != 3 or len(obstacle.get("size_m", [])) != 3:
            raise ConfigError(f"environment.obstacles[{index}] requires position_m and size_m with 3 values.")
    control_backend = env.get("crazyflie", {}).get("control_backend", "kinematic")
    if control_backend != "kinematic":
        raise ConfigError("The first training implementation supports only the kinematic control backend.")
    sensor_backend = env.get("sensors", {}).get("range_backend", "analytic")
    if sensor_backend not in {"analytic", "physx_raycast"}:
        raise ConfigError("environment.sensors.range_backend must be 'analytic' or 'physx_raycast'.")
    for index, asset in enumerate(env.get("custom_assets", [])):
        if not isinstance(asset, dict):
            raise ConfigError(f"environment.custom_assets[{index}] must be an object.")
        if asset.get("enabled", True) and not str(asset.get("usd_path", "")).strip():
            raise ConfigError(f"environment.custom_assets[{index}].usd_path is required when enabled.")
        for key in ("position_m", "rotation_deg", "scale"):
            if len(asset.get(key, [])) != 3:
                raise ConfigError(f"environment.custom_assets[{index}].{key} must have 3 values.")
        if asset.get("collision_aabb_size_m") is not None and len(asset["collision_aabb_size_m"]) != 3:
            raise ConfigError(f"environment.custom_assets[{index}].collision_aabb_size_m must have 3 values.")

    camera = cfg["camera"]
    if int(camera["width"]) < 32 or int(camera["height"]) < 32:
        raise ConfigError("Camera resolution must be at least 32x32.")

    policy = cfg["policy"]
    if list(policy["layers"]) != [2, 128, 128, 5]:
        raise ConfigError("The baseline policy layout must be [2, 128, 128, 5].")
    if float(policy["decoder"]["upper_bound"]) <= float(policy["decoder"]["lower_bound"]):
        raise ConfigError("Policy decoder upper_bound must exceed lower_bound.")

    brkga = cfg["brkga"]
    pop_size = int(brkga["population_size"])
    if pop_size < 3:
        raise ConfigError("BRKGA population_size must be at least 3.")
    elite = float(brkga["elite_fraction"])
    mutant = float(brkga["mutant_fraction"])
    if not 0 < elite < 1:
        raise ConfigError("elite_fraction must be between 0 and 1.")
    if not 0 <= mutant < 1:
        raise ConfigError("mutant_fraction must be in [0, 1).")
    if elite + mutant >= 1:
        raise ConfigError("elite_fraction + mutant_fraction must be below 1.")
    if not 0.5 < float(brkga["elite_inheritance_probability"]) <= 1.0:
        raise ConfigError("elite_inheritance_probability must be in (0.5, 1].")
    if int(brkga["generations"]) < 1:
        raise ConfigError("brkga.generations must be at least 1.")
    seeds = brkga.get("episode_seeds", [])
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("brkga.episode_seeds must contain at least one integer seed.")
    try:
        [int(value) for value in seeds]
    except (TypeError, ValueError) as exc:
        raise ConfigError("brkga.episode_seeds must contain only integers.") from exc

    mission = cfg["mission"]
    if float(mission["control_step_s"]) <= 0:
        raise ConfigError("mission.control_step_s must be positive.")
    if int(mission["max_steps"]) < 1:
        raise ConfigError("mission.max_steps must be at least 1.")
    if int(mission["lost_landmark_max_steps"]) < 1:
        raise ConfigError("mission.lost_landmark_max_steps must be at least 1.")

    data = cfg["data"]
    if data.get("execution_mode") not in {"training", "integrity"}:
        raise ConfigError("data.execution_mode must be 'training' or 'integrity'.")
    if int(data.get("heartbeat_steps", 10)) < 1:
        raise ConfigError("data.heartbeat_steps must be at least 1.")
    integrity = data["integrity"]
    if int(integrity["frame_stride"]) < 1:
        raise ConfigError("data.integrity.frame_stride must be at least 1.")


def config_summary(cfg: dict[str, Any]) -> str:
    return (
        f"project={cfg['project']['name']} | "
        f"envs={cfg['environment']['num_parallel_envs']} | "
        f"population={cfg['brkga']['population_size']} | "
        f"generations={cfg['brkga']['generations']} | "
        f"camera={cfg['camera']['width']}x{cfg['camera']['height']}"
    )
