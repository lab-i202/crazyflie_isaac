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


def _deep_setdefault(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in target:
            target[key] = copy.deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(target[key], dict):
            _deep_setdefault(target[key], value)


def compatibility_defaults() -> dict[str, Any]:
    return {
        "project": {
            "name": "crazyflie_landmark_brkga",
            "note": "Direct BRKGA evolution of baseline 2-128-128-5 Q-network weights.",
        },
        "app": {
            "headless": False,
            "width": 1280,
            "height": 720,
            "renderer": "RayTracedLighting",
        },
        "environment": {
            "num_parallel_envs": 4,
            "env_spacing_m": 6.0,
            "ambient_light_intensity": 350.0,
            "room": {
                "size_m": [4.0, 3.0, 2.0],
                "wall_thickness_m": 0.05,
                "has_roof": False,
                "floor_color": [0.25, 0.25, 0.28],
                "wall_color": [0.65, 0.68, 0.72],
                "floor_material": {
                    "opacity": 1.0,
                    "roughness": 0.75,
                    "metallic": 0.0,
                    "reflectance": 0.25,
                },
                "wall_material": {
                    "opacity": 1.0,
                    "roughness": 0.75,
                    "metallic": 0.0,
                    "reflectance": 0.25,
                },
                "camera_background_rgb": [35, 35, 35],
            },
            "crazyflie": {
                "usd_path": "",
                "collision_radius_m": 0.12,
                "initial_pose": {
                    "position_m": [-1.4, -0.9, 0.7],
                    "yaw_deg": 0.0,
                },
                "spawn_randomization_m": [0.0, 0.0, 0.0],
                "control_backend": "kinematic",
            },
            "landmark": {
                "enabled": True,
                "position_m": [1.55, 0.0, 0.8],
                "rgb_255": [255, 225, 20],
                "radius_m": 0.1,
                "emissive_intensity": 8.0,
                "light_intensity": 2500.0,
                "exposure": 0.0,
            },
            "sensors": {
                "max_range_m": 4.0,
                "horizontal_collision_threshold_m": 0.15,
                "top_collision_altitude_m": 1.82,
                "range_backend": "physx_raycast",
            },
            "obstacles": [],
            "custom_assets": [],
        },
        "camera": {
            "width": 320,
            "height": 240,
            "horizontal_fov_deg": 74.0,
            "focal_length_mm": 24.0,
            "horizontal_aperture_mm": 20.955,
            "local_position_m": [0.08, 0.0, 0.03],
            "local_rotation_deg": [0.0, 0.0, 0.0],
            "warmup_updates": 8,
            "updates_per_control_step": 1,
        },
        "scenario_preview": {
            "enabled": False,
            "refresh_interval_s": 0.25,
            "camera": {
                "auto_frame_room": True,
                "azimuth_deg": 45.0,
                "elevation_deg": 28.0,
                "distance_scale": 1.25,
                "padding_m": 0.5,
                "focal_length_mm": 35.0,
                "look_at_m": [0.0, 0.0, 0.8],
            },
        },
        "vision": {
            "threshold_mode": "hsv",
            "hsv_lower": [15, 80, 170],
            "hsv_upper": [45, 255, 255],
            "brightness_threshold": 220,
            "grayscale_threshold": 220,
            "adaptive_block_size": 21,
            "adaptive_c": 3.0,
            "gaussian_blur_kernel": 0,
            "morphology_kernel": 3,
            "morph_open_kernel": 3,
            "morph_close_kernel": 3,
            "erode_iterations": 0,
            "dilate_iterations": 0,
            "invert_mask": False,
            "min_area_px": 12.0,
            "max_area_fraction": 0.45,
            "min_circularity": 0.35,
            "prefer_brightest_blob": True,
        },
        "policy": {
            "layers": [2, 128, 128, 5],
            "device": "cuda",
            "decoder": {
                "mode": "legacy_linear",
                "lower_bound": -2.0,
                "upper_bound": 10.0,
            },
        },
        "brkga": {
            "population_size": 32,
            "generations": 20,
            "elite_fraction": 0.2,
            "mutant_fraction": 0.3,
            "elite_inheritance_probability": 0.65,
            "random_seed": 42,
            "episode_seeds": [1001],
        },
        "mission": {
            "control_step_s": 0.1,
            "max_steps": 1200,
            "front_tof_min_m": 0.0,
            "front_tof_max_m": 4.0,
            "success_front_tof_m": 0.55,
            "lost_landmark_max_steps": 500,
            "actions": {
                "forward_speed_m_s": 0.1,
                "vertical_speed_m_s": 0.1,
                "yaw_rate_rad_s": 0.05,
            },
            "reward": {
                "step_penalty_scale": 0.5,
                "time_penalty": 0.01,
                "collision_penalty": -20.0,
                "lost_landmark_penalty": -10.0,
                "timeout_penalty": -2.0,
                "failure_terminal_penalty": -3.0,
                "success_reward": 10.0,
                "near_target_partial_reward": 0.5,
                "worker_error_fitness": -1000000.0,
            },
        },
        "data": {
            "output_root": "outputs/experiments",
            "execution_mode": "training",
            "integrity": {
                "save_raw_data": False,
                "save_frames": False,
                "frame_stride": 5,
                "store_steps_in_database": False,
            },
            "heartbeat_steps": 10,
            "latest_mirror": {
                "enabled": True,
                "folder_name": "",
                "database_sync_interval_s": 2.0,
                "copy_csv": True,
                "copy_checkpoints": True,
            },
        },
    }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Configuration root must be a JSON object.")
    data = copy.deepcopy(data)
    _apply_compatibility_defaults(data)
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(PROJECT_ROOT)
    validate_config(data)
    return data


def _normalize_obstacle(obstacle: dict[str, Any], index: int) -> None:
    obstacle.setdefault("name", f"obstacle_{index:03d}")
    obstacle.setdefault("enabled", True)
    obstacle.setdefault("kind", "box")
    obstacle.setdefault("position_m", [0.0, 0.0, 0.25])
    obstacle.setdefault("size_m", [0.4, 0.4, 0.5])
    obstacle.setdefault("color", [0.8, 0.2, 0.2])
    obstacle.setdefault("collision", True)
    obstacle.setdefault("opacity", 1.0)
    obstacle.setdefault("roughness", 0.65)
    obstacle.setdefault("metallic", 0.0)
    obstacle.setdefault("reflectance", 0.25)


def _normalize_custom_asset(asset: dict[str, Any], index: int) -> None:
    asset.setdefault("name", f"asset_{index:03d}")
    asset.setdefault("enabled", True)
    asset.setdefault("usd_path", "")
    asset.setdefault("position_m", [0.0, 0.0, 0.0])
    asset.setdefault("rotation_deg", [0.0, 0.0, 0.0])
    asset.setdefault("scale", [1.0, 1.0, 1.0])
    asset.setdefault("collision_aabb_size_m", None)


def _apply_compatibility_defaults(data: dict[str, Any]) -> None:
    _deep_setdefault(data, compatibility_defaults())
    room = data["environment"]["room"]
    if "floor_opacity" in room:
        room["floor_material"]["opacity"] = room.pop("floor_opacity")
    if "floor_roughness" in room:
        room["floor_material"]["roughness"] = room.pop("floor_roughness")
    if "floor_metallic" in room:
        room["floor_material"]["metallic"] = room.pop("floor_metallic")
    if "floor_reflectance" in room:
        room["floor_material"]["reflectance"] = room.pop("floor_reflectance")
    if "wall_opacity" in room:
        room["wall_material"]["opacity"] = room.pop("wall_opacity")
    if "wall_roughness" in room:
        room["wall_material"]["roughness"] = room.pop("wall_roughness")
    if "wall_metallic" in room:
        room["wall_material"]["metallic"] = room.pop("wall_metallic")
    if "wall_reflectance" in room:
        room["wall_material"]["reflectance"] = room.pop("wall_reflectance")

    vision = data["vision"]
    legacy_kernel = max(1, int(vision.get("morphology_kernel", 3)))
    vision.setdefault("morph_open_kernel", legacy_kernel)
    vision.setdefault("morph_close_kernel", legacy_kernel)

    for index, obstacle in enumerate(data["environment"].get("obstacles", [])):
        if isinstance(obstacle, dict):
            _normalize_obstacle(obstacle, index)
    for index, asset in enumerate(data["environment"].get("custom_assets", [])):
        if isinstance(asset, dict):
            _normalize_custom_asset(asset, index)


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


def _validate_vec3(value: Any, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three values.")
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must contain numeric values.") from exc
    if positive and any(number <= 0 for number in numbers):
        raise ConfigError(f"{name} values must be positive.")


def _validate_color(value: Any, name: str, maximum: float) -> None:
    _validate_vec3(value, name)
    if any(float(item) < 0 or float(item) > maximum for item in value):
        raise ConfigError(f"{name} values must be between 0 and {maximum}.")


def _validate_unit_material(material: dict[str, Any], name: str) -> None:
    for key in ("opacity", "roughness", "metallic", "reflectance"):
        value = float(material[key])
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"{name}.{key} must be between 0 and 1.")


def validate_config(cfg: dict[str, Any]) -> None:
    required = [
        "project",
        "app",
        "environment",
        "camera",
        "scenario_preview",
        "vision",
        "policy",
        "brkga",
        "mission",
        "data",
    ]
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ConfigError(f"Missing configuration sections: {', '.join(missing)}")

    env = cfg["environment"]
    room = env["room"]
    _validate_vec3(room["size_m"], "environment.room.size_m", positive=True)
    if float(room["wall_thickness_m"]) <= 0:
        raise ConfigError("environment.room.wall_thickness_m must be positive.")
    _validate_color(room["floor_color"], "environment.room.floor_color", 1.0)
    _validate_color(room["wall_color"], "environment.room.wall_color", 1.0)
    _validate_unit_material(room["floor_material"], "environment.room.floor_material")
    _validate_unit_material(room["wall_material"], "environment.room.wall_material")
    if int(env["num_parallel_envs"]) < 1:
        raise ConfigError("environment.num_parallel_envs must be at least 1.")
    if float(env["env_spacing_m"]) <= 0:
        raise ConfigError("environment.env_spacing_m must be positive.")

    crazyflie = env["crazyflie"]
    _validate_vec3(crazyflie["initial_pose"]["position_m"], "environment.crazyflie.initial_pose.position_m")
    _validate_vec3(crazyflie["spawn_randomization_m"], "environment.crazyflie.spawn_randomization_m")
    if float(crazyflie["collision_radius_m"]) <= 0:
        raise ConfigError("environment.crazyflie.collision_radius_m must be positive.")
    control_backend = crazyflie.get("control_backend", "kinematic")
    if control_backend != "kinematic":
        raise ConfigError("The current training implementation supports only the kinematic control backend.")

    landmark = env["landmark"]
    _validate_vec3(landmark["position_m"], "environment.landmark.position_m")
    _validate_color(landmark["rgb_255"], "environment.landmark.rgb_255", 255.0)
    if float(landmark["radius_m"]) <= 0:
        raise ConfigError("environment.landmark.radius_m must be positive.")
    if float(landmark["light_intensity"]) < 0 or float(landmark["emissive_intensity"]) < 0:
        raise ConfigError("Landmark light and emissive intensities cannot be negative.")

    sensor_backend = env.get("sensors", {}).get("range_backend", "analytic")
    if sensor_backend not in {"analytic", "physx_raycast"}:
        raise ConfigError("environment.sensors.range_backend must be 'analytic' or 'physx_raycast'.")

    obstacle_kinds = {"box", "wall", "sphere", "cylinder", "pillar", "floor_patch"}
    obstacles = env.get("obstacles", [])
    if not isinstance(obstacles, list):
        raise ConfigError("environment.obstacles must be a JSON list.")
    for index, obstacle in enumerate(obstacles):
        if not isinstance(obstacle, dict):
            raise ConfigError(f"environment.obstacles[{index}] must be an object.")
        if obstacle.get("kind", "box") not in obstacle_kinds:
            raise ConfigError(f"environment.obstacles[{index}].kind is unsupported.")
        _validate_vec3(obstacle.get("position_m"), f"environment.obstacles[{index}].position_m")
        _validate_vec3(obstacle.get("size_m"), f"environment.obstacles[{index}].size_m", positive=True)
        _validate_color(obstacle.get("color"), f"environment.obstacles[{index}].color", 1.0)
        _validate_unit_material(obstacle, f"environment.obstacles[{index}]")

    assets = env.get("custom_assets", [])
    if not isinstance(assets, list):
        raise ConfigError("environment.custom_assets must be a JSON list.")
    for index, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ConfigError(f"environment.custom_assets[{index}] must be an object.")
        if asset.get("enabled", True) and not str(asset.get("usd_path", "")).strip():
            raise ConfigError(f"environment.custom_assets[{index}].usd_path is required when enabled.")
        for key in ("position_m", "rotation_deg", "scale"):
            _validate_vec3(asset.get(key), f"environment.custom_assets[{index}].{key}")
        if asset.get("collision_aabb_size_m") is not None:
            _validate_vec3(
                asset["collision_aabb_size_m"],
                f"environment.custom_assets[{index}].collision_aabb_size_m",
                positive=True,
            )

    camera = cfg["camera"]
    if int(camera["width"]) < 32 or int(camera["height"]) < 32:
        raise ConfigError("Camera resolution must be at least 32x32.")
    if float(camera["focal_length_mm"]) <= 0 or float(camera["horizontal_aperture_mm"]) <= 0:
        raise ConfigError("Camera focal length and aperture must be positive.")
    _validate_vec3(camera["local_position_m"], "camera.local_position_m")
    _validate_vec3(camera["local_rotation_deg"], "camera.local_rotation_deg")

    preview = cfg["scenario_preview"]
    if float(preview["refresh_interval_s"]) < 0.05:
        raise ConfigError("scenario_preview.refresh_interval_s must be at least 0.05 seconds.")
    preview_camera = preview["camera"]
    if float(preview_camera["distance_scale"]) <= 0:
        raise ConfigError("scenario_preview.camera.distance_scale must be positive.")
    _validate_vec3(preview_camera["look_at_m"], "scenario_preview.camera.look_at_m")

    vision = cfg["vision"]
    allowed_modes = {"hsv", "brightness", "hsv_or_brightness", "grayscale", "adaptive"}
    if vision["threshold_mode"] not in allowed_modes:
        raise ConfigError(f"vision.threshold_mode must be one of: {', '.join(sorted(allowed_modes))}.")
    _validate_color(vision["hsv_lower"], "vision.hsv_lower", 255.0)
    _validate_color(vision["hsv_upper"], "vision.hsv_upper", 255.0)
    if not 0 <= int(vision["hsv_lower"][0]) <= 179 or not 0 <= int(vision["hsv_upper"][0]) <= 179:
        raise ConfigError("HSV hue values must be between 0 and 179.")
    for key in ("gaussian_blur_kernel", "morph_open_kernel", "morph_close_kernel"):
        value = int(vision[key])
        if value < 0:
            raise ConfigError(f"vision.{key} cannot be negative.")
    if int(vision["adaptive_block_size"]) < 3:
        raise ConfigError("vision.adaptive_block_size must be at least 3.")
    if float(vision["min_area_px"]) < 0:
        raise ConfigError("vision.min_area_px cannot be negative.")
    if not 0 < float(vision["max_area_fraction"]) <= 1:
        raise ConfigError("vision.max_area_fraction must be in (0, 1].")
    if not 0 <= float(vision["min_circularity"]) <= 1:
        raise ConfigError("vision.min_circularity must be between 0 and 1.")

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
    mirror = data.get("latest_mirror", {})
    if float(mirror.get("database_sync_interval_s", 2.0)) < 0.25:
        raise ConfigError("data.latest_mirror.database_sync_interval_s must be at least 0.25 seconds.")
    folder_name = str(mirror.get("folder_name", "")).strip()
    if folder_name and ("/" in folder_name or "\\" in folder_name):
        raise ConfigError("data.latest_mirror.folder_name must be a folder name, not a path.")
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
