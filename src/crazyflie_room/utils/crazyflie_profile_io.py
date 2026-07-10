# utils/crazyflie_profile_io.py
#
# JSON loading, saving, normalization, and validation for Crazyflie room profiles.
#
# This module must not import Isaac Sim or Omniverse modules. It is safe for Tkinter.

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CRAZYFLIE_ROOM_PROFILE: dict[str, Any] = {
    "schema_version": 1,
    "scenario_id": "crazyflie_room_basic",
    "scene_name": "crazyflie_room_basic",
    "app": {
        "headless": False,
        "renderer": "RealTimePathTracing",
        "width": 1280,
        "height": 720,
        "anti_aliasing": 3,
        "sync_loads": True,
    },
    "room": {
        "enabled": True,
        "size_m": [4.0, 3.0, 2.0],
        "wall_thickness_m": 0.05,
        "has_roof": False,
        "floor_color": [0.35, 0.35, 0.35],
        "wall_color": [0.75, 0.78, 0.82],
        "visual": {
            "floor": {
                "opacity": 1.0,
                "roughness": 0.65,
                "metallic": 0.0,
                "reflectance": 0.18
            },
            "walls": {
                "opacity": 1.0,
                "roughness": 0.55,
                "metallic": 0.0,
                "reflectance": 0.35
            }
        },
    },
    "obstacles": [
        {
            "name": "box_01",
            "enabled": True,
            "kind": "box",
            "position_m": [0.8, 0.4, 0.25],
            "size_m": [0.35, 0.35, 0.5],
            "color": [0.8, 0.25, 0.15],
            "collision": True,
            "negative": False,
            "visual": {
                "opacity": 1.0,
                "roughness": 0.55,
                "metallic": 0.0,
                "reflectance": 0.35,
            },
        }
    ],
    "landmark": {
        "enabled": True,
        "kind": "sphere_light",
        "position_m": [1.8, 0.0, 1.2],
        "radius_m": 0.08,
        "color": [1.0, 0.85, 0.1],
        "intensity": 2500.0,
        "exposure": 0.0,
    },
    "crazyflie": {
        "prim_path": "/World/CrazyflieRig",
        "model_prim_path": "/World/CrazyflieRig/Crazyflie",
        "asset": {
            "mode": "isaac_builtin",
            "local_usd_path": "assets/robots/crazyflie/cf2x.usd",
            "candidates": [
                "Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
                "Isaac/Bitcraze/Crazyflie/cf2x.usd",
                "Bitcraze/Crazyflie/cf2x.usd",
                "Isaac/Robots/Crazyflie/cf2x.usd",
                "Isaac/IsaacLab/Robots/Bitcraze/Crazyflie/cf2x.usd",
                "IsaacLab/Robots/Bitcraze/Crazyflie/cf2x.usd",
            ],
            "allow_placeholder_if_missing": False,
        },
        "initial_pose": {
            "position_m": [-1.45, -0.9, 0.7],
            "rotation_deg": [0.0, 0.0, 0.0],
        },
        "limits": {
            "max_vx_m_s": 0.6,
            "max_vy_m_s": 0.6,
            "max_vz_m_s": 0.4,
            "max_yaw_rate_deg_s": 90.0,
            "collision_radius_m": 0.12,
        },
        "propellers": {
            "animate": True,
            "prim_names": ["m1_prop", "m2_prop", "m3_prop", "m4_prop"],
            "hover_spin_deg_s": 1800.0,
            "command_gain_deg_s": 2200.0,
        },
    },
    "cameras": {
        "onboard": {
            "enabled": True,
            "prim_path": "/World/CrazyflieRig/FrontCamera",
            "position_m": [0.08, 0.0, 0.03],
            "rotation_deg": [0.0, -90.0, 0.0],
            "focal_length_mm": 2.4,
            "horizontal_aperture_mm": 3.2,
            "vertical_aperture_mm": 2.4,
            "clipping_range_m": [0.03, 20.0],
            "resolution": [640, 360],
            "capture_rgb": False,
            "capture_every_n_frames": 1,
        },
        "isometric": {
            "enabled": True,
            "prim_path": "/World/IsometricCamera",
            "position_m": [3.0, -3.0, 2.6],
            "look_at_m": [0.0, 0.0, 0.75],
            "auto_frame_room": True,
            "azimuth_deg": -45.0,
            "elevation_deg": 35.0,
            "distance_scale": 1.85,
            "padding_m": 0.35,
            "focal_length_mm": 24.0,
            "resolution": [1280, 720],
            "capture_rgb": False,
            "capture_every_n_frames": 1,
        },
        "capture": {
            "rt_subframes": 1,
            "camera_params": False,
            "wait_for_render": True
        },
    },
    "runtime": {
        "fps": 60.0,
        "control_mode": "keyboard_terminal",
        "duration_s": 0.0,
        "telemetry_enabled": True,
        "telemetry_output_root": "outputs",
        "write_latest_state_json": True,
        "write_state_history_csv": True,
        "print_every_s": 0.25,
        "csv_every_s": 0.05,
        "save_usd_path": "",
    },
    "future_rl": {
        "enabled": False,
        "num_envs": 1,
        "spacing_m": 5.0,
        "note": "Reserved. Do not use until the single-Crazyflie scene is stable.",
    },
}

CONTROL_MODES = ["keyboard_terminal", "scripted_hover", "static"]
OBSTACLE_KINDS = ["box", "wall", "sphere", "cylinder", "pillar", "floor_patch"]
ASSET_MODES = ["isaac_builtin", "local_usd"]


def load_crazyflie_room_profile(profile_path: Path) -> dict[str, Any]:
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file does not exist: {profile_path}")
    if not profile_path.is_file():
        raise ValueError(f"Profile path is not a file: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    if not isinstance(profile, dict):
        raise ValueError("Crazyflie profile root must be a JSON object.")

    normalize_crazyflie_room_profile(profile)
    validate_crazyflie_room_profile(profile)
    return profile


def save_crazyflie_room_profile(profile_path: Path, profile: dict[str, Any]) -> None:
    normalize_crazyflie_room_profile(profile)
    validate_crazyflie_room_profile(profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    with profile_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)


def default_crazyflie_room_profile() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CRAZYFLIE_ROOM_PROFILE)


def normalize_crazyflie_room_profile(profile: dict[str, Any]) -> None:
    merged = copy.deepcopy(DEFAULT_CRAZYFLIE_ROOM_PROFILE)
    _deep_update(merged, profile)
    profile.clear()
    profile.update(merged)

    profile["scene_name"] = sanitize_filename(str(profile.get("scene_name", "crazyflie_room_basic")))
    profile["scenario_id"] = "crazyflie_room_basic"

    # Normalize obstacle objects without deleting user-defined obstacle entries.
    normalized_obstacles = []
    for index, obstacle in enumerate(profile.get("obstacles", [])):
        default_obstacle = {
            "name": f"obstacle_{index + 1:02d}",
            "enabled": True,
            "kind": "box",
            "position_m": [0.0, 0.0, 0.25],
            "size_m": [0.3, 0.3, 0.5],
            "color": [0.8, 0.25, 0.15],
            "collision": True,
            "negative": False,
            "visual": {
                "opacity": 1.0,
                "roughness": 0.55,
                "metallic": 0.0,
                "reflectance": 0.35,
            },
        }
        if isinstance(obstacle, dict):
            _deep_update(default_obstacle, obstacle)
        normalized_obstacles.append(default_obstacle)
    profile["obstacles"] = normalized_obstacles

    room = profile.setdefault("room", {})
    visual = room.setdefault("visual", {})
    visual.setdefault("floor", {})
    visual.setdefault("walls", {})
    for key, defaults in {
        "floor": {"opacity": 1.0, "roughness": 0.65, "metallic": 0.0, "reflectance": 0.18},
        "walls": {"opacity": 1.0, "roughness": 0.55, "metallic": 0.0, "reflectance": 0.35},
    }.items():
        for field, value in defaults.items():
            visual[key].setdefault(field, value)

    cameras = profile.setdefault("cameras", {})
    cameras.setdefault("capture", {"rt_subframes": 1, "camera_params": False, "wait_for_render": True})
    for camera_key in ["onboard", "isometric"]:
        if camera_key in cameras and isinstance(cameras[camera_key], dict):
            cameras[camera_key].setdefault("capture_every_n_frames", 1)

    if "isometric" in cameras and isinstance(cameras["isometric"], dict):
        isometric = cameras["isometric"]
        isometric.setdefault("auto_frame_room", True)
        isometric.setdefault("azimuth_deg", -45.0)
        isometric.setdefault("elevation_deg", 35.0)
        isometric.setdefault("distance_scale", 1.85)
        isometric.setdefault("padding_m", 0.35)


def sanitize_filename(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\-]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("scene_name cannot be empty.")
    return cleaned


def validate_crazyflie_room_profile(profile: dict[str, Any], project_root: Path | None = None) -> None:
    if not isinstance(profile, dict):
        raise ValueError("Crazyflie profile must be a JSON object.")

    required = [
        "schema_version",
        "scenario_id",
        "scene_name",
        "app",
        "room",
        "obstacles",
        "landmark",
        "crazyflie",
        "cameras",
        "runtime",
    ]
    for key in required:
        if key not in profile:
            raise ValueError(f"Missing required Crazyflie profile key: {key}")

    if profile["scenario_id"] != "crazyflie_room_basic":
        raise ValueError("scenario_id must be 'crazyflie_room_basic'.")
    if not isinstance(profile["scene_name"], str) or not profile["scene_name"].strip():
        raise ValueError("scene_name must be a non-empty string.")

    _validate_app(profile["app"])
    _validate_room(profile["room"])
    _validate_obstacles(profile["obstacles"])
    _validate_landmark(profile["landmark"])
    _validate_crazyflie(profile["crazyflie"], project_root=project_root)
    _validate_cameras(profile["cameras"])
    _validate_runtime(profile["runtime"])


def _validate_app(app: Any) -> None:
    if not isinstance(app, dict):
        raise ValueError("app must be a JSON object.")
    _require_bool(app, "headless", required=False)
    _require_positive_int(app, "width", required=False)
    _require_positive_int(app, "height", required=False)
    _require_positive_int(app, "anti_aliasing", required=False)
    _require_bool(app, "sync_loads", required=False)
    if "renderer" in app and not isinstance(app["renderer"], str):
        raise ValueError("app.renderer must be a string.")


def _validate_room(room: Any) -> None:
    if not isinstance(room, dict):
        raise ValueError("room must be a JSON object.")
    _require_bool(room, "enabled")
    _validate_positive_vector3(room.get("size_m"), "room.size_m")
    _require_number(room, "wall_thickness_m")
    if room["wall_thickness_m"] <= 0.0:
        raise ValueError("room.wall_thickness_m must be greater than zero.")
    _require_bool(room, "has_roof")
    _validate_color3(room.get("floor_color"), "room.floor_color")
    _validate_color3(room.get("wall_color"), "room.wall_color")
    visual = room.get("visual", {})
    if not isinstance(visual, dict):
        raise ValueError("room.visual must be a JSON object.")
    for material_name in ["floor", "walls"]:
        material = visual.get(material_name, {})
        if not isinstance(material, dict):
            raise ValueError(f"room.visual.{material_name} must be a JSON object.")
        _validate_obstacle_visual(material, f"room.visual.{material_name}")


def _validate_obstacles(obstacles: Any) -> None:
    if not isinstance(obstacles, list):
        raise ValueError("obstacles must be a list.")
    names: set[str] = set()
    for index, obstacle in enumerate(obstacles):
        prefix = f"obstacles[{index}]"
        if not isinstance(obstacle, dict):
            raise ValueError(f"{prefix} must be a JSON object.")
        if not isinstance(obstacle.get("name"), str) or not obstacle["name"].strip():
            raise ValueError(f"{prefix}.name must be a non-empty string.")
        if obstacle["name"] in names:
            raise ValueError(f"Duplicate obstacle name: {obstacle['name']}")
        names.add(obstacle["name"])
        _require_bool(obstacle, "enabled")
        if obstacle.get("kind") not in OBSTACLE_KINDS:
            raise ValueError(f"{prefix}.kind must be one of: {', '.join(OBSTACLE_KINDS)}")
        _validate_vector3(obstacle.get("position_m"), f"{prefix}.position_m")
        _validate_positive_vector3(obstacle.get("size_m"), f"{prefix}.size_m")
        _validate_color3(obstacle.get("color"), f"{prefix}.color")
        _require_bool(obstacle, "collision")
        _require_bool(obstacle, "negative", required=False)
        _validate_obstacle_visual(obstacle.get("visual", {}), f"{prefix}.visual")


def _validate_obstacle_visual(visual: Any, name: str) -> None:
    if visual is None:
        return
    if not isinstance(visual, dict):
        raise ValueError(f"{name} must be a JSON object.")
    for key in ["opacity", "roughness", "metallic", "reflectance"]:
        if key not in visual:
            continue
        value = visual[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be a number.")
        if float(value) < 0.0 or float(value) > 1.0:
            raise ValueError(f"{name}.{key} must be between 0 and 1.")


def _validate_landmark(landmark: Any) -> None:
    if not isinstance(landmark, dict):
        raise ValueError("landmark must be a JSON object.")
    _require_bool(landmark, "enabled")
    if landmark.get("kind") != "sphere_light":
        raise ValueError("landmark.kind must be 'sphere_light'.")
    _validate_vector3(landmark.get("position_m"), "landmark.position_m")
    _require_number(landmark, "radius_m")
    if landmark["radius_m"] <= 0.0:
        raise ValueError("landmark.radius_m must be greater than zero.")
    _validate_color3(landmark.get("color"), "landmark.color")
    _require_number(landmark, "intensity")
    if landmark["intensity"] < 0.0:
        raise ValueError("landmark.intensity must be non-negative.")
    _require_number(landmark, "exposure")


def _validate_crazyflie(crazyflie: Any, project_root: Path | None) -> None:
    if not isinstance(crazyflie, dict):
        raise ValueError("crazyflie must be a JSON object.")
    _require_prim_path(crazyflie, "prim_path")
    _require_prim_path(crazyflie, "model_prim_path")

    asset = crazyflie.get("asset")
    if not isinstance(asset, dict):
        raise ValueError("crazyflie.asset must be a JSON object.")
    if asset.get("mode") not in ASSET_MODES:
        raise ValueError(f"crazyflie.asset.mode must be one of: {', '.join(ASSET_MODES)}")
    if not isinstance(asset.get("local_usd_path"), str) or not asset["local_usd_path"].strip():
        raise ValueError("crazyflie.asset.local_usd_path must be a non-empty string.")
    if asset["mode"] == "local_usd" and project_root is not None:
        local_path = _resolve_project_path(project_root, asset["local_usd_path"])
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(
                "Crazyflie local USD asset does not exist. Add the USD file or switch "
                f"crazyflie.asset.mode to 'isaac_builtin'. Missing: {local_path}"
            )
    if not isinstance(asset.get("candidates"), list):
        raise ValueError("crazyflie.asset.candidates must be a list of Isaac asset paths.")
    for index, candidate in enumerate(asset["candidates"]):
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"crazyflie.asset.candidates[{index}] must be a non-empty string.")
    _require_bool(asset, "allow_placeholder_if_missing")

    pose = crazyflie.get("initial_pose")
    if not isinstance(pose, dict):
        raise ValueError("crazyflie.initial_pose must be a JSON object.")
    _validate_vector3(pose.get("position_m"), "crazyflie.initial_pose.position_m")
    _validate_vector3(pose.get("rotation_deg"), "crazyflie.initial_pose.rotation_deg")

    limits = crazyflie.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("crazyflie.limits must be a JSON object.")
    for key in [
        "max_vx_m_s",
        "max_vy_m_s",
        "max_vz_m_s",
        "max_yaw_rate_deg_s",
        "collision_radius_m",
    ]:
        _require_number(limits, key)
        if limits[key] <= 0.0:
            raise ValueError(f"crazyflie.limits.{key} must be greater than zero.")

    propellers = crazyflie.get("propellers")
    if not isinstance(propellers, dict):
        raise ValueError("crazyflie.propellers must be a JSON object.")
    _require_bool(propellers, "animate")
    if not isinstance(propellers.get("prim_names"), list):
        raise ValueError("crazyflie.propellers.prim_names must be a list.")
    for index, name in enumerate(propellers["prim_names"]):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"crazyflie.propellers.prim_names[{index}] must be a non-empty string.")
    _require_number(propellers, "hover_spin_deg_s")
    _require_number(propellers, "command_gain_deg_s")


def _validate_cameras(cameras: Any) -> None:
    if not isinstance(cameras, dict):
        raise ValueError("cameras must be a JSON object.")
    for name in ["onboard", "isometric"]:
        if name not in cameras:
            raise ValueError(f"Missing cameras.{name}")
        if not isinstance(cameras[name], dict):
            raise ValueError(f"cameras.{name} must be a JSON object.")
        _require_bool(cameras[name], "enabled")
        _require_prim_path(cameras[name], "prim_path")
        _validate_vector3(cameras[name].get("position_m"), f"cameras.{name}.position_m")
        _require_number(cameras[name], "focal_length_mm")
        if cameras[name]["focal_length_mm"] <= 0.0:
            raise ValueError(f"cameras.{name}.focal_length_mm must be greater than zero.")
        _validate_resolution(cameras[name].get("resolution"), f"cameras.{name}.resolution")
        _require_bool(cameras[name], "capture_rgb")
        _require_positive_int(cameras[name], "capture_every_n_frames", required=False)

    capture = cameras.get("capture", {})
    if not isinstance(capture, dict):
        raise ValueError("cameras.capture must be a JSON object.")
    _require_positive_int(capture, "rt_subframes", required=False)
    _require_bool(capture, "camera_params", required=False)
    _require_bool(capture, "wait_for_render", required=False)

    _validate_vector3(cameras["onboard"].get("rotation_deg"), "cameras.onboard.rotation_deg")
    _require_number(cameras["onboard"], "horizontal_aperture_mm")
    _require_number(cameras["onboard"], "vertical_aperture_mm")
    _validate_positive_vector2(cameras["onboard"].get("clipping_range_m"), "cameras.onboard.clipping_range_m")
    _validate_vector3(cameras["isometric"].get("look_at_m"), "cameras.isometric.look_at_m")
    _require_bool(cameras["isometric"], "auto_frame_room", required=False)
    for key in ["azimuth_deg", "elevation_deg", "distance_scale", "padding_m"]:
        _require_number(cameras["isometric"], key, required=False)
    if "elevation_deg" in cameras["isometric"]:
        elevation = float(cameras["isometric"]["elevation_deg"])
        if elevation <= -89.0 or elevation >= 89.0:
            raise ValueError("cameras.isometric.elevation_deg must be between -89 and 89 degrees.")
    if "distance_scale" in cameras["isometric"] and float(cameras["isometric"]["distance_scale"]) <= 0.0:
        raise ValueError("cameras.isometric.distance_scale must be greater than zero.")
    if "padding_m" in cameras["isometric"] and float(cameras["isometric"]["padding_m"]) < 0.0:
        raise ValueError("cameras.isometric.padding_m must be zero or positive.")


def _validate_runtime(runtime: Any) -> None:
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a JSON object.")
    _require_number(runtime, "fps")
    if runtime["fps"] <= 0.0:
        raise ValueError("runtime.fps must be greater than zero.")
    if runtime.get("control_mode") not in CONTROL_MODES:
        raise ValueError(f"runtime.control_mode must be one of: {', '.join(CONTROL_MODES)}")
    _require_number(runtime, "duration_s")
    if runtime["duration_s"] < 0.0:
        raise ValueError("runtime.duration_s must be zero or positive. Zero means run until Isaac Sim is closed.")
    _require_bool(runtime, "telemetry_enabled")
    if not isinstance(runtime.get("telemetry_output_root"), str) or not runtime["telemetry_output_root"].strip():
        raise ValueError("runtime.telemetry_output_root must be a non-empty string.")
    _require_bool(runtime, "write_latest_state_json")
    _require_bool(runtime, "write_state_history_csv")
    _require_number(runtime, "print_every_s")
    _require_number(runtime, "csv_every_s")
    if runtime["print_every_s"] <= 0.0:
        raise ValueError("runtime.print_every_s must be greater than zero.")
    if runtime["csv_every_s"] <= 0.0:
        raise ValueError("runtime.csv_every_s must be greater than zero.")
    if not isinstance(runtime.get("save_usd_path"), str):
        raise ValueError("runtime.save_usd_path must be a string. Use an empty string to disable.")


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _require_number(data: dict[str, Any], key: str, required: bool = True) -> None:
    if key not in data:
        if required:
            raise ValueError(f"Missing required key: {key}")
        return
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} must be a number.")


def _require_positive_int(data: dict[str, Any], key: str, required: bool = True) -> None:
    if key not in data:
        if required:
            raise ValueError(f"Missing required key: {key}")
        return
    value = data[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer.")
    if value <= 0:
        raise ValueError(f"{key} must be greater than zero.")


def _require_bool(data: dict[str, Any], key: str, required: bool = True) -> None:
    if key not in data:
        if required:
            raise ValueError(f"Missing required key: {key}")
        return
    if not isinstance(data[key], bool):
        raise ValueError(f"{key} must be true or false.")


def _require_prim_path(data: dict[str, Any], key: str) -> None:
    value = data.get(key)
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{key} must be an absolute USD prim path beginning with '/'.")


def _validate_vector3(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a list of exactly 3 numbers.")
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be a number.")


def _validate_positive_vector3(value: Any, name: str) -> None:
    _validate_vector3(value, name)
    for index, item in enumerate(value):
        if float(item) <= 0.0:
            raise ValueError(f"{name}[{index}] must be greater than zero.")


def _validate_positive_vector2(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a list of exactly 2 positive numbers.")
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be a number.")
        if float(item) <= 0.0:
            raise ValueError(f"{name}[{index}] must be greater than zero.")
    if float(value[1]) <= float(value[0]):
        raise ValueError(f"{name}[1] must be greater than {name}[0].")


def _validate_resolution(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [width, height].")
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be an integer.")
        if item <= 0:
            raise ValueError(f"{name}[{index}] must be greater than zero.")


def _validate_color3(value: Any, name: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be [r, g, b].")
    for index, item in enumerate(value):
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{name}[{index}] must be a number.")
        if float(item) < 0.0 or float(item) > 1.0:
            raise ValueError(f"{name}[{index}] must be between 0 and 1.")
