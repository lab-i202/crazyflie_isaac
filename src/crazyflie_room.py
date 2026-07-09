# crazyflie_room_keyboard_safe.py
#
# Canonical standalone Isaac Sim Crazyflie room simulation with front camera capture.
#
# VERSION: 2026-07-08_replicator_rgb_last_frame_v2_prop_safe
#
# This file intentionally follows the same lifecycle pattern as the user's working test.py:
#   1. Parse script args.
#   2. Strip script args before creating SimulationApp.
#   3. Create SimulationApp.
#   4. Create the USD stage.
#   5. while simulation_app.is_running(): update transforms, capture camera, simulation_app.update(), sleep.
#   6. Close only after loop exits.
#
# Camera implementation:
#   - Creates a normal USD Camera prim attached to /World/CrazyflieRig.
#   - Optionally switches the active viewport to that camera for live viewing.
#   - Uses omni.replicator.core render_product + rgb annotator to retrieve RGB frames.
#   - Overwrites front_camera_recs/last_frame.png whenever a new image is available.
#
# Keyboard control is read from the PowerShell/terminal using msvcrt on Windows.
# Keep the terminal focused and press keys there.
#
# Controls:
#   Arrow Up / W      : forward
#   Arrow Down / S    : backward
#   Arrow Left / A    : lateral left
#   Arrow Right / D   : lateral right
#   R / PageUp        : up
#   F / PageDown      : down
#   Q                 : yaw left
#   E                 : yaw right
#   Space             : hover / stop command
#   Esc               : exit cleanly

import argparse
import csv
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyyaml\n"
        "Install it with:\n"
        "  C:\\isaac-lab\\IsaacLab\\isaaclab.bat -p -m pip install pyyaml pillow"
    ) from exc


SCRIPT_VERSION = "2026-07-08_replicator_rgb_last_frame_v2_prop_safe"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dictionary.")

    return data


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--save-usd", default="")
parser.add_argument(
    "--config",
    default=str(Path(__file__).with_name("crazyflie_room.yaml")),
    help="Path to crazyflie_room.yaml",
)
args, unknown_args = parser.parse_known_args()

# Critical: do not pass this script's --config option into Kit/Isaac.
# Earlier logs showed '--config ...crazyflie_room.yaml' being passed to the base Kit app.
# That is not needed and can cause confusing behavior.
sys.argv = [sys.argv[0]] + unknown_args

CONFIG_PATH = Path(args.config).resolve()
CFG = load_yaml(CONFIG_PATH)
BASE_DIR = CONFIG_PATH.parent

app_cfg = CFG.get("app", {})
headless = bool(args.headless or app_cfg.get("headless", False))
app_width = int(app_cfg.get("width", 1280))
app_height = int(app_cfg.get("height", 720))

# Isaac Sim must start before importing Omniverse / Isaac modules.
from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": headless,
        "width": app_width,
        "height": app_height,
    }
)

import carb
import omni.client
import omni.usd
from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

import isaacsim.core.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path

try:
    from isaacsim.core.utils.viewports import set_camera_view
except Exception:
    set_camera_view = None

try:
    import msvcrt
except ImportError:
    msvcrt = None


Vector3 = Tuple[float, float, float]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw_rad: float


@dataclass
class VelocityCommand:
    vx: float
    vy: float
    vz: float
    yaw_rate_rad_s: float


@dataclass
class AABB:
    name: str
    bmin: Vector3
    bmax: Vector3


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def body_to_world_xy(vx_body: float, vy_body: float, yaw_rad: float) -> Tuple[float, float]:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return c * vx_body - s * vy_body, s * vx_body + c * vy_body


def normalize(v: Vector3) -> Vector3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-12:
        return 0.0, 0.0, 0.0
    return v[0] / n, v[1] / n, v[2] / n


def rotate_body_direction(direction_body: Vector3, yaw_rad: float) -> Vector3:
    x, y, z = direction_body
    wx, wy = body_to_world_xy(x, y, yaw_rad)
    return normalize((wx, wy, z))


def make_aabb(name: str, center: Vector3, size: Vector3) -> AABB:
    return AABB(
        name=name,
        bmin=(
            center[0] - size[0] / 2.0,
            center[1] - size[1] / 2.0,
            center[2] - size[2] / 2.0,
        ),
        bmax=(
            center[0] + size[0] / 2.0,
            center[1] + size[1] / 2.0,
            center[2] + size[2] / 2.0,
        ),
    )


def omni_path_exists(path: str) -> bool:
    result, _entry = omni.client.stat(path)
    return result == omni.client.Result.OK


def add_reference_to_stage_compatible(usd_path: str, prim_path: str) -> None:
    try:
        stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    except TypeError:
        stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)


def find_crazyflie_usd(assets_root_path: str) -> str:
    candidates = [
        f"{assets_root_path}/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
        f"{assets_root_path}/Isaac/Bitcraze/Crazyflie/cf2x.usd",
        f"{assets_root_path}/Bitcraze/Crazyflie/cf2x.usd",
        f"{assets_root_path}/Isaac/Robots/Crazyflie/cf2x.usd",
        f"{assets_root_path}/Isaac/IsaacLab/Robots/Bitcraze/Crazyflie/cf2x.usd",
        f"{assets_root_path}/IsaacLab/Robots/Bitcraze/Crazyflie/cf2x.usd",
    ]

    for path in candidates:
        if omni_path_exists(path):
            return path

    raise RuntimeError(
        "Could not find Crazyflie USD. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def create_box(
    stage,
    path: str,
    center: Vector3,
    size: Vector3,
    color: Vector3,
    collision: bool = True,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3d(*size))

    if collision:
        UsdPhysics.CollisionAPI.Apply(prim)

    return prim


def create_transform_prim(stage, path: str, pose: Pose):
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    translate_op = xformable.AddTranslateOp()
    rotate_op = xformable.AddRotateXYZOp()

    translate_op.Set(Gf.Vec3d(pose.x, pose.y, pose.z))
    rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(pose.yaw_rad)))

    return translate_op, rotate_op


class BoxRoom:
    def __init__(self, cfg: dict):
        room_cfg = cfg["room"]
        self.size = tuple(float(v) for v in room_cfg["size"])
        self.wall_t = float(room_cfg.get("wall_thickness", 0.05))
        self.L = self.size[0]
        self.W = self.size[1]
        self.H = self.size[2]
        self.floor_color = tuple(float(v) for v in room_cfg.get("floor_color", [0.35, 0.35, 0.35]))
        self.wall_color = tuple(float(v) for v in room_cfg.get("wall_color", [0.75, 0.78, 0.82]))
        self.solid_aabbs: List[AABB] = []
        self.obstacle_aabbs: List[AABB] = []

    def build(self, stage, cfg: dict):
        t = self.wall_t
        L = self.L
        W = self.W
        H = self.H

        floor_center = (0.0, 0.0, -t / 2.0)
        floor_size = (L, W, t)
        create_box(stage, "/World/Room/Floor", floor_center, floor_size, self.floor_color)
        self.solid_aabbs.append(make_aabb("floor", floor_center, floor_size))

        # Four walls. No roof.
        walls = [
            ("Wall_PosX", (L / 2.0 + t / 2.0, 0.0, H / 2.0), (t, W, H)),
            ("Wall_NegX", (-L / 2.0 - t / 2.0, 0.0, H / 2.0), (t, W, H)),
            ("Wall_PosY", (0.0, W / 2.0 + t / 2.0, H / 2.0), (L, t, H)),
            ("Wall_NegY", (0.0, -W / 2.0 - t / 2.0, H / 2.0), (L, t, H)),
        ]

        for name, center, size in walls:
            create_box(stage, f"/World/Room/{name}", center, size, self.wall_color)
            self.solid_aabbs.append(make_aabb(name, center, size))

        for obj in cfg.get("obstacles", []):
            name = str(obj["name"])
            center = tuple(float(v) for v in obj["position"])
            size = tuple(float(v) for v in obj["size"])
            color = tuple(float(v) for v in obj.get("color", [0.8, 0.2, 0.2]))

            create_box(stage, f"/World/Obstacles/{name}", center, size, color)
            box = make_aabb(name, center, size)
            self.solid_aabbs.append(box)
            self.obstacle_aabbs.append(box)

    def clip_pose(self, old_pose: Pose, proposed: Pose, radius: float) -> Pose:
        x = clamp(proposed.x, -self.L / 2.0 + radius, self.L / 2.0 - radius)
        y = clamp(proposed.y, -self.W / 2.0 + radius, self.W / 2.0 - radius)
        z = clamp(proposed.z, radius, self.H - radius)

        candidate = Pose(x, y, z, proposed.yaw_rad)

        for box in self.obstacle_aabbs:
            inside = (
                box.bmin[0] - radius <= candidate.x <= box.bmax[0] + radius
                and box.bmin[1] - radius <= candidate.y <= box.bmax[1] + radius
                and box.bmin[2] - radius <= candidate.z <= box.bmax[2] + radius
            )
            if inside:
                return Pose(old_pose.x, old_pose.y, old_pose.z, proposed.yaw_rad)

        return candidate


class MultiRanger:
    def __init__(self, room: BoxRoom, cfg: dict):
        sensor_cfg = cfg["sensors"]["multiranger"]
        self.room = room
        self.max_range = float(sensor_cfg.get("max_range_m", 4.0))
        self.offset_body = tuple(float(v) for v in sensor_cfg.get("origin_offset_body", [0.0, 0.0, 0.0]))
        self.directions_body: Dict[str, Vector3] = {
            "front": (1.0, 0.0, 0.0),
            "back": (-1.0, 0.0, 0.0),
            "left": (0.0, 1.0, 0.0),
            "right": (0.0, -1.0, 0.0),
            "up": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
        }

    @staticmethod
    def ray_aabb_distance(
        origin: Vector3,
        direction: Vector3,
        box: AABB,
        max_range: float,
    ) -> Optional[float]:
        tmin = -float("inf")
        tmax = float("inf")

        for i in range(3):
            o = origin[i]
            d = direction[i]
            mn = box.bmin[i]
            mx = box.bmax[i]

            if abs(d) < 1e-12:
                if o < mn or o > mx:
                    return None
            else:
                t1 = (mn - o) / d
                t2 = (mx - o) / d

                if t1 > t2:
                    t1, t2 = t2, t1

                tmin = max(tmin, t1)
                tmax = min(tmax, t2)

                if tmin > tmax:
                    return None

        if tmax < 0.0:
            return None

        hit = tmin if tmin >= 0.0 else tmax
        if 0.0 <= hit <= max_range:
            return hit
        return None

    def read(self, pose: Pose) -> Dict[str, Optional[float]]:
        off_x, off_y = body_to_world_xy(self.offset_body[0], self.offset_body[1], pose.yaw_rad)
        origin = (pose.x + off_x, pose.y + off_y, pose.z + self.offset_body[2])

        readings: Dict[str, Optional[float]] = {}

        for name, direction_body in self.directions_body.items():
            direction = rotate_body_direction(direction_body, pose.yaw_rad)
            hits = []

            for solid in self.room.solid_aabbs:
                d = self.ray_aabb_distance(origin, direction, solid, self.max_range)
                if d is not None:
                    hits.append(d)

            readings[name] = min(hits) if hits else None

        return readings


class TerminalKeyboardController:
    def __init__(self, cfg: dict):
        kcfg = cfg.get("keyboard", {})
        self.forward = float(kcfg.get("forward_speed_m_s", 0.35))
        self.lateral = float(kcfg.get("lateral_speed_m_s", 0.25))
        self.vertical = float(kcfg.get("vertical_speed_m_s", 0.20))
        self.yaw_rate = math.radians(float(kcfg.get("yaw_rate_deg_s", 55.0)))
        self.command_hold_s = float(kcfg.get("command_hold_s", 0.30))
        self.cmd = VelocityCommand(0.0, 0.0, 0.0, 0.0)
        self.action = "hover"
        self.last_key_time = 0.0
        self.exit_requested = False

    def update(self, now: float) -> Tuple[VelocityCommand, str]:
        if msvcrt is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0), "hover"

        got_key = False

        while msvcrt.kbhit():
            raw = msvcrt.getch()
            got_key = True

            if raw in (b"\x00", b"\xe0"):
                code = msvcrt.getch()
                self._handle_extended_key(code)
            else:
                self._handle_ascii_key(raw)

        if got_key:
            self.last_key_time = now

        if now - self.last_key_time > self.command_hold_s:
            self.cmd = VelocityCommand(0.0, 0.0, 0.0, 0.0)
            self.action = "hover"

        return self.cmd, self.action

    def _handle_extended_key(self, code: bytes):
        # Arrow codes in Windows console: H/P/K/M = up/down/left/right.
        # PageUp/PageDown are I/Q in the same extended-key stream.
        if code == b"H":
            self.cmd = VelocityCommand(self.forward, 0.0, 0.0, 0.0)
            self.action = "forward"
        elif code == b"P":
            self.cmd = VelocityCommand(-self.forward, 0.0, 0.0, 0.0)
            self.action = "back"
        elif code == b"K":
            self.cmd = VelocityCommand(0.0, self.lateral, 0.0, 0.0)
            self.action = "left"
        elif code == b"M":
            self.cmd = VelocityCommand(0.0, -self.lateral, 0.0, 0.0)
            self.action = "right"
        elif code == b"I":
            self.cmd = VelocityCommand(0.0, 0.0, self.vertical, 0.0)
            self.action = "up"
        elif code == b"Q":
            self.cmd = VelocityCommand(0.0, 0.0, -self.vertical, 0.0)
            self.action = "down"

    def _handle_ascii_key(self, raw: bytes):
        c = raw.lower()

        if raw == b"\x1b":
            self.exit_requested = True
            return

        if c == b" ":
            self.cmd = VelocityCommand(0.0, 0.0, 0.0, 0.0)
            self.action = "hover"
        elif c == b"r":
            self.cmd = VelocityCommand(0.0, 0.0, self.vertical, 0.0)
            self.action = "up"
        elif c == b"f":
            self.cmd = VelocityCommand(0.0, 0.0, -self.vertical, 0.0)
            self.action = "down"
        elif c == b"q":
            self.cmd = VelocityCommand(0.0, 0.0, 0.0, self.yaw_rate)
            self.action = "yaw_left"
        elif c == b"e":
            self.cmd = VelocityCommand(0.0, 0.0, 0.0, -self.yaw_rate)
            self.action = "yaw_right"
        elif c == b"w":
            self.cmd = VelocityCommand(self.forward, 0.0, 0.0, 0.0)
            self.action = "forward"
        elif c == b"s":
            self.cmd = VelocityCommand(-self.forward, 0.0, 0.0, 0.0)
            self.action = "back"
        elif c == b"a":
            self.cmd = VelocityCommand(0.0, self.lateral, 0.0, 0.0)
            self.action = "left"
        elif c == b"d":
            self.cmd = VelocityCommand(0.0, -self.lateral, 0.0, 0.0)
            self.action = "right"


class ToFHistoryCSV:
    def __init__(self, base_dir: Path):
        self.output_dir = base_dir / "History_CSV"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = "tof_history_" + datetime.now().strftime("%d_%m_%H-%M-%S") + ".csv"
        self.path = self.output_dir / filename
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "time_s",
                "action",
                "x_m",
                "y_m",
                "z_m",
                "yaw_deg",
                "vx_body_m_s",
                "vy_body_m_s",
                "vz_m_s",
                "yaw_rate_deg_s",
                "tof_front_m",
                "tof_back_m",
                "tof_left_m",
                "tof_right_m",
                "tof_up_m",
                "tof_down_m",
            ]
        )
        self.rows = 0
        print(f"CSV history: {self.path}")

    @staticmethod
    def value(v: Optional[float]) -> str:
        return "" if v is None else f"{v:.6f}"

    def write(
        self,
        t: float,
        action: str,
        pose: Pose,
        cmd: VelocityCommand,
        tof: Dict[str, Optional[float]],
    ):
        self.writer.writerow(
            [
                f"{t:.6f}",
                action,
                f"{pose.x:.6f}",
                f"{pose.y:.6f}",
                f"{pose.z:.6f}",
                f"{math.degrees(pose.yaw_rad):.6f}",
                f"{cmd.vx:.6f}",
                f"{cmd.vy:.6f}",
                f"{cmd.vz:.6f}",
                f"{math.degrees(cmd.yaw_rate_rad_s):.6f}",
                self.value(tof.get("front")),
                self.value(tof.get("back")),
                self.value(tof.get("left")),
                self.value(tof.get("right")),
                self.value(tof.get("up")),
                self.value(tof.get("down")),
            ]
        )
        self.rows += 1
        if self.rows % 30 == 0:
            self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass


def fmt_tof(v: Optional[float]) -> str:
    return "----" if v is None else f"{v:4.2f}"


def create_wall_landmark_light(stage, cfg: dict):
    landmark = cfg["landmark_light"]
    pos = tuple(float(v) for v in landmark["position"])
    color = tuple(float(v) for v in landmark.get("color", [1.0, 0.95, 0.05]))
    radius = float(landmark.get("radius_m", 0.10))

    sphere = UsdGeom.Sphere.Define(stage, "/World/LandmarkLight/Visual")
    sphere.CreateRadiusAttr(radius)
    sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    sphere_xform = UsdGeom.Xformable(sphere.GetPrim())
    sphere_xform.ClearXformOpOrder()
    sphere_xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    light = UsdLux.SphereLight.Define(stage, "/World/LandmarkLight/WallLight")
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(float(landmark.get("light_intensity", 90000.0)))
    light_xform = UsdGeom.Xformable(light.GetPrim())
    light_xform.ClearXformOpOrder()
    light_xform.AddTranslateOp().Set(Gf.Vec3d(*pos))


def try_attach_propellers(stage, drone_path: str):
    """
    Register existing Crazyflie propeller paths without caching UsdGeom.XformOp handles.

    Cached XformOp/schema objects can become invalid after camera render-product updates.
    The previous version crashed here:
        RuntimeError: Accessed schema on invalid prim
    This version re-resolves the prim/op each frame and skips invalid updates.
    """
    prop_dirs = {
        "m1_prop": 1.0,
        "m2_prop": -1.0,
        "m3_prop": 1.0,
        "m4_prop": -1.0,
    }

    prop_infos = []
    missing = []

    for prop_name, direction in prop_dirs.items():
        path = f"{drone_path}/{prop_name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            missing.append(prop_name)
            continue
        prop_infos.append({"name": prop_name, "path": path, "direction": direction})

    if missing:
        print("WARNING: propeller prims not found, propeller spin disabled:", ", ".join(missing))
        return []

    print("Existing Crazyflie propeller paths registered: m1_prop, m2_prop, m3_prop, m4_prop")
    return prop_infos


def get_or_create_prop_spin_op(stage, prop_path: str):
    prim = stage.GetPrimAtPath(prop_path)
    if not prim.IsValid():
        return None

    try:
        xform = UsdGeom.Xformable(prim)
        if not xform:
            return None

        target_name = "xformOp:rotateZ:prop_spin"
        for existing_op in xform.GetOrderedXformOps():
            if str(existing_op.GetName()) == target_name:
                return existing_op

        return xform.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat, "prop_spin")
    except RuntimeError:
        return None


def set_propeller_visibility(stage, prop_infos, visible: bool) -> None:
    if not prop_infos:
        return

    token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible

    for info in prop_infos:
        try:
            prim = stage.GetPrimAtPath(info["path"])
            if not prim.IsValid():
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                continue
            imageable.GetVisibilityAttr().Set(token)
        except Exception:
            continue


def spin_propellers(stage, prop_infos, prop_angle_deg: float, cmd: VelocityCommand, dt: float, cfg: dict) -> float:
    if not prop_infos:
        return prop_angle_deg

    pcfg = cfg["drone"].get("propeller_spin", {})
    if not bool(pcfg.get("enabled", True)):
        return prop_angle_deg

    hover = float(pcfg.get("hover_deg_s", 1800.0))
    gain = float(pcfg.get("command_gain_deg_s", 2200.0))
    cmd_norm = abs(cmd.vx) + abs(cmd.vy) + abs(cmd.vz) + abs(cmd.yaw_rate_rad_s)
    spin_rate = hover + gain * min(1.0, cmd_norm)
    prop_angle_deg = (prop_angle_deg + spin_rate * dt) % 360.0

    invalid_count = 0
    for info in prop_infos:
        op = get_or_create_prop_spin_op(stage, info["path"])
        if op is None:
            invalid_count += 1
            continue
        try:
            op.Set(float(info["direction"]) * prop_angle_deg)
        except RuntimeError:
            invalid_count += 1

    if invalid_count and bool(pcfg.get("print_invalid_propeller_warnings", False)):
        print(f"WARNING: skipped {invalid_count} invalid propeller prim/op update(s).")

    return prop_angle_deg

def apply_velocity_command(pose: Pose, cmd: VelocityCommand, dt: float, room: BoxRoom, radius: float) -> Pose:
    wx, wy = body_to_world_xy(cmd.vx, cmd.vy, pose.yaw_rad)
    proposed = Pose(
        x=pose.x + wx * dt,
        y=pose.y + wy * dt,
        z=pose.z + cmd.vz * dt,
        yaw_rad=wrap_pi(pose.yaw_rad + cmd.yaw_rate_rad_s * dt),
    )
    return room.clip_pose(pose, proposed, radius)


def create_front_camera_prim(stage, rig_path: str, cfg: dict) -> str:
    camera_cfg = cfg.get("camera", {})
    camera_path = str(camera_cfg.get("prim_path", f"{rig_path}/FrontCamera"))

    camera = UsdGeom.Camera.Define(stage, camera_path)
    focal_length = float(camera_cfg.get("focal_length_mm", 3.0))
    horizontal_aperture = float(camera_cfg.get("horizontal_aperture_mm", 4.8))
    vertical_aperture = float(camera_cfg.get("vertical_aperture_mm", 3.6))

    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateClippingRangeAttr(
        Gf.Vec2f(
            float(camera_cfg.get("near_clip_m", 0.03)),
            float(camera_cfg.get("far_clip_m", 20.0)),
        )
    )

    prim = camera.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()

    offset = tuple(float(v) for v in camera_cfg.get("offset_body", [0.06, 0.0, 0.025]))
    rotation = tuple(float(v) for v in camera_cfg.get("rotation_xyz_deg", [90.0, 0.0, -90.0]))

    xform.AddTranslateOp().Set(Gf.Vec3d(*offset))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

    hfov = math.degrees(2.0 * math.atan(horizontal_aperture / (2.0 * focal_length)))
    vfov = math.degrees(2.0 * math.atan(vertical_aperture / (2.0 * focal_length)))

    print(f"Front camera prim created: {camera_path}")
    print(f"Front camera pinhole FOV approx: horizontal={hfov:.1f} deg, vertical={vfov:.1f} deg")

    return camera_path


def switch_active_viewport_to_camera(camera_path: str, cfg: dict, status_lines: List[str]) -> bool:
    live_view_cfg = cfg.get("live_view", {})
    if not bool(live_view_cfg.get("switch_active_viewport_to_front_camera", True)):
        status_lines.append("Live viewport switch disabled by YAML.")
        return False

    if headless:
        status_lines.append("Live viewport switch skipped because --headless is active.")
        return False

    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport_api = get_active_viewport()
        if viewport_api is None:
            status_lines.append("Live viewport switch failed: get_active_viewport() returned None.")
            return False

        viewport_api.camera_path = camera_path
        status_lines.append(f"Active viewport camera set to: {camera_path}")
        print(f"Active viewport camera set to: {camera_path}")
        return True

    except Exception as exc:
        status_lines.append(f"Live viewport switch failed: {repr(exc)}")
        print(f"WARNING: could not switch active viewport to front camera: {exc}")
        return False


class ReplicatorRGBCameraCapture:
    def __init__(self, base_dir: Path, camera_path: str, cfg: dict, stage=None, prop_infos=None):
        self.cfg = cfg.get("camera_capture", {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.camera_path = camera_path
        self.stage = stage
        self.prop_infos = prop_infos or []
        self.hide_propellers_during_capture = bool(self.cfg.get("hide_propellers_during_capture", True))
        self.output_dir = base_dir / str(self.cfg.get("output_dir", "front_camera_recs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.status_path = self.output_dir / "CAMERA_STATUS.txt"
        self.last_frame_path = self.output_dir / "last_frame.png"
        self.tmp_frame_path = self.output_dir / "last_frame.tmp.png"
        self.open_bat_path = self.output_dir / "open_last_frame.bat"

        self.width = int(self.cfg.get("width", 640))
        self.height = int(self.cfg.get("height", 480))
        self.save_every_s = float(self.cfg.get("save_every_s", 0.10))
        self.warmup_frames = int(self.cfg.get("warmup_frames", 120))
        self.rt_subframes = int(self.cfg.get("rt_subframes", 1))
        self.keep_timestamped = bool(self.cfg.get("keep_timestamped", False))
        self.print_every_n_captures = max(1, int(self.cfg.get("print_every_n_captures", 10)))
        self.max_errors_before_disable = int(self.cfg.get("max_errors_before_disable", 20))

        self.rgb_annotator = None
        self.render_product = None
        self.frame_counter = 0
        self.capture_counter = 0
        self.error_counter = 0
        self.last_capture_t = -1.0
        self.last_error = ""

        # Remove stale marker from old code.
        disabled_marker = self.output_dir / "CAMERA_CAPTURE_DISABLED.txt"
        if disabled_marker.exists():
            try:
                disabled_marker.unlink()
            except Exception:
                pass

        self.open_bat_path.write_text(
            "@echo off\n"
            f'start "" "{self.last_frame_path}"\n',
            encoding="utf-8",
        )

        if not self.enabled:
            self._write_status("Camera capture disabled by YAML: camera_capture.enabled=false")
            print("Camera capture disabled by YAML: camera_capture.enabled=false")
            return

        self._initialize_replicator()

    def _write_status(self, text: str):
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.status_path.write_text(
            f"version: {SCRIPT_VERSION}\n"
            f"time: {stamp}\n"
            f"camera_path: {self.camera_path}\n"
            f"resolution: {self.width}x{self.height}\n"
            f"last_frame_path: {self.last_frame_path}\n"
            f"capture_counter: {self.capture_counter}\n"
            f"error_counter: {self.error_counter}\n"
            f"hide_propellers_during_capture: {self.hide_propellers_during_capture}\n"
            f"status: {text}\n",
            encoding="utf-8",
        )

    def _initialize_replicator(self):
        try:
            import omni.replicator.core as rep

            self.rep = rep
            self.render_product = rep.create.render_product(
                self.camera_path,
                (self.width, self.height),
            )

            self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self.rgb_annotator.attach([self.render_product])

            self._write_status("Camera capture initialized. Waiting for warmup frames.")
            print("Camera capture enabled using Replicator RGB annotator.")
            print(f"Last frame will be overwritten at: {self.last_frame_path}")
            print(f"Camera warmup frames: {self.warmup_frames}")
            print(f"Camera save interval: {self.save_every_s:.3f} s")
            print(f"Hide propellers during camera capture: {self.hide_propellers_during_capture}")

        except Exception as exc:
            self.enabled = False
            self.last_error = traceback.format_exc()
            self._write_status(f"Camera capture initialization failed: {repr(exc)}\n{self.last_error}")
            print("ERROR: Camera capture initialization failed.")
            print(self.last_error)

    def capture_if_due(self, sim_t: float) -> None:
        if not self.enabled:
            return

        self.frame_counter += 1

        if self.frame_counter < self.warmup_frames:
            return

        if self.save_every_s > 0.0 and self.last_capture_t >= 0.0:
            if sim_t - self.last_capture_t < self.save_every_s:
                return

        self.last_capture_t = sim_t

        props_hidden = False

        try:
            from PIL import Image
            import numpy as np

            # The onboard camera should not see the vehicle propellers. Hide them only
            # during the render step and restore them immediately afterwards.
            if self.hide_propellers_during_capture and self.stage is not None and self.prop_infos:
                set_propeller_visibility(self.stage, self.prop_infos, visible=False)
                props_hidden = True

            # Synchronous Replicator render step. This is the documented standalone data path
            # for retrieving RGB data from a render product.
            self.rep.orchestrator.step(rt_subframes=self.rt_subframes)

            data = self.rgb_annotator.get_data()

            if isinstance(data, dict):
                if "data" in data:
                    data = data["data"]
                elif "rgb" in data:
                    data = data["rgb"]

            if data is None:
                raise RuntimeError("rgb annotator returned None")

            arr = np.asarray(data)

            if arr.size == 0:
                raise RuntimeError("rgb annotator returned an empty array")

            if arr.ndim != 3 or arr.shape[2] < 3:
                raise RuntimeError(f"unexpected RGB array shape: {arr.shape}")

            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            rgb = arr[:, :, :3]
            img = Image.fromarray(rgb)

            # Atomic overwrite: save temp file, then replace last_frame.png.
            img.save(self.tmp_frame_path)
            os.replace(self.tmp_frame_path, self.last_frame_path)

            self.capture_counter += 1
            self.error_counter = 0

            if self.keep_timestamped:
                timestamped = self.output_dir / ("frame_" + datetime.now().strftime("%d_%m_%H-%M-%S-%f") + ".png")
                img.save(timestamped)

            if self.capture_counter == 1 or self.capture_counter % self.print_every_n_captures == 0:
                print(f"Camera frame #{self.capture_counter} saved: {self.last_frame_path}")

            self._write_status("OK - last_frame.png overwritten successfully.")

        except Exception as exc:
            self.error_counter += 1
            self.last_error = traceback.format_exc()
            self._write_status(f"Camera capture failed: {repr(exc)}\n{self.last_error}")
            print(f"WARNING: camera capture failed ({self.error_counter}/{self.max_errors_before_disable}): {exc}")

            if self.error_counter >= self.max_errors_before_disable:
                self.enabled = False
                self._write_status("DISABLED after repeated camera capture errors.\n" + self.last_error)
                print("ERROR: Camera capture disabled after repeated errors. See CAMERA_STATUS.txt.")

        finally:
            if props_hidden and self.stage is not None and self.prop_infos:
                set_propeller_visibility(self.stage, self.prop_infos, visible=True)

    def close(self):
        # Do not aggressively destroy render products during shutdown; in practice this avoids
        # some native teardown crashes in long-running Kit sessions.
        pass


def main() -> None:
    print(f"CANONICAL_SCRIPT_VERSION={SCRIPT_VERSION}")
    print(f"CONFIG_PATH={CONFIG_PATH}")
    print(f"BASE_DIR={BASE_DIR}")

    front_camera_dir = BASE_DIR / "front_camera_recs"
    front_camera_dir.mkdir(parents=True, exist_ok=True)

    # Clean stale disabled marker from older versions.
    disabled_marker = front_camera_dir / "CAMERA_CAPTURE_DISABLED.txt"
    if disabled_marker.exists():
        try:
            disabled_marker.unlink()
            print(f"Removed stale marker: {disabled_marker}")
        except Exception as exc:
            print(f"WARNING: could not remove stale marker {disabled_marker}: {exc}")

    omni.usd.get_context().new_stage()
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Room")
    UsdGeom.Xform.Define(stage, "/World/Obstacles")
    UsdGeom.Xform.Define(stage, "/World/LandmarkLight")

    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(float(CFG.get("lighting", {}).get("dome_intensity", 800.0)))

    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(float(CFG.get("lighting", {}).get("sun_intensity", 600.0)))
    sun.CreateAngleAttr(float(CFG.get("lighting", {}).get("sun_angle_deg", 0.5)))

    room = BoxRoom(CFG)
    room.build(stage, CFG)
    create_wall_landmark_light(stage, CFG)

    pose_cfg = CFG["drone"]["initial_pose"]
    pose = Pose(
        x=float(pose_cfg.get("x", 0.0)),
        y=float(pose_cfg.get("y", 0.0)),
        z=float(pose_cfg.get("z", 0.7)),
        yaw_rad=math.radians(float(pose_cfg.get("yaw_deg", 0.0))),
    )

    drone_rig_path = "/World/CrazyflieRig"
    drone_translate_op, drone_rotate_op = create_transform_prim(stage, drone_rig_path, pose)

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets root.")

    crazyflie_usd = find_crazyflie_usd(assets_root_path)
    carb.log_info(f"Using Crazyflie asset: {crazyflie_usd}")
    print(f"Crazyflie asset: {crazyflie_usd}")

    crazyflie_prim_path = f"{drone_rig_path}/Crazyflie"
    add_reference_to_stage_compatible(crazyflie_usd, crazyflie_prim_path)

    # Create camera after the rig exists, so the camera is a child of the moving drone rig.
    camera_path = create_front_camera_prim(stage, drone_rig_path, CFG)

    # Let USD references and initial camera/render systems settle.
    startup_warmup_frames = int(CFG.get("runtime", {}).get("startup_warmup_frames", 30))
    for _ in range(startup_warmup_frames):
        simulation_app.update()
        time.sleep(1.0 / 60.0)

    prop_infos = try_attach_propellers(stage, crazyflie_prim_path)
    prop_angle_deg = 0.0

    if set_camera_view is not None:
        set_camera_view(
            eye=CFG.get("viewer", {}).get("eye", [3.0, -3.0, 2.2]),
            target=CFG.get("viewer", {}).get("target", [0.0, 0.0, 0.8]),
            camera_prim_path="/OmniverseKit_Persp",
        )

    status_lines: List[str] = []
    switch_active_viewport_to_camera(camera_path, CFG, status_lines)

    camera_capture = ReplicatorRGBCameraCapture(BASE_DIR, camera_path, CFG, stage=stage, prop_infos=prop_infos)

    if args.save_usd:
        omni.usd.get_context().save_as_stage(args.save_usd)
        print(f"Saved scene to: {args.save_usd}")

    keyboard = TerminalKeyboardController(CFG)
    tof_sensor = MultiRanger(room, CFG)
    csv_logger = ToFHistoryCSV(BASE_DIR) if bool(CFG.get("runtime", {}).get("enable_csv", True)) else None

    fps = float(CFG.get("runtime", {}).get("fps", 60.0))
    dt = 1.0 / fps
    radius = float(CFG["drone"].get("collision_radius_m", 0.12))
    print_every_s = float(CFG.get("logging", {}).get("print_every_s", 0.25))
    csv_every_s = float(CFG.get("logging", {}).get("csv_every_s", 0.05))

    print("\nCrazyflie room simulation started.")
    print("This version follows your working test.py lifecycle.")
    print("Keyboard input is read from PowerShell/terminal, not the Isaac viewport.")
    print("Keep this terminal focused while pressing keys.")
    print("Controls:")
    print("  Arrow Up / W      : forward")
    print("  Arrow Down / S    : backward")
    print("  Arrow Left / A    : lateral left")
    print("  Arrow Right / D   : lateral right")
    print("  R / PageUp        : up")
    print("  F / PageDown      : down")
    print("  Q                 : yaw left")
    print("  E                 : yaw right")
    print("  Space             : hover")
    print("  Esc               : exit cleanly")
    print("\nNo roof is created. ToF is printed and saved to CSV.")
    print(f"Front camera path: {camera_path}")
    print(f"Last camera frame path: {camera_capture.last_frame_path}")
    print(f"Camera status path: {camera_capture.status_path}")
    print(f"Camera capture enabled: {camera_capture.enabled}\n")

    start_time = time.perf_counter()
    last_print_t = -1.0
    last_csv_t = -1.0

    try:
        while simulation_app.is_running():
            loop_t0 = time.perf_counter()
            sim_t = loop_t0 - start_time

            cmd, action = keyboard.update(loop_t0)
            if keyboard.exit_requested:
                print("Esc pressed. Exiting cleanly.")
                break

            pose = apply_velocity_command(pose, cmd, dt, room, radius)
            drone_translate_op.Set(Gf.Vec3d(pose.x, pose.y, pose.z))
            drone_rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(pose.yaw_rad)))
            prop_angle_deg = spin_propellers(stage, prop_infos, prop_angle_deg, cmd, dt, CFG)

            tof = tof_sensor.read(pose)

            if csv_logger is not None and (last_csv_t < 0.0 or sim_t - last_csv_t >= csv_every_s):
                csv_logger.write(sim_t, action, pose, cmd, tof)
                last_csv_t = sim_t

            camera_capture.capture_if_due(sim_t)

            if last_print_t < 0.0 or sim_t - last_print_t >= print_every_s:
                print(
                    f"action={action:9s} "
                    f"pose=({pose.x:+.2f}, {pose.y:+.2f}, {pose.z:+.2f}, yaw={math.degrees(pose.yaw_rad):+.1f}) "
                    f"ToF[m] "
                    f"F={fmt_tof(tof.get('front'))} "
                    f"B={fmt_tof(tof.get('back'))} "
                    f"L={fmt_tof(tof.get('left'))} "
                    f"R={fmt_tof(tof.get('right'))} "
                    f"U={fmt_tof(tof.get('up'))} "
                    f"D={fmt_tof(tof.get('down'))} "
                    f"cam_frames={camera_capture.capture_counter}"
                )
                last_print_t = sim_t

            simulation_app.update()
            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0.0, dt - elapsed))

    finally:
        camera_capture.close()

        if csv_logger is not None:
            csv_logger.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        traceback.print_exc()
        simulation_app.close()
        raise
