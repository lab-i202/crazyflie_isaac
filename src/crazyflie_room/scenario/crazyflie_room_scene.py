# scenario/crazyflie_room_scene.py
#
# Profile-driven Crazyflie room scenario.
#
# Important import rule:
#   This module imports Isaac/Omniverse modules inside run_crazyflie_room_scene().
#   The launcher must create SimulationApp before this runner is imported/executed.

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.telemetry_io import TelemetryWriter


Vector3 = tuple[float, float, float]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass
class VelocityCommand:
    vx_m_s: float
    vy_m_s: float
    vz_m_s: float
    yaw_rate_deg_s: float
    label: str


@dataclass
class AABB:
    name: str
    bmin: Vector3
    bmax: Vector3

    def contains_inflated(self, point: Vector3, radius: float) -> bool:
        x, y, z = point
        return (
            self.bmin[0] - radius <= x <= self.bmax[0] + radius
            and self.bmin[1] - radius <= y <= self.bmax[1] + radius
            and self.bmin[2] - radius <= z <= self.bmax[2] + radius
        )


class BoxRoom:
    def __init__(self, cfg: dict[str, Any]) -> None:
        room_cfg = cfg["room"]
        size = room_cfg["size_m"]
        self.size_x = float(size[0])
        self.size_y = float(size[1])
        self.size_z = float(size[2])
        self.enabled = bool(room_cfg.get("enabled", True))
        self.obstacle_aabbs: list[AABB] = []

    @property
    def x_min(self) -> float:
        return -0.5 * self.size_x

    @property
    def x_max(self) -> float:
        return 0.5 * self.size_x

    @property
    def y_min(self) -> float:
        return -0.5 * self.size_y

    @property
    def y_max(self) -> float:
        return 0.5 * self.size_y

    @property
    def z_min(self) -> float:
        return 0.0

    @property
    def z_max(self) -> float:
        return self.size_z

    def clip_pose(self, previous: Pose, proposed: Pose, radius: float) -> Pose:
        x = clamp(proposed.x, self.x_min + radius, self.x_max - radius)
        y = clamp(proposed.y, self.y_min + radius, self.y_max - radius)
        z = clamp(proposed.z, self.z_min + radius, self.z_max - radius)

        candidate = Pose(
            x=x,
            y=y,
            z=z,
            roll_deg=proposed.roll_deg,
            pitch_deg=proposed.pitch_deg,
            yaw_deg=wrap_degrees(proposed.yaw_deg),
        )

        # Conservative obstacle handling: if the next body-center enters an inflated obstacle AABB,
        # reject position but keep yaw. This is deliberate; it is stable and easy to debug.
        for aabb in self.obstacle_aabbs:
            if aabb.contains_inflated((candidate.x, candidate.y, candidate.z), radius):
                return Pose(
                    x=previous.x,
                    y=previous.y,
                    z=previous.z,
                    roll_deg=previous.roll_deg,
                    pitch_deg=previous.pitch_deg,
                    yaw_deg=candidate.yaw_deg,
                )

        return candidate


class TerminalKeyboardController:
    """Terminal keyboard controller.

    This intentionally does not use viewport keyboard subscriptions because the previous
    working Crazyflie scripts were more stable when the terminal owned input.

    Controls:
        W / Arrow Up       forward body +X
        S / Arrow Down     backward
        A / Arrow Left     lateral left
        D / Arrow Right    lateral right
        R / PageUp         up
        F / PageDown       down
        Q                  yaw left
        E                  yaw right
        Space              hover / zero command
        Esc                request exit
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import msvcrt  # type: ignore
        except ImportError:
            msvcrt = None

        self.msvcrt = msvcrt
        limits = cfg["crazyflie"]["limits"]
        self.max_vx = float(limits["max_vx_m_s"])
        self.max_vy = float(limits["max_vy_m_s"])
        self.max_vz = float(limits["max_vz_m_s"])
        self.max_yaw_rate = float(limits["max_yaw_rate_deg_s"])
        self.exit_requested = False
        self._last_keys: list[str] = []

    def get_command(self) -> VelocityCommand:
        if self.msvcrt is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, "hover_no_msvcrt")

        keys = self._read_available_keys()
        self._last_keys = keys

        vx = 0.0
        vy = 0.0
        vz = 0.0
        yaw = 0.0
        labels: list[str] = []

        for key in keys:
            if key in {"esc"}:
                self.exit_requested = True
                labels.append("exit")
            elif key in {"w", "arrow_up"}:
                vx += self.max_vx
                labels.append("forward")
            elif key in {"s", "arrow_down"}:
                vx -= self.max_vx
                labels.append("backward")
            elif key in {"a", "arrow_left"}:
                vy += self.max_vy
                labels.append("left")
            elif key in {"d", "arrow_right"}:
                vy -= self.max_vy
                labels.append("right")
            elif key in {"r", "page_up"}:
                vz += self.max_vz
                labels.append("up")
            elif key in {"f", "page_down"}:
                vz -= self.max_vz
                labels.append("down")
            elif key == "q":
                yaw += self.max_yaw_rate
                labels.append("yaw_left")
            elif key == "e":
                yaw -= self.max_yaw_rate
                labels.append("yaw_right")
            elif key == "space":
                vx = vy = vz = yaw = 0.0
                labels = ["hover"]

        if not labels:
            labels = ["hover"]

        return VelocityCommand(
            vx_m_s=clamp(vx, -self.max_vx, self.max_vx),
            vy_m_s=clamp(vy, -self.max_vy, self.max_vy),
            vz_m_s=clamp(vz, -self.max_vz, self.max_vz),
            yaw_rate_deg_s=clamp(yaw, -self.max_yaw_rate, self.max_yaw_rate),
            label="+".join(labels),
        )

    def _read_available_keys(self) -> list[str]:
        assert self.msvcrt is not None
        keys: list[str] = []

        while self.msvcrt.kbhit():
            raw = self.msvcrt.getwch()
            if raw in ("\x00", "\xe0"):
                if not self.msvcrt.kbhit():
                    continue
                code = self.msvcrt.getwch()
                keys.append(
                    {
                        "H": "arrow_up",
                        "P": "arrow_down",
                        "K": "arrow_left",
                        "M": "arrow_right",
                        "I": "page_up",
                        "Q": "page_down",
                    }.get(code, f"special_{ord(code)}")
                )
            else:
                if raw == "\x1b":
                    keys.append("esc")
                elif raw == " ":
                    keys.append("space")
                else:
                    keys.append(raw.lower())

        return keys


class ScriptedController:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.exit_requested = False

    def get_command(self) -> VelocityCommand:
        return VelocityCommand(0.0, 0.0, 0.0, 0.0, self.mode)


def run_crazyflie_room_scene(
    simulation_app: Any,
    scene_config: dict[str, Any],
    project_root: Path,
    profile_path: Path | None = None,
) -> None:
    # Isaac/Omniverse imports must happen only after SimulationApp has been created.
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

    print("=" * 100)
    print("Starting Crazyflie room scenario")
    print("=" * 100)
    print(f"Scene name: {scene_config['scene_name']}")
    print(f"Profile path: {profile_path if profile_path else '<in-memory profile>'}")
    print("Keyboard mode uses the terminal, not the Isaac viewport. Keep PowerShell focused for keys.")
    print("Controls: W/S/A/D, arrows, R/F, PageUp/PageDown, Q/E, Space, Esc")
    print("=" * 100)

    output_root = resolve_project_path(project_root, scene_config["runtime"]["telemetry_output_root"])
    scene_name = str(scene_config["scene_name"])
    telemetry = TelemetryWriter(
        output_root=output_root,
        scene_name=scene_name,
        enabled=bool(scene_config["runtime"].get("telemetry_enabled", True)),
        write_latest_state_json=bool(scene_config["runtime"].get("write_latest_state_json", True)),
        write_state_history_csv=bool(scene_config["runtime"].get("write_state_history_csv", True)),
    )

    try:
        context = omni.usd.get_context()
        context.new_stage()
        simulation_app.update()

        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Room")
        UsdGeom.Xform.Define(stage, "/World/Obstacles")
        UsdGeom.Xform.Define(stage, "/World/Landmark")

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(800.0)

        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(600.0)
        sun.CreateAngleAttr(0.5)

        room = BoxRoom(scene_config)
        build_room(stage, scene_config, room, Gf, UsdGeom)
        build_obstacles(stage, scene_config, room, Gf, UsdGeom)
        build_landmark(stage, scene_config, Gf, UsdLux, UsdGeom)

        crazyflie_cfg = scene_config["crazyflie"]
        pose = make_initial_pose(crazyflie_cfg)
        translate_op, rotate_op = create_transform_prim(
            stage=stage,
            path=str(crazyflie_cfg["prim_path"]),
            pose=pose,
            Gf=Gf,
            UsdGeom=UsdGeom,
        )

        asset_path = resolve_crazyflie_asset_path(
            scene_config=scene_config,
            project_root=project_root,
            omni_client=omni.client,
            get_assets_root_path=get_assets_root_path,
        )
        carb.log_info(f"Using Crazyflie asset: {asset_path}")
        print(f"Crazyflie asset: {asset_path}")

        add_reference_to_stage_compatible(
            stage_utils=stage_utils,
            usd_path=asset_path,
            prim_path=str(crazyflie_cfg["model_prim_path"]),
        )

        # Let the referenced USD load through normal app updates before scanning child prims.
        for _ in range(10):
            simulation_app.update()
            time.sleep(1.0 / 60.0)

        prop_ops = attach_propeller_ops(
            stage=stage,
            crazyflie_cfg=crazyflie_cfg,
            UsdGeom=UsdGeom,
        )
        prop_angle_deg = 0.0

        create_cameras(
            stage=stage,
            cfg=scene_config,
            Gf=Gf,
            UsdGeom=UsdGeom,
            set_camera_view=set_camera_view,
        )
        write_camera_capture_disabled_notes(output_root=output_root, scene_name=scene_name)

        save_usd_path = str(scene_config["runtime"].get("save_usd_path", "")).strip()
        if save_usd_path:
            resolved_save_path = resolve_project_path(project_root, save_usd_path)
            resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
            context.save_as_stage(str(resolved_save_path))
            print(f"Saved generated USD stage: {resolved_save_path}")

        telemetry.write_metadata(
            {
                "scenario_id": scene_config["scenario_id"],
                "scene_name": scene_name,
                "profile_path": str(profile_path) if profile_path else "",
                "crazyflie_asset_path": asset_path,
                "telemetry_output_dir": str(telemetry.output_dir),
                "camera_capture": "disabled_by_default",
                "control_mode": scene_config["runtime"].get("control_mode"),
            }
        )

        control_mode = str(scene_config["runtime"].get("control_mode", "keyboard_terminal"))
        if control_mode == "keyboard_terminal":
            controller: Any = TerminalKeyboardController(scene_config)
        else:
            controller = ScriptedController(control_mode)

        fps = float(scene_config["runtime"].get("fps", 60.0))
        dt = 1.0 / fps
        duration_s = float(scene_config["runtime"].get("duration_s", 0.0))
        print_every_s = float(scene_config["runtime"].get("print_every_s", 0.25))
        csv_every_s = float(scene_config["runtime"].get("csv_every_s", 0.05))
        collision_radius = float(crazyflie_cfg["limits"].get("collision_radius_m", 0.12))

        start_time = time.perf_counter()
        last_print_s = -1e9
        last_csv_s = -1e9
        frame = 0

        print("\nCrazyflie room running.")
        print(f"Telemetry directory: {telemetry.output_dir}")
        print("Close Isaac Sim or press Esc in the terminal to stop.\n")

        while simulation_app.is_running():
            now = time.perf_counter()
            elapsed_s = now - start_time
            if duration_s > 0.0 and elapsed_s >= duration_s:
                print(f"Duration reached: {duration_s:.3f} s")
                break

            cmd = controller.get_command()
            if getattr(controller, "exit_requested", False):
                print("Exit requested from keyboard.")
                break

            proposed = integrate_pose(pose, cmd, dt)

            if control_mode == "scripted_hover":
                proposed.z = pose.z + 0.02 * math.sin(2.0 * math.pi * elapsed_s / 2.0)

            pose = room.clip_pose(previous=pose, proposed=proposed, radius=collision_radius)
            apply_pose(translate_op=translate_op, rotate_op=rotate_op, pose=pose, Gf=Gf)

            prop_angle_deg = spin_propellers(
                prop_ops=prop_ops,
                prop_angle_deg=prop_angle_deg,
                cmd=cmd,
                dt=dt,
                cfg=scene_config,
            )

            write_csv_row = elapsed_s - last_csv_s >= csv_every_s
            if write_csv_row:
                last_csv_s = elapsed_s

            state = make_state_payload(
                scene_name=scene_name,
                frame=frame,
                elapsed_s=elapsed_s,
                pose=pose,
                cmd=cmd,
                control_mode=control_mode,
                asset_path=asset_path,
            )
            telemetry.write_state(state, write_csv_row=write_csv_row)

            if elapsed_s - last_print_s >= print_every_s:
                print(
                    f"t={elapsed_s:8.3f}s frame={frame:06d} "
                    f"pos=({pose.x:+.3f}, {pose.y:+.3f}, {pose.z:+.3f}) "
                    f"yaw={pose.yaw_deg:+.1f} cmd={cmd.label}"
                )
                last_print_s = elapsed_s

            simulation_app.update()
            frame += 1
            sleep_s = dt - (time.perf_counter() - now)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    finally:
        telemetry.close()


def make_initial_pose(crazyflie_cfg: dict[str, Any]) -> Pose:
    position = crazyflie_cfg["initial_pose"]["position_m"]
    rotation = crazyflie_cfg["initial_pose"]["rotation_deg"]
    return Pose(
        x=float(position[0]),
        y=float(position[1]),
        z=float(position[2]),
        roll_deg=float(rotation[0]),
        pitch_deg=float(rotation[1]),
        yaw_deg=float(rotation[2]),
    )


def create_transform_prim(stage: Any, path: str, pose: Pose, Gf: Any, UsdGeom: Any) -> tuple[Any, Any]:
    prim = UsdGeom.Xform.Define(stage, path)
    xform = UsdGeom.Xformable(prim.GetPrim())
    xform.ClearXformOpOrder()
    translate_op = xform.AddTranslateOp()
    rotate_op = xform.AddRotateXYZOp()
    apply_pose(translate_op=translate_op, rotate_op=rotate_op, pose=pose, Gf=Gf)
    return translate_op, rotate_op


def apply_pose(translate_op: Any, rotate_op: Any, pose: Pose, Gf: Any) -> None:
    translate_op.Set(Gf.Vec3d(pose.x, pose.y, pose.z))
    rotate_op.Set(Gf.Vec3f(pose.roll_deg, pose.pitch_deg, pose.yaw_deg))


def integrate_pose(pose: Pose, cmd: VelocityCommand, dt: float) -> Pose:
    yaw_rad = math.radians(pose.yaw_deg)
    wx = math.cos(yaw_rad) * cmd.vx_m_s - math.sin(yaw_rad) * cmd.vy_m_s
    wy = math.sin(yaw_rad) * cmd.vx_m_s + math.cos(yaw_rad) * cmd.vy_m_s
    return Pose(
        x=pose.x + wx * dt,
        y=pose.y + wy * dt,
        z=pose.z + cmd.vz_m_s * dt,
        roll_deg=pose.roll_deg,
        pitch_deg=pose.pitch_deg,
        yaw_deg=wrap_degrees(pose.yaw_deg + cmd.yaw_rate_deg_s * dt),
    )


def build_room(stage: Any, cfg: dict[str, Any], room: BoxRoom, Gf: Any, UsdGeom: Any) -> None:
    if not room.enabled:
        return

    room_cfg = cfg["room"]
    sx, sy, sz = [float(v) for v in room_cfg["size_m"]]
    t = float(room_cfg["wall_thickness_m"])
    floor_color = tuple(float(v) for v in room_cfg["floor_color"])
    wall_color = tuple(float(v) for v in room_cfg["wall_color"])

    create_box(
        stage=stage,
        path="/World/Room/Floor",
        position=(0.0, 0.0, -0.5 * t),
        size=(sx, sy, t),
        color=floor_color,
        Gf=Gf,
        UsdGeom=UsdGeom,
    )
    create_box(stage, "/World/Room/Wall_X_Pos", (0.5 * sx + 0.5 * t, 0.0, 0.5 * sz), (t, sy, sz), wall_color, Gf, UsdGeom)
    create_box(stage, "/World/Room/Wall_X_Neg", (-0.5 * sx - 0.5 * t, 0.0, 0.5 * sz), (t, sy, sz), wall_color, Gf, UsdGeom)
    create_box(stage, "/World/Room/Wall_Y_Pos", (0.0, 0.5 * sy + 0.5 * t, 0.5 * sz), (sx, t, sz), wall_color, Gf, UsdGeom)
    create_box(stage, "/World/Room/Wall_Y_Neg", (0.0, -0.5 * sy - 0.5 * t, 0.5 * sz), (sx, t, sz), wall_color, Gf, UsdGeom)

    if bool(room_cfg.get("has_roof", False)):
        create_box(stage, "/World/Room/Roof", (0.0, 0.0, sz + 0.5 * t), (sx, sy, t), wall_color, Gf, UsdGeom)


def build_obstacles(stage: Any, cfg: dict[str, Any], room: BoxRoom, Gf: Any, UsdGeom: Any) -> None:
    for obstacle in cfg.get("obstacles", []):
        if not bool(obstacle.get("enabled", True)):
            continue

        name = sanitize_prim_name(str(obstacle["name"]))
        kind = str(obstacle.get("kind", "box"))
        path = f"/World/Obstacles/{name}"
        position = tuple(float(v) for v in obstacle["position_m"])
        size = tuple(float(v) for v in obstacle["size_m"])
        color = tuple(float(v) for v in obstacle["color"])

        if kind == "box":
            create_box(stage, path, position, size, color, Gf, UsdGeom)
        elif kind == "sphere":
            create_sphere(stage, path, position, size, color, Gf, UsdGeom)
        elif kind == "cylinder":
            create_cylinder(stage, path, position, size, color, Gf, UsdGeom)
        else:
            raise ValueError(f"Unsupported obstacle kind: {kind}")

        if bool(obstacle.get("collision", True)):
            half = (0.5 * size[0], 0.5 * size[1], 0.5 * size[2])
            room.obstacle_aabbs.append(
                AABB(
                    name=str(obstacle["name"]),
                    bmin=(position[0] - half[0], position[1] - half[1], position[2] - half[2]),
                    bmax=(position[0] + half[0], position[1] + half[1], position[2] + half[2]),
                )
            )


def build_landmark(stage: Any, cfg: dict[str, Any], Gf: Any, UsdLux: Any, UsdGeom: Any) -> None:
    landmark = cfg["landmark"]
    if not bool(landmark.get("enabled", True)):
        return

    path = "/World/Landmark/LuminousSphere"
    position = tuple(float(v) for v in landmark["position_m"])
    color = tuple(float(v) for v in landmark["color"])
    radius = float(landmark["radius_m"])

    light = UsdLux.SphereLight.Define(stage, path)
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(float(landmark["intensity"]))
    light.CreateExposureAttr(float(landmark.get("exposure", 0.0)))
    light.CreateColorAttr(Gf.Vec3f(*color))

    xform = UsdGeom.Xformable(light.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # A small visible non-light sphere helps in rasterized viewport modes.
    create_sphere(
        stage=stage,
        path="/World/Landmark/VisibleLandmark",
        position=position,
        size=(radius * 2.0, radius * 2.0, radius * 2.0),
        color=color,
        Gf=Gf,
        UsdGeom=UsdGeom,
    )


def create_box(stage: Any, path: str, position: Vector3, size: Vector3, color: Vector3, Gf: Any, UsdGeom: Any) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    set_display_color(prim, color, Gf, UsdGeom)


def create_sphere(stage: Any, path: str, position: Vector3, size: Vector3, color: Vector3, Gf: Any, UsdGeom: Any) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.5)
    prim = sphere.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    set_display_color(prim, color, Gf, UsdGeom)


def create_cylinder(stage: Any, path: str, position: Vector3, size: Vector3, color: Vector3, Gf: Any, UsdGeom: Any) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(0.5)
    cylinder.CreateHeightAttr(1.0)
    prim = cylinder.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
    set_display_color(prim, color, Gf, UsdGeom)


def set_display_color(prim: Any, color: Vector3, Gf: Any, UsdGeom: Any) -> None:
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def resolve_crazyflie_asset_path(
    scene_config: dict[str, Any],
    project_root: Path,
    omni_client: Any,
    get_assets_root_path: Any,
) -> str:
    asset_cfg = scene_config["crazyflie"]["asset"]
    mode = str(asset_cfg.get("mode", "isaac_builtin"))

    local_path = resolve_project_path(project_root, str(asset_cfg["local_usd_path"]))

    if mode == "local_usd":
        if local_path.exists() and local_path.is_file():
            return str(local_path)
        raise RuntimeError(
            "Crazyflie local USD asset was requested but was not found.\n"
            f"Expected file: {local_path}\n"
            "Add cf2x.usd there or set crazyflie.asset.mode to 'isaac_builtin'."
        )

    assets_root_path = get_assets_root_path()
    tried: list[str] = []

    if assets_root_path is not None:
        for candidate in asset_cfg.get("candidates", []):
            candidate = str(candidate).strip()
            if not candidate:
                continue
            full_path = join_omniverse_path(str(assets_root_path), candidate)
            tried.append(full_path)
            if omni_path_exists(omni_client, full_path):
                return full_path

    # Local fallback check is useful when the Isaac asset server is unavailable.
    tried.append(str(local_path))
    if local_path.exists() and local_path.is_file():
        return str(local_path)

    if bool(asset_cfg.get("allow_placeholder_if_missing", False)):
        return "__PLACEHOLDER__"

    raise RuntimeError(
        "Could not find the Crazyflie USD asset.\n\n"
        "Tried these Isaac/local paths:\n"
        + "\n".join(f"  - {path}" for path in tried)
        + "\n\nFix: either make sure Isaac Sim assets are reachable, or copy cf2x.usd to:\n"
        f"  {local_path}\n"
        "Then set crazyflie.asset.mode to 'local_usd' if needed."
    )


def omni_path_exists(omni_client: Any, path: str) -> bool:
    try:
        result, _entry = omni_client.stat(path)
    except Exception:
        return False
    return result == omni_client.Result.OK


def join_omniverse_path(root: str, child: str) -> str:
    return root.rstrip("/") + "/" + child.lstrip("/")


def add_reference_to_stage_compatible(stage_utils: Any, usd_path: str, prim_path: str) -> None:
    if usd_path == "__PLACEHOLDER__":
        raise RuntimeError(
            "Placeholder Crazyflie asset path was requested, but placeholder geometry is not enabled "
            "in this Step 1 runner. Add a real USD asset instead."
        )

    try:
        stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    except TypeError:
        stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)


def attach_propeller_ops(stage: Any, crazyflie_cfg: dict[str, Any], UsdGeom: Any) -> list[tuple[Any, float]]:
    prop_cfg = crazyflie_cfg.get("propellers", {})
    if not bool(prop_cfg.get("animate", True)):
        return []

    model_path = str(crazyflie_cfg["model_prim_path"]).rstrip("/")
    prop_names = [str(name) for name in prop_cfg.get("prim_names", [])]
    directions = [1.0, -1.0, 1.0, -1.0]
    ops: list[tuple[Any, float]] = []
    missing: list[str] = []

    for index, name in enumerate(prop_names):
        path = f"{model_path}/{name}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            missing.append(path)
            continue

        xform = UsdGeom.Xformable(prim)
        target_name = "xformOp:rotateZ:prop_spin"
        op = None
        for existing_op in xform.GetOrderedXformOps():
            if str(existing_op.GetName()) == target_name:
                op = existing_op
                break
        if op is None:
            op = xform.AddRotateZOp(UsdGeom.XformOp.PrecisionFloat, "prop_spin")
        ops.append((op, directions[index % len(directions)]))

    if missing:
        print("WARNING: some Crazyflie propeller prims were not found. Propeller animation will be partial.")
        for path in missing:
            print(f"  missing: {path}")

    return ops


def spin_propellers(
    prop_ops: list[tuple[Any, float]],
    prop_angle_deg: float,
    cmd: VelocityCommand,
    dt: float,
    cfg: dict[str, Any],
) -> float:
    if not prop_ops:
        return prop_angle_deg

    prop_cfg = cfg["crazyflie"]["propellers"]
    hover = float(prop_cfg.get("hover_spin_deg_s", 1800.0))
    gain = float(prop_cfg.get("command_gain_deg_s", 2200.0))
    command_magnitude = abs(cmd.vx_m_s) + abs(cmd.vy_m_s) + abs(cmd.vz_m_s) + abs(cmd.yaw_rate_deg_s) / 90.0
    speed = hover + gain * min(command_magnitude, 1.0)
    prop_angle_deg = wrap_degrees(prop_angle_deg + speed * dt)

    for op, direction in prop_ops:
        op.Set(direction * prop_angle_deg)

    return prop_angle_deg


def create_cameras(stage: Any, cfg: dict[str, Any], Gf: Any, UsdGeom: Any, set_camera_view: Any) -> None:
    cameras = cfg["cameras"]

    onboard = cameras["onboard"]
    if bool(onboard.get("enabled", True)):
        camera = UsdGeom.Camera.Define(stage, str(onboard["prim_path"]))
        camera.CreateFocalLengthAttr(float(onboard["focal_length_mm"]))
        camera.CreateHorizontalApertureAttr(float(onboard["horizontal_aperture_mm"]))
        camera.CreateVerticalApertureAttr(float(onboard["vertical_aperture_mm"]))
        near_far = onboard.get("clipping_range_m", [0.03, 20.0])
        camera.CreateClippingRangeAttr(Gf.Vec2f(float(near_far[0]), float(near_far[1])))

        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in onboard["position_m"]]))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*[float(v) for v in onboard["rotation_deg"]]))
        print(f"Onboard camera prim created: {onboard['prim_path']}")

    isometric = cameras["isometric"]
    if bool(isometric.get("enabled", True)):
        camera = UsdGeom.Camera.Define(stage, str(isometric["prim_path"]))
        camera.CreateFocalLengthAttr(float(isometric["focal_length_mm"]))
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in isometric["position_m"]]))

        if set_camera_view is not None:
            try:
                set_camera_view(
                    eye=[float(v) for v in isometric["position_m"]],
                    target=[float(v) for v in isometric["look_at_m"]],
                    camera_prim_path=str(isometric["prim_path"]),
                )
            except Exception as exc:
                print(f"WARNING: set_camera_view failed for isometric camera: {exc}")

        print(f"Isometric camera prim created: {isometric['prim_path']}")


def write_camera_capture_disabled_notes(output_root: Path, scene_name: str) -> None:
    base = output_root / scene_name
    for folder in ["camera_onboard", "camera_isometric"]:
        folder_path = base / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        (folder_path / "CAMERA_CAPTURE_DISABLED.txt").write_text(
            "RGB capture is intentionally disabled in Step 1. The camera prims are created, "
            "but capture should be enabled only after the base scene is stable.\n",
            encoding="utf-8",
        )


def make_state_payload(
    scene_name: str,
    frame: int,
    elapsed_s: float,
    pose: Pose,
    cmd: VelocityCommand,
    control_mode: str,
    asset_path: str,
) -> dict[str, Any]:
    return {
        "scene_name": scene_name,
        "frame": frame,
        "time_s": elapsed_s,
        "pose": {
            "position_m": [pose.x, pose.y, pose.z],
            "rotation_deg": [pose.roll_deg, pose.pitch_deg, pose.yaw_deg],
        },
        "command": {
            "label": cmd.label,
            "vx_m_s": cmd.vx_m_s,
            "vy_m_s": cmd.vy_m_s,
            "vz_m_s": cmd.vz_m_s,
            "yaw_rate_deg_s": cmd.yaw_rate_deg_s,
        },
        "control_mode": control_mode,
        "crazyflie_asset_path": asset_path,
    }


def resolve_project_path(project_root: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def sanitize_prim_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "unnamed"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_degrees(value: float) -> float:
    wrapped = (value + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped
