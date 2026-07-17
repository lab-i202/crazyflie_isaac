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
import shutil
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
        self._held_command = VelocityCommand(0.0, 0.0, 0.0, 0.0, "hover")
        self._held_until = 0.0
        self.command_hold_s = max(float(cfg.get("runtime", {}).get("keyboard_command_hold_s", 0.20)), 0.0)

    def get_command(self) -> VelocityCommand:
        if self.msvcrt is None:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, "hover_no_msvcrt")

        keys = self._read_available_keys()
        self._last_keys = keys

        if not keys and self.command_hold_s > 0.0 and time.perf_counter() < self._held_until:
            return self._held_command

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

        command = VelocityCommand(
            vx_m_s=clamp(vx, -self.max_vx, self.max_vx),
            vy_m_s=clamp(vy, -self.max_vy, self.max_vy),
            vz_m_s=clamp(vz, -self.max_vz, self.max_vz),
            yaw_rate_deg_s=clamp(yaw, -self.max_yaw_rate, self.max_yaw_rate),
            label="+".join(labels),
        )
        if command.label != "hover" and self.command_hold_s > 0.0:
            self._held_command = command
            self._held_until = time.perf_counter() + self.command_hold_s
        return command

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
    def __init__(self, mode: str, cfg: dict[str, Any]) -> None:
        self.mode = mode
        self.cfg = cfg
        self.exit_requested = False
        self.start_time = time.perf_counter()
        limits = cfg["crazyflie"]["limits"]
        self.max_vx = float(limits["max_vx_m_s"])
        self.max_vy = float(limits["max_vy_m_s"])
        self.max_vz = float(limits["max_vz_m_s"])

    def get_command(self) -> VelocityCommand:
        if self.mode == "static":
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, "static")
        if self.mode == "scripted_loop":
            return self._scripted_loop_command()
        return VelocityCommand(0.0, 0.0, 0.0, 0.0, self.mode)

    def _scripted_loop_command(self) -> VelocityCommand:
        loop_cfg = self.cfg.get("runtime", {}).get("scripted_loop", {})
        segment_s = max(float(loop_cfg.get("segment_s", 1.5)), 1e-6)
        hover_s = max(float(loop_cfg.get("hover_s", 0.5)), 0.0)
        phases = [
            ("forward", self.max_vx, 0.0, 0.0),
            ("backward", -self.max_vx, 0.0, 0.0),
            ("left", 0.0, self.max_vy, 0.0),
            ("right", 0.0, -self.max_vy, 0.0),
            ("up", 0.0, 0.0, self.max_vz),
            ("down", 0.0, 0.0, -self.max_vz),
        ]
        phase_period = segment_s + hover_s
        cycle_s = len(phases) * phase_period
        elapsed = time.perf_counter() - self.start_time
        t = elapsed % cycle_s
        phase_index = int(t // phase_period)
        phase_t = t - phase_index * phase_period
        label, vx, vy, vz = phases[phase_index]
        if phase_t >= segment_s:
            return VelocityCommand(0.0, 0.0, 0.0, 0.0, f"hover_after_{label}")
        return VelocityCommand(vx, vy, vz, 0.0, f"scripted_{label}")


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
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

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
    camera_capture_session: dict[str, Any] | None = None

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
        UsdGeom.Xform.Define(stage, "/World/CustomAssets")
        UsdGeom.Xform.Define(stage, "/World/Landmark")
        UsdGeom.Xform.Define(stage, "/World/Materials")

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(800.0)

        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(600.0)
        sun.CreateAngleAttr(0.5)

        room = BoxRoom(scene_config)
        build_room(stage, scene_config, room, Gf, UsdGeom, UsdShade, Sdf)
        build_obstacles(stage, scene_config, room, Gf, UsdGeom, UsdShade, Sdf)
        build_custom_assets(stage, scene_config, project_root, stage_utils, Gf, UsdGeom)
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
        camera_capture_session = create_camera_capture_session(
            cfg=scene_config,
            output_root=output_root,
            scene_name=scene_name,
            carb=carb,
        )
        if camera_capture_session is None:
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
                "custom_asset_count": len([a for a in scene_config.get("custom_assets", []) if a.get("enabled", False)]),
                "telemetry_output_dir": str(telemetry.output_dir),
                "camera_capture": "enabled" if camera_capture_session is not None else "disabled",
                "control_mode": scene_config["runtime"].get("control_mode"),
            }
        )

        control_mode = str(scene_config["runtime"].get("control_mode", "keyboard_terminal"))
        if control_mode == "keyboard_terminal":
            controller: Any = TerminalKeyboardController(scene_config)
        else:
            controller = ScriptedController(control_mode, scene_config)

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
            step_camera_capture(camera_capture_session, frame)
            frame += 1
            sleep_s = dt - (time.perf_counter() - now)
            if sleep_s > 0.0:
                time.sleep(sleep_s)

    finally:
        cleanup_camera_capture(camera_capture_session)
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


def build_room(stage: Any, cfg: dict[str, Any], room: BoxRoom, Gf: Any, UsdGeom: Any, UsdShade: Any, Sdf: Any) -> None:
    if not room.enabled:
        return

    room_cfg = cfg["room"]
    sx, sy, sz = [float(v) for v in room_cfg["size_m"]]
    t = float(room_cfg["wall_thickness_m"])
    floor_color = tuple(float(v) for v in room_cfg["floor_color"])
    wall_color = tuple(float(v) for v in room_cfg["wall_color"])
    room_visual = room_cfg.get("visual", {})
    floor_visual = room_visual.get("floor", {"opacity": 1.0, "roughness": 0.65, "metallic": 0.0, "reflectance": 0.18})
    wall_visual = room_visual.get("walls", {"opacity": 1.0, "roughness": 0.55, "metallic": 0.0, "reflectance": 0.35})

    create_box(
        stage=stage,
        path="/World/Room/Floor",
        position=(0.0, 0.0, -0.5 * t),
        size=(sx, sy, t),
        color=floor_color,
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        Sdf=Sdf,
        visual=floor_visual,
    )
    # Step 5 intentionally stops using negative obstacles for wall CSG/cutout work.
    # Use Custom assets for designed wall geometry instead.
    cutouts = {"x_pos": [], "x_neg": [], "y_pos": [], "y_neg": []}
    create_x_wall_with_cutout(stage, "/World/Room/Wall_X_Pos", 0.5 * sx + 0.5 * t, sy, sz, t, wall_color, cutouts["x_pos"], Gf, UsdGeom, UsdShade, Sdf, wall_visual)
    create_x_wall_with_cutout(stage, "/World/Room/Wall_X_Neg", -0.5 * sx - 0.5 * t, sy, sz, t, wall_color, cutouts["x_neg"], Gf, UsdGeom, UsdShade, Sdf, wall_visual)
    create_y_wall_with_cutout(stage, "/World/Room/Wall_Y_Pos", 0.5 * sy + 0.5 * t, sx, sz, t, wall_color, cutouts["y_pos"], Gf, UsdGeom, UsdShade, Sdf, wall_visual)
    create_y_wall_with_cutout(stage, "/World/Room/Wall_Y_Neg", -0.5 * sy - 0.5 * t, sx, sz, t, wall_color, cutouts["y_neg"], Gf, UsdGeom, UsdShade, Sdf, wall_visual)

    if bool(room_cfg.get("has_roof", False)):
        create_box(stage, "/World/Room/Roof", (0.0, 0.0, sz + 0.5 * t), (sx, sy, t), wall_color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=wall_visual)



def collect_wall_cutouts(cfg: dict[str, Any], sx: float, sy: float, sz: float) -> dict[str, list[dict[str, float]]]:
    """Collect rectangular negative obstacles that cut generated room walls.

    This is intentionally constrained: the runner does not do arbitrary CSG booleans.
    A negative obstacle is interpreted as an invisible wall cutter when its footprint
    touches or nearly touches one generated room wall. Any obstacle kind is accepted;
    its axis-aligned bounding box is used as the rectangular cutter.
    """
    eps = 1e-6
    wall_snap_tolerance = max(float(cfg["room"].get("wall_thickness_m", 0.05)) * 2.0, 0.10)
    cutouts: dict[str, list[dict[str, float]]] = {"x_pos": [], "x_neg": [], "y_pos": [], "y_neg": []}
    x_room_min = -0.5 * sx
    x_room_max = 0.5 * sx
    y_room_min = -0.5 * sy
    y_room_max = 0.5 * sy

    for obstacle in cfg.get("obstacles", []):
        if not bool(obstacle.get("enabled", True)) or not bool(obstacle.get("negative", False)):
            continue
        px, py, pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
        ox, oy, oz = [float(v) for v in obstacle.get("size_m", [0.0, 0.0, 0.0])]

        x0 = px - 0.5 * ox
        x1 = px + 0.5 * ox
        y0 = py - 0.5 * oy
        y1 = py + 0.5 * oy
        z0 = max(0.0, pz - 0.5 * oz)
        z1 = min(sz, pz + 0.5 * oz)
        if z1 <= z0 + eps:
            print(f"WARNING: negative obstacle '{obstacle.get('name', '<unnamed>')}' has no vertical overlap with the room wall height.")
            continue

        candidates: list[tuple[float, str]] = [
            (abs(x1 - x_room_max), "x_pos"),
            (abs(x0 - x_room_min), "x_neg"),
            (abs(y1 - y_room_max), "y_pos"),
            (abs(y0 - y_room_min), "y_neg"),
        ]
        candidates.sort(key=lambda item: item[0])
        distance_to_wall, wall_key = candidates[0]
        reaches_wall = (
            x1 >= x_room_max - wall_snap_tolerance
            or x0 <= x_room_min + wall_snap_tolerance
            or y1 >= y_room_max - wall_snap_tolerance
            or y0 <= y_room_min + wall_snap_tolerance
        )
        if not reaches_wall:
            print(
                f"WARNING: negative obstacle '{obstacle.get('name', '<unnamed>')}' is not close enough to a room wall for a cutout. "
                f"Nearest wall distance={distance_to_wall:.3f} m, tolerance={wall_snap_tolerance:.3f} m. It will be invisible and non-colliding."
            )
            continue

        if wall_key == "x_pos":
            u0 = max(y_room_min, y0)
            u1 = min(y_room_max, y1)
        elif wall_key == "x_neg":
            u0 = max(y_room_min, y0)
            u1 = min(y_room_max, y1)
        elif wall_key == "y_pos":
            u0 = max(x_room_min, x0)
            u1 = min(x_room_max, x1)
        else:
            u0 = max(x_room_min, x0)
            u1 = min(x_room_max, x1)

        if u1 <= u0 + eps:
            print(f"WARNING: negative obstacle '{obstacle.get('name', '<unnamed>')}' does not overlap the selected wall span.")
            continue
        cutouts[wall_key].append({"u0": u0, "u1": u1, "z0": z0, "z1": z1})
        print(
            f"Negative obstacle '{obstacle.get('name', '<unnamed>')}' assigned to wall cutout {wall_key}: "
            f"u=[{u0:+.3f}, {u1:+.3f}], z=[{z0:+.3f}, {z1:+.3f}]."
        )

    for wall_name, wall_cutouts in cutouts.items():
        if len(wall_cutouts) > 1:
            print(
                f"WARNING: {wall_name} has {len(wall_cutouts)} negative cutouts. "
                "This runner currently applies only the first one to keep wall generation deterministic."
            )
            cutouts[wall_name] = wall_cutouts[:1]
    return cutouts


def negative_obstacle_reaches_room_wall(cfg: dict[str, Any], obstacle: dict[str, Any]) -> bool:
    room_cfg = cfg["room"]
    sx, sy, sz = [float(v) for v in room_cfg["size_m"]]
    tolerance = max(float(room_cfg.get("wall_thickness_m", 0.05)) * 2.0, 0.10)
    px, py, pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
    ox, oy, oz = [float(v) for v in obstacle.get("size_m", [0.0, 0.0, 0.0])]
    x0 = px - 0.5 * ox
    x1 = px + 0.5 * ox
    y0 = py - 0.5 * oy
    y1 = py + 0.5 * oy
    z0 = pz - 0.5 * oz
    z1 = pz + 0.5 * oz
    z_overlaps = z1 > 0.0 and z0 < sz
    if not z_overlaps:
        return False
    return (
        x1 >= 0.5 * sx - tolerance
        or x0 <= -0.5 * sx + tolerance
        or y1 >= 0.5 * sy - tolerance
        or y0 <= -0.5 * sy + tolerance
    )

def create_x_wall_with_cutout(
    stage: Any,
    path: str,
    x_center: float,
    sy: float,
    sz: float,
    thickness: float,
    color: Vector3,
    cutouts: list[dict[str, float]],
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any,
    Sdf: Any,
    visual: dict[str, Any],
) -> None:
    if not cutouts:
        create_box(stage, path, (x_center, 0.0, 0.5 * sz), (thickness, sy, sz), color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        return

    cutout = cutouts[0]
    y_min = -0.5 * sy
    y_max = 0.5 * sy
    y0 = clamp(float(cutout["u0"]), y_min, y_max)
    y1 = clamp(float(cutout["u1"]), y_min, y_max)
    z0 = clamp(float(cutout["z0"]), 0.0, sz)
    z1 = clamp(float(cutout["z1"]), 0.0, sz)
    _create_rectangular_wall_segments(
        stage=stage,
        path=path,
        fixed_axis="x",
        fixed_center=x_center,
        u_min=y_min,
        u_max=y_max,
        z_min=0.0,
        z_max=sz,
        hole_u0=y0,
        hole_u1=y1,
        hole_z0=z0,
        hole_z1=z1,
        thickness=thickness,
        color=color,
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        Sdf=Sdf,
        visual=visual,
    )


def create_y_wall_with_cutout(
    stage: Any,
    path: str,
    y_center: float,
    sx: float,
    sz: float,
    thickness: float,
    color: Vector3,
    cutouts: list[dict[str, float]],
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any,
    Sdf: Any,
    visual: dict[str, Any],
) -> None:
    if not cutouts:
        create_box(stage, path, (0.0, y_center, 0.5 * sz), (sx, thickness, sz), color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        return

    cutout = cutouts[0]
    x_min = -0.5 * sx
    x_max = 0.5 * sx
    x0 = clamp(float(cutout["u0"]), x_min, x_max)
    x1 = clamp(float(cutout["u1"]), x_min, x_max)
    z0 = clamp(float(cutout["z0"]), 0.0, sz)
    z1 = clamp(float(cutout["z1"]), 0.0, sz)
    _create_rectangular_wall_segments(
        stage=stage,
        path=path,
        fixed_axis="y",
        fixed_center=y_center,
        u_min=x_min,
        u_max=x_max,
        z_min=0.0,
        z_max=sz,
        hole_u0=x0,
        hole_u1=x1,
        hole_z0=z0,
        hole_z1=z1,
        thickness=thickness,
        color=color,
        Gf=Gf,
        UsdGeom=UsdGeom,
        UsdShade=UsdShade,
        Sdf=Sdf,
        visual=visual,
    )


def _create_rectangular_wall_segments(
    stage: Any,
    path: str,
    fixed_axis: str,
    fixed_center: float,
    u_min: float,
    u_max: float,
    z_min: float,
    z_max: float,
    hole_u0: float,
    hole_u1: float,
    hole_z0: float,
    hole_z1: float,
    thickness: float,
    color: Vector3,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any | None = None,
    Sdf: Any | None = None,
    visual: dict[str, Any] | None = None,
) -> None:
    eps = 1e-6

    if hole_u1 <= hole_u0 + eps or hole_z1 <= hole_z0 + eps:
        # Degenerate cutout. Make a normal wall.
        if fixed_axis == "x":
            create_box(stage, path, (fixed_center, 0.5 * (u_min + u_max), 0.5 * (z_min + z_max)), (thickness, u_max - u_min, z_max - z_min), color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        else:
            create_box(stage, path, (0.5 * (u_min + u_max), fixed_center, 0.5 * (z_min + z_max)), (u_max - u_min, thickness, z_max - z_min), color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        return

    segments = [
        ("U_Neg", u_min, hole_u0, z_min, z_max),
        ("U_Pos", hole_u1, u_max, z_min, z_max),
        ("Z_Neg", hole_u0, hole_u1, z_min, hole_z0),
        ("Z_Pos", hole_u0, hole_u1, hole_z1, z_max),
    ]

    for suffix, seg_u0, seg_u1, seg_z0, seg_z1 in segments:
        if seg_u1 <= seg_u0 + eps or seg_z1 <= seg_z0 + eps:
            continue
        u_center = 0.5 * (seg_u0 + seg_u1)
        z_center = 0.5 * (seg_z0 + seg_z1)
        u_size = seg_u1 - seg_u0
        z_size = seg_z1 - seg_z0
        if fixed_axis == "x":
            position = (fixed_center, u_center, z_center)
            size = (thickness, u_size, z_size)
        else:
            position = (u_center, fixed_center, z_center)
            size = (u_size, thickness, z_size)
        create_box(stage, f"{path}_{suffix}", position, size, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)

def build_obstacles(stage: Any, cfg: dict[str, Any], room: BoxRoom, Gf: Any, UsdGeom: Any, UsdShade: Any, Sdf: Any) -> None:
    for obstacle in cfg.get("obstacles", []):
        if not bool(obstacle.get("enabled", True)):
            continue

        name = sanitize_prim_name(str(obstacle["name"]))
        kind = str(obstacle.get("kind", "box"))
        path = f"/World/Obstacles/{name}"
        position = tuple(float(v) for v in obstacle["position_m"])
        size = tuple(float(v) for v in obstacle["size_m"])
        color = tuple(float(v) for v in obstacle["color"])

        visual = obstacle.get("visual", {})
        if bool(obstacle.get("negative", False)):
            if negative_obstacle_reaches_room_wall(cfg, obstacle):
                print(f"Negative obstacle '{obstacle['name']}' applied as a generated room-wall cutout. No visible obstacle prim was created.")
            else:
                print(
                    f"Negative obstacle '{obstacle['name']}' did not reach a room wall. "
                    "No visible obstacle prim was created and no collision AABB was registered."
                )
            continue

        if kind in {"box", "wall", "floor_patch"}:
            create_box(stage, path, position, size, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        elif kind == "sphere":
            create_sphere(stage, path, position, size, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
        elif kind in {"cylinder", "pillar"}:
            create_cylinder(stage, path, position, size, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)
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



def build_custom_assets(stage: Any, cfg: dict[str, Any], project_root: Path, stage_utils: Any, Gf: Any, UsdGeom: Any) -> None:
    """Reference user-provided USD assets from the local assets/ tree.

    This intentionally does not do boolean operations and does not auto-generate
    physics collision for arbitrary imported geometry. The asset authoring tool
    should own the detailed geometry. The scenario profile owns placement.
    """
    for index, asset in enumerate(cfg.get("custom_assets", [])):
        if not bool(asset.get("enabled", False)):
            continue
        usd_value = str(asset.get("usd_path", "")).strip()
        if not usd_value:
            print(f"WARNING: custom asset {index} is enabled but usd_path is empty; skipping.")
            continue
        usd_path = resolve_project_path(project_root, usd_value)
        if not usd_path.exists() or not usd_path.is_file():
            raise FileNotFoundError(
                "Custom USD asset is enabled but the file does not exist.\n"
                f"Asset name: {asset.get('name', index)}\n"
                f"Expected file: {usd_path}"
            )
        prim_path = str(asset.get("prim_path", f"/World/CustomAssets/Asset_{index + 1:02d}")).strip()
        if not prim_path.startswith("/"):
            prim_path = f"/World/CustomAssets/{sanitize_prim_name(prim_path)}"
        root_xform = UsdGeom.Xform.Define(stage, prim_path)
        xform = UsdGeom.Xformable(root_xform.GetPrim())
        xform.ClearXformOpOrder()
        pos = [float(v) for v in asset.get("position_m", [0.0, 0.0, 0.0])]
        rot = [float(v) for v in asset.get("rotation_deg", [0.0, 0.0, 0.0])]
        scale = [float(v) for v in asset.get("scale", [1.0, 1.0, 1.0])]
        xform.AddTranslateOp().Set(Gf.Vec3d(pos[0], pos[1], pos[2]))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(rot[0], rot[1], rot[2]))
        xform.AddScaleOp().Set(Gf.Vec3f(scale[0], scale[1], scale[2]))
        reference_path = f"{prim_path}/ReferencedUSD"
        add_reference_to_stage_compatible(
            stage_utils=stage_utils,
            usd_path=str(usd_path),
            prim_path=reference_path,
        )
        if bool(asset.get("collision", False)):
            print(
                f"WARNING: custom asset '{asset.get('name', index)}' requested collision, "
                "but Step 5 does not auto-generate collision meshes for arbitrary imported USD files. "
                "Author collision in the USD asset or add this in a later dedicated physics step."
            )
        print(f"Custom asset loaded: {asset.get('name', index)} -> {usd_path} at {prim_path}")


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


def create_box(
    stage: Any,
    path: str,
    position: Vector3,
    size: Vector3,
    color: Vector3,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any | None = None,
    Sdf: Any | None = None,
    visual: dict[str, Any] | None = None,
) -> None:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    set_visual(stage, prim, path, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)


def create_sphere(
    stage: Any,
    path: str,
    position: Vector3,
    size: Vector3,
    color: Vector3,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any | None = None,
    Sdf: Any | None = None,
    visual: dict[str, Any] | None = None,
) -> None:
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(0.5)
    prim = sphere.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    set_visual(stage, prim, path, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)


def create_cylinder(
    stage: Any,
    path: str,
    position: Vector3,
    size: Vector3,
    color: Vector3,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any | None = None,
    Sdf: Any | None = None,
    visual: dict[str, Any] | None = None,
) -> None:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(0.5)
    cylinder.CreateHeightAttr(1.0)
    prim = cylinder.GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(size[0], size[1], size[2]))
    set_visual(stage, prim, path, color, Gf, UsdGeom, UsdShade=UsdShade, Sdf=Sdf, visual=visual)


def create_cutout_marker(stage: Any, path: str, position: Vector3, size: Vector3, color: Vector3, Gf: Any, UsdGeom: Any) -> None:
    marker_visual = {"opacity": 0.18, "roughness": 0.9, "metallic": 0.0, "reflectance": 0.0}
    create_box(stage, path + "_CutoutMarker", position, size, color, Gf, UsdGeom, visual=marker_visual)


def set_visual(
    stage: Any,
    prim: Any,
    prim_path: str,
    color: Vector3,
    Gf: Any,
    UsdGeom: Any,
    UsdShade: Any | None = None,
    Sdf: Any | None = None,
    visual: dict[str, Any] | None = None,
) -> None:
    set_display_color(prim, color, Gf, UsdGeom, opacity=None if visual is None else visual.get("opacity"))
    if UsdShade is None or Sdf is None:
        return
    bind_preview_surface_material(
        stage=stage,
        prim=prim,
        prim_path=prim_path,
        color=color,
        visual=visual or {},
        Gf=Gf,
        UsdShade=UsdShade,
        Sdf=Sdf,
    )


def set_display_color(prim: Any, color: Vector3, Gf: Any, UsdGeom: Any, opacity: Any = None) -> None:
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        if opacity is not None:
            gprim.CreateDisplayOpacityAttr([float(opacity)])


def bind_preview_surface_material(
    stage: Any,
    prim: Any,
    prim_path: str,
    color: Vector3,
    visual: dict[str, Any],
    Gf: Any,
    UsdShade: Any,
    Sdf: Any,
) -> None:
    material_name = sanitize_prim_name(prim_path.replace("/", "_")) + "_Material"
    material_path = f"/World/Materials/{material_name}"
    shader_path = f"{material_path}/PreviewSurface"

    opacity = clamp(float(visual.get("opacity", 1.0)), 0.0, 1.0)
    roughness = clamp(float(visual.get("roughness", 0.55)), 0.0, 1.0)
    metallic = clamp(float(visual.get("metallic", 0.0)), 0.0, 1.0)
    reflectance = clamp(float(visual.get("reflectance", 0.35)), 0.0, 1.0)

    try:
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(reflectance, reflectance, reflectance))
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)
    except Exception as exc:
        print(f"WARNING: failed to bind preview material for {prim_path}: {exc}")


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
        camera.CreateHorizontalApertureAttr(float(isometric.get("horizontal_aperture_mm", 36.0)))
        camera.CreateVerticalApertureAttr(float(isometric.get("vertical_aperture_mm", 20.25)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 1000.0))

        eye, target = compute_isometric_eye_and_target(cfg)
        set_camera_look_at_transform(
            camera_prim=camera.GetPrim(),
            eye=eye,
            target=target,
            Gf=Gf,
            UsdGeom=UsdGeom,
        )

        if set_camera_view is not None:
            try:
                set_camera_view(
                    eye=list(eye),
                    target=list(target),
                    camera_prim_path=str(isometric["prim_path"]),
                )
            except Exception as exc:
                print(f"WARNING: set_camera_view failed for isometric camera: {exc}")

        print(f"Isometric camera prim created: {isometric['prim_path']}")
        print(f"Isometric camera eye: ({eye[0]:+.3f}, {eye[1]:+.3f}, {eye[2]:+.3f})")
        print(f"Isometric camera target: ({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:+.3f})")


def compute_isometric_eye_and_target(cfg: dict[str, Any]) -> tuple[Vector3, Vector3]:
    """Return a robust isometric camera eye and target.

    When auto_frame_room is true, the camera is placed from azimuth/elevation so the
    full room is visible instead of relying on a hand-entered eye position that may
    point nowhere useful. The distance formula is deliberately conservative.
    """
    isometric = cfg["cameras"]["isometric"]
    room_cfg = cfg["room"]
    room_size = [float(v) for v in room_cfg.get("size_m", [4.0, 3.0, 2.0])]

    if not bool(isometric.get("auto_frame_room", True)):
        eye = tuple(float(v) for v in isometric.get("position_m", [3.0, -3.0, 2.6]))
        target = tuple(float(v) for v in isometric.get("look_at_m", [0.0, 0.0, 0.75]))
        return eye, target

    padding = max(float(isometric.get("padding_m", 0.35)), 0.0)
    target = (
        float(isometric.get("look_at_m", [0.0, 0.0, 0.5 * room_size[2]])[0]),
        float(isometric.get("look_at_m", [0.0, 0.0, 0.5 * room_size[2]])[1]),
        0.5 * room_size[2],
    )
    azimuth_rad = math.radians(float(isometric.get("azimuth_deg", -45.0)))
    elevation_rad = math.radians(float(isometric.get("elevation_deg", 35.0)))
    distance_scale = max(float(isometric.get("distance_scale", 1.85)), 0.01)

    padded = [max(v + 2.0 * padding, 0.01) for v in room_size]
    room_diagonal = math.sqrt(padded[0] * padded[0] + padded[1] * padded[1] + padded[2] * padded[2])
    distance = distance_scale * room_diagonal

    horizontal = math.cos(elevation_rad)
    direction = (
        horizontal * math.cos(azimuth_rad),
        horizontal * math.sin(azimuth_rad),
        math.sin(elevation_rad),
    )
    eye = (
        target[0] + distance * direction[0],
        target[1] + distance * direction[1],
        target[2] + distance * direction[2],
    )
    return eye, target


def set_camera_look_at_transform(camera_prim: Any, eye: Vector3, target: Vector3, Gf: Any, UsdGeom: Any) -> None:
    """Orient a USD camera to look at a target.

    USD cameras look along local -Z with +Y as the camera-up axis. Gf.SetLookAt
    returns a view matrix, so the inverse is authored as the camera-to-world xform.
    This is more reliable than raw Euler rotations for capture cameras.
    """
    eye_vec = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
    target_vec = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
    direction = target_vec - eye_vec
    if direction.GetLength() < 1e-9:
        raise ValueError("Camera eye and target are identical; cannot build a look-at transform.")

    up_vec = Gf.Vec3d(0.0, 0.0, 1.0)
    direction_normalized = direction.GetNormalized()
    if abs(Gf.Dot(direction_normalized, up_vec)) > 0.98:
        up_vec = Gf.Vec3d(0.0, 1.0, 0.0)

    view_matrix = Gf.Matrix4d().SetLookAt(eye_vec, target_vec, up_vec)
    camera_to_world = view_matrix.GetInverse()

    xform = UsdGeom.Xformable(camera_prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(camera_to_world)


def create_camera_capture_session(
    cfg: dict[str, Any],
    output_root: Path,
    scene_name: str,
    carb: Any,
) -> dict[str, Any] | None:
    """Create a Replicator capture session.

    Two modes are supported per camera:
      1. archive_rgb=True: keep rgb_0000.png, rgb_0001.png, ... and also update last_frame.png.
      2. archive_rgb=False with save_last_frame_png=True: write temporary RGB files internally, copy the newest
         frame to camera_*/last_frame.png, then delete the temporary RGB files. This gives a live preview frame
         without filling the output folder during long unattended runs.
    """
    cameras_cfg = cfg.get("cameras", {})
    capture_cfg = cameras_cfg.get("capture", {})
    save_last_frame_png = bool(capture_cfg.get("save_last_frame_png", True))

    requested: list[tuple[str, str, dict[str, Any], bool, bool]] = []
    for camera_key, output_folder in [("onboard", "camera_onboard"), ("isometric", "camera_isometric")]:
        camera_cfg = cameras_cfg.get(camera_key, {})
        enabled = bool(camera_cfg.get("enabled", True))
        archive_rgb = bool(camera_cfg.get("capture_rgb", False))
        # Important: last_frame.png is a live-preview mode, not merely a copy of archived RGB frames.
        # Therefore it must run even when archive_rgb is disabled.
        last_frame_only = bool(save_last_frame_png and not archive_rgb)
        if enabled and (archive_rgb or save_last_frame_png):
            requested.append((camera_key, output_folder, camera_cfg, archive_rgb, last_frame_only))

    if not requested:
        return None

    import omni.replicator.core as rep

    settings = carb.settings.get_settings()
    settings.set("/omni/replicator/backends/disk/root_dir", str(output_root.resolve()))

    camera_params = bool(capture_cfg.get("camera_params", False))
    rt_subframes = int(capture_cfg.get("rt_subframes", 1))
    wait_for_render = bool(capture_cfg.get("wait_for_render", True))

    render_products: list[Any] = []
    writers: list[Any] = []
    camera_metadata: list[dict[str, Any]] = []
    temp_capture_root = output_root / scene_name / ".last_frame_tmp"

    for camera_key, output_folder, camera_cfg, archive_rgb, last_frame_only in requested:
        prim_path = str(camera_cfg["prim_path"])
        resolution = tuple(int(v) for v in camera_cfg.get("resolution", [640, 360]))

        final_folder = output_root / scene_name / output_folder
        final_folder.mkdir(parents=True, exist_ok=True)

        if archive_rgb:
            writer_output_dir = f"{scene_name}/{output_folder}"
            writer_folder = final_folder
        else:
            # Replicator BasicWriter writes numbered files. For last-frame-only mode, isolate those files
            # in a hidden temporary folder so the user-facing camera folder only contains last_frame.png.
            writer_output_dir = f"{scene_name}/.last_frame_tmp/{camera_key}"
            writer_folder = temp_capture_root / camera_key
            writer_folder.mkdir(parents=True, exist_ok=True)

        render_product = rep.create.render_product(
            prim_path,
            resolution,
            name=f"{camera_key}_render_product",
        )
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir=writer_output_dir,
            rgb=True,
            camera_params=(camera_params if archive_rgb else False),
        )
        writer.attach([render_product])
        render_products.append(render_product)
        writers.append(writer)
        camera_metadata.append(
            {
                "camera_key": camera_key,
                "prim_path": prim_path,
                "output_folder": str(final_folder.resolve()),
                "writer_folder": str(writer_folder.resolve()),
                "resolution": list(resolution),
                "archive_rgb": archive_rgb,
                "last_frame_only": last_frame_only,
            }
        )

    rep.orchestrator.set_capture_on_play(False)
    metadata_path = output_root / scene_name / "camera_capture_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "scene_name": scene_name,
                "rt_subframes": rt_subframes,
                "wait_for_render": wait_for_render,
                "camera_params": camera_params,
                "save_last_frame_png": save_last_frame_png,
                "temp_capture_root": str(temp_capture_root.resolve()),
                "cameras": camera_metadata,
                "note": (
                    "Capture uses Replicator BasicWriter. If archive_rgb is false and save_last_frame_png "
                    "is true, numbered RGB files are written only to a hidden temporary folder and deleted "
                    "after last_frame.png is refreshed."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("Crazyflie camera capture enabled")
    for item in camera_metadata:
        mode = "archive rgb_* + last_frame.png" if item["archive_rgb"] else "last_frame.png only"
        print(f"{item['camera_key']}: {item['prim_path']} -> {item['output_folder']} ({mode})")
    print(f"RT subframes: {rt_subframes}")
    print("=" * 100)

    return {
        "rep": rep,
        "render_products": render_products,
        "writers": writers,
        "rt_subframes": rt_subframes,
        "wait_for_render": wait_for_render,
        "save_last_frame_png": save_last_frame_png,
        "camera_metadata": camera_metadata,
        "last_frame_seen": {},
    }


def step_camera_capture(session: dict[str, Any] | None, frame: int) -> None:
    if session is None:
        return
    rep = session["rep"]
    rep.orchestrator.step(
        rt_subframes=int(session.get("rt_subframes", 1)),
        pause_timeline=True,
        delta_time=0.0,
        wait_for_render=bool(session.get("wait_for_render", True)),
    )
    if bool(session.get("save_last_frame_png", True)):
        update_last_frame_pngs(session)



def update_last_frame_pngs(session: dict[str, Any]) -> None:
    metadata = session.get("camera_metadata", [])
    seen = session.setdefault("last_frame_seen", {})
    for item in metadata:
        output_folder = Path(item["output_folder"])
        writer_folder = Path(item.get("writer_folder", item["output_folder"]))
        if not writer_folder.exists():
            continue
        files = sorted(
            [path for path in writer_folder.glob("rgb_*.png") if path.is_file()],
            key=lambda p: (p.stat().st_mtime_ns, p.name),
        )
        if not files:
            continue
        latest = files[-1]
        key = str(writer_folder)
        if seen.get(key) != latest.name:
            output_folder.mkdir(parents=True, exist_ok=True)
            target = output_folder / "last_frame.png"
            try:
                shutil.copy2(latest, target)
                seen[key] = latest.name
            except Exception as exc:
                print(f"WARNING: failed to update {target}: {exc}")

        if bool(item.get("last_frame_only", False)):
            # Keep disk usage bounded during long runs. Do not expose numbered RGB files when the user only
            # asked for last_frame.png.
            for path in files:
                try:
                    path.unlink()
                except Exception as exc:
                    print(f"WARNING: failed to delete temporary frame {path}: {exc}")


def cleanup_camera_capture(session: dict[str, Any] | None) -> None:
    if session is None:
        return
    rep = session.get("rep")
    try:
        if rep is not None:
            rep.orchestrator.wait_until_complete()
    except Exception as exc:
        print(f"WARNING: rep.orchestrator.wait_until_complete failed: {exc}")
    # One final last_frame refresh catches the final frame written by wait_until_complete().
    try:
        if bool(session.get("save_last_frame_png", True)):
            update_last_frame_pngs(session)
    except Exception as exc:
        print(f"WARNING: final last_frame update failed: {exc}")
    for writer in session.get("writers", []):
        try:
            writer.detach()
        except Exception as exc:
            print(f"WARNING: writer.detach failed: {exc}")
    for render_product in session.get("render_products", []):
        try:
            render_product.destroy()
        except Exception as exc:
            print(f"WARNING: render_product.destroy failed: {exc}")


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
