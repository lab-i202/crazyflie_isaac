from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE = SCRIPT_DIR / "profiles" / "crazyflie" / "crazyflie_room_basic.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "outputs" / "cloned_crazyflie_envs.usd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one Crazyflie room from the repository profile, clone it "
            "with GridCloner, run a short headless physics smoke test, and save the stage."
        )
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Crazyflie room profile JSON.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of environments. Defaults to profile.future_rl.num_envs, or 4.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=None,
        help="Grid spacing in metres. Defaults to profile.future_rl.spacing_m, or 5.0.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=120,
        help="Number of simulation updates before exit. Use 0 to only build and save.",
    )
    parser.add_argument(
        "--gravity",
        type=float,
        default=0.0,
        help="Gravity magnitude in m/s^2. Default is 0 so unpowered drones do not fall.",
    )
    parser.add_argument(
        "--save-usd",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output USD file. Relative paths are resolved from this script's directory.",
    )
    parser.add_argument(
        "--show-gui",
        action="store_true",
        help="Open the Isaac Sim GUI. The default is headless.",
    )
    parser.add_argument(
        "--no-physics-replication",
        action="store_true",
        help="Disable GridCloner physics replication for debugging.",
    )
    parser.add_argument(
        "--copy-from-source",
        action="store_true",
        help="Create independent USD copies instead of inherited clones.",
    )
    return parser.parse_args()


def load_profile(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        profile = json.load(file)

    required = ("room", "crazyflie")
    missing = [key for key in required if key not in profile]
    if missing:
        raise ValueError(f"Profile is missing required sections: {missing}")

    return profile


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def vector3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values: {values}")
    return float(values[0]), float(values[1]), float(values[2])


def validate_configuration(
    profile: dict[str, Any],
    num_envs: int,
    spacing: float,
    steps: int,
    gravity: float,
) -> None:
    if num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    if spacing <= 0.0:
        raise ValueError("--spacing must be greater than 0.")
    if steps < 0:
        raise ValueError("--steps cannot be negative.")
    if gravity < 0.0:
        raise ValueError("--gravity cannot be negative.")

    room = profile["room"]
    if bool(room.get("enabled", True)):
        room_x, room_y, _ = vector3(room["size_m"], "room.size_m")
        wall_thickness = float(room.get("wall_thickness_m", 0.05))
        minimum_spacing = max(room_x, room_y) + 2.0 * wall_thickness
        if spacing < minimum_spacing:
            raise ValueError(
                f"Grid spacing {spacing:.3f} m is too small for a "
                f"{room_x:.3f} x {room_y:.3f} m room. "
                f"Use at least {minimum_spacing:.3f} m."
            )


def main() -> int:
    args = parse_args()
    profile_path = args.profile.expanduser().resolve()
    profile = load_profile(profile_path)

    future_rl = profile.get("future_rl", {})
    num_envs = (
        args.num_envs
        if args.num_envs is not None
        else int(future_rl.get("num_envs", 4))
    )
    spacing = (
        args.spacing
        if args.spacing is not None
        else float(future_rl.get("spacing_m", 5.0))
    )

    # The repository profile currently reserves num_envs=1 for future RL.
    # A clone test is more useful with several environments, so use four when
    # the caller did not provide --num-envs and the profile still says one.
    if args.num_envs is None and num_envs == 1:
        num_envs = 4

    validate_configuration(profile, num_envs, spacing, args.steps, args.gravity)
    save_path = resolve_path(args.save_usd)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Isaac Sim must be launched before importing omni, pxr, or isaacsim APIs.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.show_gui})

    try:
        import omni.client
        import omni.timeline
        import omni.usd
        from isaacsim.core.cloner import GridCloner
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

        def set_pose(
            prim: Any,
            position: Sequence[float],
            rotation_deg: Sequence[float] = (0.0, 0.0, 0.0),
            scale: Sequence[float] | None = None,
        ) -> None:
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(*vector3(position, "position")))
            xformable.AddRotateXYZOp().Set(
                Gf.Vec3f(*vector3(rotation_deg, "rotation_deg"))
            )
            if scale is not None:
                xformable.AddScaleOp().Set(Gf.Vec3f(*vector3(scale, "scale")))

        def set_color(geometry: Any, color: Sequence[float]) -> None:
            geometry.CreateDisplayColorAttr(
                [Gf.Vec3f(*vector3(color, "display color"))]
            )

        def create_cube(
            stage: Any,
            prim_path: str,
            position: Sequence[float],
            size: Sequence[float],
            color: Sequence[float],
            collision: bool,
        ) -> Any:
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            set_pose(cube.GetPrim(), position=position, scale=size)
            set_color(cube, color)
            if collision:
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            return cube.GetPrim()

        def create_room(stage: Any, env_path: str) -> None:
            room_cfg = profile["room"]
            if not bool(room_cfg.get("enabled", True)):
                return

            size_x, size_y, size_z = vector3(room_cfg["size_m"], "room.size_m")
            thickness = float(room_cfg.get("wall_thickness_m", 0.05))
            floor_color = room_cfg.get("floor_color", [0.35, 0.35, 0.35])
            wall_color = room_cfg.get("wall_color", [0.75, 0.78, 0.82])
            room_path = f"{env_path}/Room"
            UsdGeom.Xform.Define(stage, room_path)

            create_cube(
                stage,
                f"{room_path}/Floor",
                position=(0.0, 0.0, -thickness / 2.0),
                size=(size_x, size_y, thickness),
                color=floor_color,
                collision=True,
            )
            create_cube(
                stage,
                f"{room_path}/WallPositiveX",
                position=(size_x / 2.0 + thickness / 2.0, 0.0, size_z / 2.0),
                size=(thickness, size_y + 2.0 * thickness, size_z),
                color=wall_color,
                collision=True,
            )
            create_cube(
                stage,
                f"{room_path}/WallNegativeX",
                position=(-size_x / 2.0 - thickness / 2.0, 0.0, size_z / 2.0),
                size=(thickness, size_y + 2.0 * thickness, size_z),
                color=wall_color,
                collision=True,
            )
            create_cube(
                stage,
                f"{room_path}/WallPositiveY",
                position=(0.0, size_y / 2.0 + thickness / 2.0, size_z / 2.0),
                size=(size_x, thickness, size_z),
                color=wall_color,
                collision=True,
            )
            create_cube(
                stage,
                f"{room_path}/WallNegativeY",
                position=(0.0, -size_y / 2.0 - thickness / 2.0, size_z / 2.0),
                size=(size_x, thickness, size_z),
                color=wall_color,
                collision=True,
            )

            if bool(room_cfg.get("has_roof", False)):
                create_cube(
                    stage,
                    f"{room_path}/Roof",
                    position=(0.0, 0.0, size_z + thickness / 2.0),
                    size=(size_x, size_y, thickness),
                    color=wall_color,
                    collision=True,
                )

        def create_obstacles(stage: Any, env_path: str) -> None:
            obstacles_path = f"{env_path}/Obstacles"
            UsdGeom.Xform.Define(stage, obstacles_path)

            for index, obstacle in enumerate(profile.get("obstacles", [])):
                if not bool(obstacle.get("enabled", True)):
                    continue
                if obstacle.get("kind", "box") != "box":
                    print(
                        f"WARNING: skipping unsupported obstacle kind: "
                        f"{obstacle.get('kind')}"
                    )
                    continue

                name = str(obstacle.get("name", f"box_{index:03d}"))
                create_cube(
                    stage,
                    f"{obstacles_path}/{name}",
                    position=obstacle["position_m"],
                    size=obstacle["size_m"],
                    color=obstacle.get("color", [0.7, 0.2, 0.2]),
                    collision=bool(obstacle.get("collision", True)),
                )

        def create_landmark(stage: Any, env_path: str) -> None:
            landmark_cfg = profile.get("landmark", {})
            if not bool(landmark_cfg.get("enabled", True)):
                return

            landmark_path = f"{env_path}/Landmark"
            sphere = UsdGeom.Sphere.Define(stage, f"{landmark_path}/Visual")
            sphere.CreateRadiusAttr(float(landmark_cfg.get("radius_m", 0.08)))
            set_pose(sphere.GetPrim(), landmark_cfg["position_m"])
            set_color(sphere, landmark_cfg.get("color", [1.0, 0.85, 0.1]))

            light = UsdLux.SphereLight.Define(stage, f"{landmark_path}/Light")
            light.CreateRadiusAttr(float(landmark_cfg.get("radius_m", 0.08)))
            light.CreateIntensityAttr(float(landmark_cfg.get("intensity", 2500.0)))
            light.CreateExposureAttr(float(landmark_cfg.get("exposure", 0.0)))
            light.CreateColorAttr(
                Gf.Vec3f(*vector3(landmark_cfg.get("color", [1.0, 0.85, 0.1]), "landmark color"))
            )
            set_pose(light.GetPrim(), landmark_cfg["position_m"])

        def asset_exists(url: str) -> bool:
            try:
                result = omni.client.stat(url)[0]
                return result == omni.client.Result.OK
            except Exception:
                return False

        def resolve_crazyflie_asset() -> str:
            asset_cfg = profile["crazyflie"]["asset"]
            attempted: list[str] = []

            assets_root = get_assets_root_path()
            if assets_root:
                for relative_path in asset_cfg.get("candidates", []):
                    candidate = (
                        f"{str(assets_root).rstrip('/')}/"
                        f"{str(relative_path).lstrip('/')}"
                    )
                    attempted.append(candidate)
                    if asset_exists(candidate):
                        return candidate

            local_value = str(asset_cfg.get("local_usd_path", "")).strip()
            if local_value:
                local_path = Path(local_value).expanduser()
                if not local_path.is_absolute():
                    local_path = SCRIPT_DIR / local_path
                local_path = local_path.resolve()
                attempted.append(str(local_path))
                if local_path.is_file():
                    return local_path.as_posix()

            checked = "\n  - ".join(attempted) if attempted else "<none>"
            raise FileNotFoundError(
                "Crazyflie USD asset was not found. Checked:\n"
                f"  - {checked}\n"
                "Install/mount the Isaac Sim assets or place cf2x.usd at the "
                "profile's local_usd_path."
            )

        def create_crazyflie(stage: Any, env_path: str, asset_path: str) -> None:
            crazyflie_cfg = profile["crazyflie"]
            pose_cfg = crazyflie_cfg["initial_pose"]
            rig_path = f"{env_path}/CrazyflieRig"
            model_path = f"{rig_path}/Crazyflie"

            rig = UsdGeom.Xform.Define(stage, rig_path)
            set_pose(
                rig.GetPrim(),
                position=pose_cfg["position_m"],
                rotation_deg=pose_cfg.get("rotation_deg", [0.0, 0.0, 0.0]),
            )

            model_prim = stage.DefinePrim(model_path, "Xform")
            if not model_prim.GetReferences().AddReference(asset_path):
                raise RuntimeError(
                    f"Failed to add Crazyflie USD reference: {asset_path}"
                )

        def create_template_environment(
            stage: Any,
            env_path: str,
            asset_path: str,
        ) -> None:
            UsdGeom.Xform.Define(stage, env_path)
            create_room(stage, env_path)
            create_obstacles(stage, env_path)
            create_landmark(stage, env_path)
            create_crazyflie(stage, env_path, asset_path)

        usd_context = omni.usd.get_context()
        usd_context.new_stage()
        simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError("Isaac Sim did not create a USD stage.")

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr(float(args.gravity))

        dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(600.0)

        crazyflie_asset = resolve_crazyflie_asset()

        cloner = GridCloner(spacing=spacing)
        cloner.define_base_env("/World/envs")
        env_paths = cloner.generate_paths("/World/envs/env", num_envs)

        source_env_path = env_paths[0]
        create_template_environment(stage, source_env_path, crazyflie_asset)

        # Let the referenced robot asset load before PhysX replication.
        for _ in range(10):
            simulation_app.update()

        env_origins = cloner.clone(
            source_prim_path=source_env_path,
            prim_paths=env_paths,
            replicate_physics=not args.no_physics_replication,
            copy_from_source=args.copy_from_source,
        )

        for _ in range(10):
            simulation_app.update()

        invalid_paths = [
            path
            for path in env_paths
            if not stage.GetPrimAtPath(f"{path}/CrazyflieRig/Crazyflie").IsValid()
        ]
        if invalid_paths:
            raise RuntimeError(
                "The Crazyflie model prim is missing in cloned environments: "
                + ", ".join(invalid_paths)
            )

        if args.steps > 0:
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(args.steps):
                simulation_app.update()
            timeline.stop()

        if not usd_context.save_as_stage(save_path.as_posix()):
            raise RuntimeError(f"Failed to save stage: {save_path}")

        print("=" * 80)
        print("CRAZYFLIE CLONE TEST COMPLETED")
        print("=" * 80)
        print(f"Profile:              {profile_path}")
        print(f"Crazyflie asset:      {crazyflie_asset}")
        print(f"Environment count:    {len(env_paths)}")
        print(f"Grid spacing:         {spacing:.3f} m")
        print(f"Physics replication:  {not args.no_physics_replication}")
        print(f"Independent copies:   {args.copy_from_source}")
        print(f"Gravity:              {args.gravity:.3f} m/s^2")
        print(f"Saved stage:          {save_path}")
        print("Environment roots:")
        for index, (env_path, origin) in enumerate(zip(env_paths, env_origins)):
            print(f"  [{index:03d}] {env_path} origin={origin}")
        print("=" * 80)
        return 0

    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
