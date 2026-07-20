from __future__ import annotations

import math
import random
import time
from pathlib import Path


# =============================================================================
# SETTINGS
# =============================================================================

NUM_ENVS = 4
ENV_SPACING_M = 5.0

ROOM_SIZE_M = (4.0, 4.0, 2.5)
WALL_THICKNESS_M = 0.05

START_POSITION_M = (0.0, 0.0, 0.7)

# False opens the Isaac Sim interface.
HEADLESS = False

# Used only when HEADLESS is True.
HEADLESS_STEPS = 3600

# Fixed control and simulation step.
DT_S = 1.0 / 60.0

MIN_SPEED_M_S = 0.25
MAX_SPEED_M_S = 0.65

WALL_MARGIN_M = 0.35
MIN_ALTITUDE_M = 0.35
MAX_ALTITUDE_M = 2.00

TARGET_REACHED_M = 0.06
MIN_TARGET_DISTANCE_M = 0.75

RANDOM_SEED = 7

CRAZYFLIE_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd"
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Saving is optional and disabled by default.
SAVE_STAGE = False
OUTPUT_USD = (
    SCRIPT_DIR
    / "outputs"
    / "crazyflie_random_motion.usd"
)

# =============================================================================


if NUM_ENVS < 1:
    raise ValueError("NUM_ENVS must be at least 1.")

if ENV_SPACING_M <= max(ROOM_SIZE_M[0], ROOM_SIZE_M[1]):
    raise ValueError(
        "ENV_SPACING_M must be larger than the room width."
    )

if MIN_SPEED_M_S <= 0.0:
    raise ValueError("MIN_SPEED_M_S must be greater than zero.")

if MAX_SPEED_M_S < MIN_SPEED_M_S:
    raise ValueError(
        "MAX_SPEED_M_S must be greater than or equal to MIN_SPEED_M_S."
    )

if MIN_ALTITUDE_M < 0.0:
    raise ValueError("MIN_ALTITUDE_M cannot be negative.")

if MAX_ALTITUDE_M <= MIN_ALTITUDE_M:
    raise ValueError(
        "MAX_ALTITUDE_M must be greater than MIN_ALTITUDE_M."
    )

if MAX_ALTITUDE_M >= ROOM_SIZE_M[2]:
    raise ValueError(
        "MAX_ALTITUDE_M must be below the room height."
    )

OUTPUT_USD.parent.mkdir(
    parents=True,
    exist_ok=True,
)

rng = random.Random(RANDOM_SEED)


# Isaac Sim must be launched before importing omni or pxr.
from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": HEADLESS,
        "renderer": "RayTracedLighting",
        "width": 1280,
        "height": 720,
        "fast_shutdown": False,
    }
)


try:
    import omni.client
    import omni.usd

    from isaacsim.core.api import SimulationContext
    from isaacsim.core.cloner import GridCloner
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics


    def set_transform(
        prim,
        position=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=None,
    ):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()

        translate_op = xform.AddTranslateOp()
        rotate_op = xform.AddRotateXYZOp()

        translate_op.Set(
            Gf.Vec3d(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )

        rotate_op.Set(
            Gf.Vec3f(
                float(rotation_deg[0]),
                float(rotation_deg[1]),
                float(rotation_deg[2]),
            )
        )

        if scale is not None:
            scale_op = xform.AddScaleOp()

            scale_op.Set(
                Gf.Vec3f(
                    float(scale[0]),
                    float(scale[1]),
                    float(scale[2]),
                )
            )

        return translate_op, rotate_op


    def create_box(
        stage,
        prim_path,
        position,
        size,
        color,
    ):
        cube = UsdGeom.Cube.Define(
            stage,
            prim_path,
        )

        cube.CreateSizeAttr(1.0)

        set_transform(
            cube.GetPrim(),
            position=position,
            scale=size,
        )

        cube.CreateDisplayColorAttr(
            [
                Gf.Vec3f(
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                )
            ]
        )

        UsdPhysics.CollisionAPI.Apply(
            cube.GetPrim()
        )


    def create_room(
        stage,
        env_path,
    ):
        room_x, room_y, room_z = ROOM_SIZE_M
        thickness = WALL_THICKNESS_M

        room_path = f"{env_path}/Room"

        UsdGeom.Xform.Define(
            stage,
            room_path,
        )

        floor_color = (
            0.30,
            0.30,
            0.30,
        )

        wall_color = (
            0.70,
            0.75,
            0.80,
        )

        create_box(
            stage=stage,
            prim_path=f"{room_path}/Floor",
            position=(
                0.0,
                0.0,
                -thickness / 2.0,
            ),
            size=(
                room_x,
                room_y,
                thickness,
            ),
            color=floor_color,
        )

        create_box(
            stage=stage,
            prim_path=f"{room_path}/WallPositiveX",
            position=(
                room_x / 2.0 + thickness / 2.0,
                0.0,
                room_z / 2.0,
            ),
            size=(
                thickness,
                room_y + 2.0 * thickness,
                room_z,
            ),
            color=wall_color,
        )

        create_box(
            stage=stage,
            prim_path=f"{room_path}/WallNegativeX",
            position=(
                -room_x / 2.0 - thickness / 2.0,
                0.0,
                room_z / 2.0,
            ),
            size=(
                thickness,
                room_y + 2.0 * thickness,
                room_z,
            ),
            color=wall_color,
        )

        create_box(
            stage=stage,
            prim_path=f"{room_path}/WallPositiveY",
            position=(
                0.0,
                room_y / 2.0 + thickness / 2.0,
                room_z / 2.0,
            ),
            size=(
                room_x,
                thickness,
                room_z,
            ),
            color=wall_color,
        )

        create_box(
            stage=stage,
            prim_path=f"{room_path}/WallNegativeY",
            position=(
                0.0,
                -room_y / 2.0 - thickness / 2.0,
                room_z / 2.0,
            ),
            size=(
                room_x,
                thickness,
                room_z,
            ),
            color=wall_color,
        )


    def create_crazyflie(
        stage,
        env_path,
    ):
        rig_path = f"{env_path}/CrazyflieRig"
        model_path = f"{rig_path}/Crazyflie"

        rig = UsdGeom.Xform.Define(
            stage,
            rig_path,
        )

        set_transform(
            rig.GetPrim(),
            position=START_POSITION_M,
        )

        model_prim = stage.DefinePrim(
            model_path,
            "Xform",
        )

        reference_added = (
            model_prim
            .GetReferences()
            .AddReference(CRAZYFLIE_USD)
        )

        if not reference_added:
            raise RuntimeError(
                "Failed to reference the Crazyflie USD."
            )


    def create_environment(
        stage,
        env_path,
    ):
        UsdGeom.Xform.Define(
            stage,
            env_path,
        )

        create_room(
            stage,
            env_path,
        )

        create_crazyflie(
            stage,
            env_path,
        )


    def get_rig_ops(
        stage,
        env_path,
    ):
        rig_path = f"{env_path}/CrazyflieRig"

        rig_prim = stage.GetPrimAtPath(
            rig_path
        )

        if not rig_prim.IsValid():
            raise RuntimeError(
                f"Missing Crazyflie rig: {rig_path}"
            )

        xform = UsdGeom.Xformable(
            rig_prim
        )

        translate_op = None
        rotate_op = None

        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op

            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op

        if translate_op is None:
            translate_op = xform.AddTranslateOp()

        if rotate_op is None:
            rotate_op = xform.AddRotateXYZOp()

        return translate_op, rotate_op


    def distance(
        first,
        second,
    ):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        dz = second[2] - first[2]

        return math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )


    def random_target(
        current_position,
    ):
        max_x = (
            ROOM_SIZE_M[0] / 2.0
            - WALL_MARGIN_M
        )

        max_y = (
            ROOM_SIZE_M[1] / 2.0
            - WALL_MARGIN_M
        )

        for _ in range(100):
            target = (
                rng.uniform(
                    -max_x,
                    max_x,
                ),
                rng.uniform(
                    -max_y,
                    max_y,
                ),
                rng.uniform(
                    MIN_ALTITUDE_M,
                    MAX_ALTITUDE_M,
                ),
            )

            if (
                distance(
                    current_position,
                    target,
                )
                >= MIN_TARGET_DISTANCE_M
            ):
                return target

        return (
            -current_position[0],
            -current_position[1],
            min(
                MAX_ALTITUDE_M,
                max(
                    MIN_ALTITUDE_M,
                    current_position[2] + 0.5,
                ),
            ),
        )


    def create_motion_state(
        stage,
        env_path,
    ):
        translate_op, rotate_op = get_rig_ops(
            stage,
            env_path,
        )

        position = [
            float(START_POSITION_M[0]),
            float(START_POSITION_M[1]),
            float(START_POSITION_M[2]),
        ]

        return {
            "env_path": env_path,
            "position": position,
            "target": random_target(position),
            "speed": rng.uniform(
                MIN_SPEED_M_S,
                MAX_SPEED_M_S,
            ),
            "translate_op": translate_op,
            "rotate_op": rotate_op,
        }


    def select_new_target(
        state,
    ):
        state["target"] = random_target(
            state["position"]
        )

        state["speed"] = rng.uniform(
            MIN_SPEED_M_S,
            MAX_SPEED_M_S,
        )

        print(
            f'{state["env_path"]}: '
            f'new target={state["target"]}, '
            f'speed={state["speed"]:.3f} m/s'
        )


    def update_drone(
        state,
        dt_s,
    ):
        position = state["position"]
        target = state["target"]

        dx = target[0] - position[0]
        dy = target[1] - position[1]
        dz = target[2] - position[2]

        remaining = math.sqrt(
            dx * dx
            + dy * dy
            + dz * dz
        )

        if remaining <= TARGET_REACHED_M:
            select_new_target(state)

            target = state["target"]

            dx = target[0] - position[0]
            dy = target[1] - position[1]
            dz = target[2] - position[2]

            remaining = math.sqrt(
                dx * dx
                + dy * dy
                + dz * dz
            )

        if remaining <= 1.0e-9:
            return

        step_distance = min(
            state["speed"] * dt_s,
            remaining,
        )

        position[0] += (
            dx / remaining
            * step_distance
        )

        position[1] += (
            dy / remaining
            * step_distance
        )

        position[2] += (
            dz / remaining
            * step_distance
        )

        state["translate_op"].Set(
            Gf.Vec3d(
                position[0],
                position[1],
                position[2],
            )
        )

        yaw_deg = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        state["rotate_op"].Set(
            Gf.Vec3f(
                0.0,
                0.0,
                yaw_deg,
            )
        )


    asset_status = omni.client.stat(
        CRAZYFLIE_USD
    )[0]

    if asset_status != omni.client.Result.OK:
        raise FileNotFoundError(
            "Cannot access Crazyflie USD:\n"
            f"{CRAZYFLIE_USD}"
        )


    usd_context = omni.usd.get_context()
    usd_context.new_stage()

    simulation_app.update()

    stage = usd_context.get_stage()

    if stage is None:
        raise RuntimeError(
            "Isaac Sim did not create a USD stage."
        )


    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.z,
    )

    UsdGeom.SetStageMetersPerUnit(
        stage,
        1.0,
    )

    UsdGeom.Xform.Define(
        stage,
        "/World",
    )


    physics_scene = UsdPhysics.Scene.Define(
        stage,
        "/World/PhysicsScene",
    )

    physics_scene.CreateGravityDirectionAttr(
        Gf.Vec3f(
            0.0,
            0.0,
            -1.0,
        )
    )

    physics_scene.CreateGravityMagnitudeAttr(
        0.0
    )


    dome_light = UsdLux.DomeLight.Define(
        stage,
        "/World/DomeLight",
    )

    dome_light.CreateIntensityAttr(
        800.0
    )


    cloner = GridCloner(
        spacing=ENV_SPACING_M
    )

    cloner.define_base_env(
        "/World/envs"
    )

    env_paths = cloner.generate_paths(
        "/World/envs/env",
        NUM_ENVS,
    )

    create_environment(
        stage,
        env_paths[0],
    )


    # Allow the Crazyflie reference to load.
    for _ in range(30):
        simulation_app.update()


    env_origins = cloner.clone(
        source_prim_path=env_paths[0],
        prim_paths=env_paths,

        # Independent USD copies are needed because each rig moves separately.
        copy_from_source=True,

        # Motion is kinematic, not PhysX-driven.
        replicate_physics=False,
    )


    for _ in range(30):
        simulation_app.update()


    motion_states = []

    for env_path in env_paths:
        crazyflie_path = (
            f"{env_path}/CrazyflieRig/Crazyflie"
        )

        if not stage.GetPrimAtPath(
            crazyflie_path
        ).IsValid():
            raise RuntimeError(
                f"Missing Crazyflie clone: {crazyflie_path}"
            )

        state = create_motion_state(
            stage,
            env_path,
        )

        motion_states.append(
            state
        )

        print(
            f'{env_path}: '
            f'target={state["target"]}, '
            f'speed={state["speed"]:.3f} m/s'
        )


    if SAVE_STAGE:
        if not usd_context.save_as_stage(
            OUTPUT_USD.as_posix()
        ):
            raise RuntimeError(
                f"Failed to save stage: {OUTPUT_USD}"
            )


    simulation_context = SimulationContext(
        physics_dt=DT_S,
        rendering_dt=DT_S,
        stage_units_in_meters=1.0,
    )

    simulation_context.initialize_physics()
    simulation_context.play()


    print()
    print("=" * 80)
    print("RANDOM CRAZYFLIE MOVEMENT RUNNING")
    print("=" * 80)
    print(f"Environments: {NUM_ENVS}")
    print(f"Room size:    {ROOM_SIZE_M}")
    print(
        f"Speed:        "
        f"{MIN_SPEED_M_S:.2f} to "
        f"{MAX_SPEED_M_S:.2f} m/s"
    )
    print(
        f"Altitude:     "
        f"{MIN_ALTITUDE_M:.2f} to "
        f"{MAX_ALTITUDE_M:.2f} m"
    )
    print(f"Headless:     {HEADLESS}")
    print()
    print("This version does not use simulation_app.is_running().")
    print("Press Ctrl+C in PowerShell to stop.")
    print("=" * 80)
    print()


    frame_index = 0
    start_time = time.perf_counter()
    last_heartbeat_time = start_time


    try:
        if HEADLESS:
            while frame_index < HEADLESS_STEPS:
                for state in motion_states:
                    update_drone(
                        state,
                        DT_S,
                    )

                simulation_context.step(
                    render=True
                )

                frame_index += 1

        else:
            # Deliberately do not use:
            #
            #     while simulation_app.is_running():
            #
            # Your log shows that this flag becomes False immediately,
            # causing a clean exit even though no Python exception occurs.
            while True:
                for state in motion_states:
                    update_drone(
                        state,
                        DT_S,
                    )

                simulation_context.step(
                    render=True
                )

                frame_index += 1

                current_time = time.perf_counter()

                if (
                    current_time
                    - last_heartbeat_time
                    >= 5.0
                ):
                    elapsed = (
                        current_time
                        - start_time
                    )

                    print(
                        f"Running: "
                        f"{frame_index} frames, "
                        f"{elapsed:.1f} s"
                    )

                    last_heartbeat_time = current_time

    except KeyboardInterrupt:
        print()
        print("Ctrl+C received. Stopping cleanly.")


    simulation_context.stop()


finally:
    simulation_app.close()
