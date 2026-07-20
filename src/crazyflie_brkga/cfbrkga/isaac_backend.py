from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .debug_recorder import DebugEpisodeRecorder
from .geometry import Pose, RoomGeometry
from .image_processing import LuminousBlobDetector
from .model import BatchedQNetwork
from .reward import BaselineReward, MissionState
from .types import EvaluationResult, Individual


@dataclass(slots=True)
class IsaacSlot:
    env_index: int
    root_path: str
    rig_path: str
    camera_path: str
    translate_op: Any
    rotate_op: Any
    annotator: Any
    render_product: Any
    world_origin: tuple[float, float, float]
    pose: Pose | None = None


class IsaacBatchEvaluator:
    """Single-process, multi-room Isaac Sim evaluator.

    The implementation deliberately follows the lifecycle and Replicator annotator
    pattern used by the working ``crazyflie_isaac`` skeleton. Each room owns one
    Crazyflie, one luminous landmark and one onboard RGB camera. Training movement
    is kinematic, matching the current skeleton. The controller is isolated behind
    ``RoomGeometry.integrate_action`` so a rigid-body flight controller can replace
    it without changing BRKGA, database or dashboard code.
    """

    backend_name = "isaac_sim"

    def __init__(self, simulation_app: Any, cfg: dict, project_root: Path) -> None:
        self.app = simulation_app
        self.cfg = cfg
        self.project_root = project_root
        self.output_dir = project_root / "outputs"
        self.geometry = RoomGeometry.from_config(cfg)
        self.detector = LuminousBlobDetector.from_config(cfg)
        self.reward_model = BaselineReward(cfg)
        self.slots: list[IsaacSlot] = []
        self._rep = None
        self._stage = None
        self._scene_query = None
        self._carb = None
        self._timeline = None
        self._preview_camera_path: str | None = None
        self._build_stage()

    def close(self) -> None:
        if self._timeline is not None:
            try:
                self._timeline.stop()
            except Exception:
                pass
        if self._rep is not None:
            for slot in self.slots:
                try:
                    slot.annotator.detach([slot.render_product])
                except Exception:
                    pass
                try:
                    slot.render_product.destroy()
                except Exception:
                    pass

    def prepare_preview(self) -> None:
        """Place every preview robot at the configured initial pose."""
        position = self.cfg["environment"]["crazyflie"]["initial_pose"]["position_m"]
        yaw = math.radians(float(self.cfg["environment"]["crazyflie"]["initial_pose"]["yaw_deg"]))
        pose = Pose(float(position[0]), float(position[1]), float(position[2]), yaw)
        for slot in self.slots:
            slot.pose = pose
            self._apply_pose(slot, pose)
        self._render_updates(int(self.cfg["camera"].get("warmup_updates", 8)))

    def capture_preview_frame(self, env_index: int = 0, updates: int = 1) -> np.ndarray | None:
        if env_index < 0 or env_index >= len(self.slots):
            raise IndexError(f"Invalid preview environment index {env_index}.")
        self._render_updates(max(1, int(updates)))
        return self._camera_image(self.slots[env_index])

    def evaluate_batch(
        self,
        population: list[Individual],
        episode_seed: int,
        evaluation_ids: list[int],
        generation_index: int,
        step_callback=None,
    ) -> list[EvaluationResult]:
        active = len(population)
        if active > len(self.slots):
            raise ValueError("Evaluation batch exceeds configured Isaac environment slots.")
        weights = np.stack([item.decoded_weights for item in population])
        network = BatchedQNetwork(weights, self.cfg["policy"].get("device", "cuda"))
        mission_states = [MissionState() for _ in population]
        episode_rewards = np.zeros(active, dtype=np.float64)
        done = np.zeros(active, dtype=bool)
        success = np.zeros(active, dtype=bool)
        reasons = ["running"] * active
        steps = np.zeros(active, dtype=np.int64)
        visible_steps = np.zeros(active, dtype=np.int64)
        collision_counts = np.zeros(active, dtype=np.int64)
        invalid_frames = np.zeros(active, dtype=np.int64)
        sensor_fallbacks = np.zeros(active, dtype=np.int64)
        action_counts = np.zeros((active, 5), dtype=np.int64)
        euclidean_values: list[list[float]] = [[] for _ in population]
        front_values: list[list[float]] = [[] for _ in population]
        recorders: list[DebugEpisodeRecorder] = []
        debug_cfg = self.cfg["data"]["integrity"]
        debug_enabled = self.cfg["data"]["execution_mode"] == "integrity" and bool(debug_cfg["save_raw_data"])
        for index, item in enumerate(population):
            pose = self.geometry.initial_pose(episode_seed + item.index_in_generation * 1009)
            self.slots[index].pose = pose
            self._apply_pose(self.slots[index], pose)
            debug_dir = self.output_dir / "debug" / f"generation_{generation_index:04d}" / f"evaluation_{evaluation_ids[index]:08d}"
            recorders.append(DebugEpisodeRecorder(debug_dir, debug_enabled, bool(debug_cfg["save_frames"]), int(debug_cfg["frame_stride"])))
        for index in range(active, len(self.slots)):
            # Keep inactive agents below the room so their cameras are not used.
            hidden = Pose(0.0, 0.0, -100.0, 0.0)
            self.slots[index].pose = hidden
            self._apply_pose(self.slots[index], hidden)

        self._render_updates(int(self.cfg["camera"].get("warmup_updates", 8)))
        started = time.perf_counter()
        try:
            current_images = [self._camera_image(self.slots[index]) for index in range(active)]
            for step_index in range(int(self.cfg["mission"]["max_steps"])):
                observations = np.zeros((active, 2), dtype=np.float32)
                for index in range(active):
                    if done[index]:
                        continue
                    image = current_images[index]
                    blob = self.detector.detect(image) if image is not None else self.detector._missing()
                    invalid_frames[index] += int(image is None)
                    readings, used_fallback = self._sensor_readings(self.slots[index], self.slots[index].pose)
                    sensor_fallbacks[index] += int(used_fallback)
                    observations[index] = [readings["front"], blob.euclidean_norm]
                actions = network.actions(observations)
                geometric_collisions = [False] * active
                for index in range(active):
                    if done[index]:
                        continue
                    action_counts[index, int(actions[index])] += 1
                    next_pose, collision = self.geometry.integrate_action(self.slots[index].pose, int(actions[index]), self.cfg)
                    self.slots[index].pose = next_pose
                    geometric_collisions[index] = collision
                    self._apply_pose(self.slots[index], next_pose)
                self._render_updates(int(self.cfg["camera"].get("updates_per_control_step", 1)))
                current_images = [self._camera_image(self.slots[index]) for index in range(active)]

                for index, item in enumerate(population):
                    if done[index]:
                        continue
                    image = current_images[index]
                    blob = self.detector.detect(image) if image is not None else self.detector._missing()
                    invalid_frames[index] += int(image is None)
                    pose = self.slots[index].pose
                    readings, used_fallback = self._sensor_readings(self.slots[index], pose)
                    sensor_fallbacks[index] += int(used_fallback)
                    front = float(readings["front"])
                    colliding = geometric_collisions[index] or self.geometry.range_collision(readings, pose, self.cfg)
                    outcome = self.reward_model.step(
                        mission_states[index], step_index, front, blob.euclidean_norm, blob.visible, colliding
                    )
                    episode_rewards[index] += outcome.reward
                    visible_steps[index] += int(blob.visible)
                    collision_counts[index] += int(colliding)
                    euclidean_values[index].append(blob.euclidean_norm)
                    front_values[index].append(front)
                    steps[index] = step_index + 1
                    reasons[index] = outcome.reason
                    success[index] = outcome.success
                    row = {
                        "step_index": step_index,
                        "simulation_time_s": (step_index + 1) * float(self.cfg["mission"]["control_step_s"]),
                        "camera_frame_index": step_index,
                        "action": int(actions[index]),
                        "reward": outcome.reward,
                        "front_tof_m": front,
                        "euclid_norm": blob.euclidean_norm,
                        "visible": blob.visible,
                        "blob_x": blob.centroid_x,
                        "blob_y": blob.centroid_y,
                        "x_m": pose.x,
                        "y_m": pose.y,
                        "z_m": pose.z,
                        "yaw_rad": pose.yaw_rad,
                        "collision": colliding,
                        "reason": outcome.reason,
                    }
                    if step_callback is not None:
                        step_callback(evaluation_ids[index], row)
                    overlay = self.detector.overlay(image, blob) if image is not None else None
                    recorders[index].write_step(row, image, overlay)
                    done[index] = outcome.terminated or outcome.truncated
                if bool(np.all(done)):
                    break
        finally:
            for recorder in recorders:
                recorder.close()
        wall_time = time.perf_counter() - started
        results: list[EvaluationResult] = []
        for index, item in enumerate(population):
            debug_dir = self.output_dir / "debug" / f"generation_{generation_index:04d}" / f"evaluation_{evaluation_ids[index]:08d}"
            fronts = front_values[index]
            errors = euclidean_values[index]
            results.append(
                EvaluationResult(
                    individual_id=item.individual_id,
                    generation=item.generation,
                    env_index=index,
                    episode_seed=episode_seed,
                    reward=float(episode_rewards[index]),
                    status="finished" if reasons[index] != "simulation_error" else "failed",
                    reason=reasons[index],
                    success=bool(success[index]),
                    steps=int(steps[index]),
                    simulated_time_s=float(steps[index]) * float(self.cfg["mission"]["control_step_s"]),
                    wall_time_s=wall_time,
                    collision_count=int(collision_counts[index]),
                    visible_steps=int(visible_steps[index]),
                    invalid_camera_frames=int(invalid_frames[index]),
                    mean_euclid_norm=float(np.mean(errors)) if errors else 1.0,
                    min_front_tof_m=float(np.min(fronts)) if fronts else self.geometry.max_range,
                    final_front_tof_m=float(fronts[-1]) if fronts else self.geometry.max_range,
                    final_euclid_norm=float(errors[-1]) if errors else 1.0,
                    debug_dir=str(debug_dir) if debug_enabled else "",
                    extra={
                        "backend": "isaac_sim",
                        "camera_path": self.slots[index].camera_path,
                        "range_backend_requested": self.cfg["environment"]["sensors"].get("range_backend", "analytic"),
                        "range_fallback_count": int(sensor_fallbacks[index]),
                        "action_counts": action_counts[index].tolist(),
                    },
                )
            )
        return results

    def _build_stage(self) -> None:
        import carb
        import omni.client
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from omni.physx import get_physx_scene_query_interface
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.storage.native import get_assets_root_path

        self._rep = rep
        self._carb = carb
        self._scene_query = get_physx_scene_query_interface()
        self._timeline = omni.timeline.get_timeline_interface()
        context = omni.usd.get_context()
        context.new_stage()
        self.app.update()
        stage = context.get_stage()
        self._stage = stage
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/Envs")
        UsdGeom.Xform.Define(stage, "/World/Materials")
        physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
        control_backend = self.cfg["environment"]["crazyflie"].get("control_backend", "kinematic")
        physics_scene.CreateGravityMagnitudeAttr().Set(0.0 if control_backend == "kinematic" else 9.81)
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(float(self.cfg["environment"].get("ambient_light_intensity", 500.0)))

        asset_path = self._resolve_crazyflie_asset(omni.client, get_assets_root_path)
        count = int(self.cfg["environment"]["num_parallel_envs"])
        spacing = float(self.cfg["environment"]["env_spacing_m"])
        columns = max(1, int(math.ceil(math.sqrt(count))))
        for env_index in range(count):
            col = env_index % columns
            row = env_index // columns
            origin = ((col - (columns - 1) / 2) * spacing, row * spacing, 0.0)
            root = f"/World/Envs/env_{env_index:04d}"
            root_xform = UsdGeom.Xform.Define(stage, root)
            root_xform.AddTranslateOp().Set(Gf.Vec3d(*origin))
            self._build_room(root, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade)
            self._build_landmark(root, Gf, Sdf, UsdGeom, UsdLux, UsdShade)
            self._build_custom_assets(root, Gf, UsdGeom, stage_utils)
            rig_path = f"{root}/CrazyflieRig"
            rig = UsdGeom.Xform.Define(stage, rig_path)
            rig_xform = UsdGeom.Xformable(rig.GetPrim())
            rig_xform.ClearXformOpOrder()
            translate_op = rig_xform.AddTranslateOp()
            rotate_op = rig_xform.AddRotateXYZOp()
            model_path = f"{rig_path}/Model"
            try:
                stage_utils.add_reference_to_stage(usd_path=asset_path, prim_path=model_path)
            except TypeError:
                stage_utils.add_reference_to_stage(usd_path=asset_path, path=model_path)
            camera_path = f"{rig_path}/OnboardCamera"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.CreateFocalLengthAttr(float(self.cfg["camera"]["focal_length_mm"]))
            camera.CreateHorizontalApertureAttr(float(self.cfg["camera"]["horizontal_aperture_mm"]))
            cam_xform = UsdGeom.Xformable(camera.GetPrim())
            cam_xform.ClearXformOpOrder()
            eye = Gf.Vec3d(*[float(v) for v in self.cfg["camera"]["local_position_m"]])
            target = Gf.Vec3d(eye[0] + 1.0, eye[1], eye[2])
            matrix = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse()
            cam_xform.AddTransformOp().Set(matrix)
            local_rotation = [float(v) for v in self.cfg["camera"].get("local_rotation_deg", [0, 0, 0])]
            if any(abs(value) > 1.0e-9 for value in local_rotation):
                # Keep the proven look-at transform, then apply an optional local XYZ trim.
                # Separate xform ops avoid fragile Gf.Rotation composition across Isaac versions.
                cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(*local_rotation))
            render_product = rep.create.render_product(
                camera_path, (int(self.cfg["camera"]["width"]), int(self.cfg["camera"]["height"]))
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            try:
                annotator.attach([render_product])
            except TypeError as exc:
                message = str(exc)
                if "Unable to write from unknown dtype" in message:
                    import numpy as runtime_numpy
                    raise RuntimeError(
                        "Isaac Sim could not attach the RGB camera annotator. "
                        "The installed NumPy version is incompatible with Isaac Sim 5.1's SyntheticData bindings. "
                        f"Detected NumPy {runtime_numpy.__version__}. Run repair_isaac_environment.bat and retry."
                    ) from exc
                raise
            self.slots.append(IsaacSlot(env_index, root, rig_path, camera_path, translate_op, rotate_op, annotator, render_product, tuple(float(v) for v in origin)))

        if bool(self.cfg.get("scenario_preview", {}).get("enabled", False)):
            self._configure_preview_camera(Gf, UsdGeom)
        if self.cfg["environment"]["sensors"].get("range_backend") == "physx_raycast":
            self._timeline.play()
        for _ in range(12):
            self.app.update()
            time.sleep(1 / 120)
        print(f"Isaac stage ready: {count} independent rooms, asset={asset_path}")

    def _build_room(self, root, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade) -> None:
        room = self.cfg["environment"]["room"]
        sx, sy, sz = (float(v) for v in room["size_m"])
        t = float(room["wall_thickness_m"])
        floor_material = room.get("floor_material", {})
        wall_material = room.get("wall_material", {})
        self._shape(f"{root}/Room/Floor", "box", (0, 0, -t / 2), (sx, sy, t), room["floor_color"], floor_material, True, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade)
        walls = [
            ("WallXPos", (sx / 2 + t / 2, 0, sz / 2), (t, sy, sz)),
            ("WallXNeg", (-sx / 2 - t / 2, 0, sz / 2), (t, sy, sz)),
            ("WallYPos", (0, sy / 2 + t / 2, sz / 2), (sx, t, sz)),
            ("WallYNeg", (0, -sy / 2 - t / 2, sz / 2), (sx, t, sz)),
        ]
        if room.get("has_roof", False):
            walls.append(("Roof", (0, 0, sz + t / 2), (sx, sy, t)))
        for name, position, size in walls:
            self._shape(f"{root}/Room/{name}", "box", position, size, room["wall_color"], wall_material, True, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade)
        for index, obstacle in enumerate(self.cfg["environment"].get("obstacles", [])):
            if not obstacle.get("enabled", True):
                continue
            material = {
                "opacity": obstacle.get("opacity", 1.0),
                "roughness": obstacle.get("roughness", 0.65),
                "metallic": obstacle.get("metallic", 0.0),
                "reflectance": obstacle.get("reflectance", 0.25),
            }
            name = re.sub(r"[^A-Za-z0-9_]", "_", str(obstacle.get("name", f"obstacle_{index:03d}")))
            self._shape(
                f"{root}/Obstacles/{index:03d}_{name}",
                str(obstacle.get("kind", "box")),
                obstacle["position_m"],
                obstacle["size_m"],
                obstacle.get("color", [0.8, 0.2, 0.2]),
                material,
                bool(obstacle.get("collision", True)),
                Gf, Sdf, UsdGeom, UsdPhysics, UsdShade,
            )

    def _build_custom_assets(self, root, Gf, UsdGeom, stage_utils) -> None:
        for index, asset in enumerate(self.cfg["environment"].get("custom_assets", [])):
            if not asset.get("enabled", True):
                continue
            raw_path = str(asset.get("usd_path", "")).strip()
            if not raw_path:
                raise ValueError(f"Custom asset #{index} has an empty usd_path.")
            if raw_path.startswith(("omniverse://", "http://", "https://")):
                usd_path = raw_path
            else:
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = self.project_root / candidate
                if not candidate.exists():
                    raise FileNotFoundError(f"Custom asset USD not found: {candidate}")
                usd_path = str(candidate.resolve())
            name = re.sub(r"[^A-Za-z0-9_]", "_", str(asset.get("name", f"asset_{index:03d}"))).strip("_")
            if not name:
                name = f"asset_{index:03d}"
            prim_path = f"{root}/CustomAssets/{index:03d}_{name}"
            holder = UsdGeom.Xform.Define(self._stage, prim_path)
            xform = UsdGeom.Xformable(holder.GetPrim())
            xform.ClearXformOpOrder()
            position = [float(v) for v in asset.get("position_m", [0.0, 0.0, 0.0])]
            rotation = [float(v) for v in asset.get("rotation_deg", [0.0, 0.0, 0.0])]
            scale = [float(v) for v in asset.get("scale", [1.0, 1.0, 1.0])]
            xform.AddTranslateOp().Set(Gf.Vec3d(*position))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))
            reference_path = f"{prim_path}/ReferencedUSD"
            try:
                stage_utils.add_reference_to_stage(usd_path=usd_path, prim_path=reference_path)
            except TypeError:
                stage_utils.add_reference_to_stage(usd_path=usd_path, path=reference_path)

    def _build_landmark(self, root, Gf, Sdf, UsdGeom, UsdLux, UsdShade) -> None:
        landmark = self.cfg["environment"]["landmark"]
        if not landmark.get("enabled", True):
            return
        position = [float(v) for v in landmark["position_m"]]
        color = [float(v) / 255.0 for v in landmark["rgb_255"]]
        sphere = UsdGeom.Sphere.Define(self._stage, f"{root}/Landmark/VisibleSphere")
        sphere.CreateRadiusAttr(float(landmark["radius_m"]))
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        material_path = f"{root}/Landmark/EmissiveMaterial"
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        emissive_scale = max(0.0, float(landmark.get("emissive_intensity", 1.0)))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[component * emissive_scale for component in color]))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)
        light = UsdLux.SphereLight.Define(self._stage, f"{root}/Landmark/Light")
        light.CreateRadiusAttr(float(landmark["radius_m"]))
        light.CreateIntensityAttr(float(landmark["light_intensity"]))
        light.CreateColorAttr(Gf.Vec3f(*color))
        try:
            light.CreateExposureAttr(float(landmark.get("exposure", 0.0)))
        except Exception:
            pass
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*position))

    def _shape(self, path, kind, position, size, color, material_cfg, collision, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade) -> None:
        kind = str(kind).lower()
        if kind in {"sphere"}:
            geometry = UsdGeom.Sphere.Define(self._stage, path)
            geometry.CreateRadiusAttr(0.5)
        elif kind in {"cylinder", "pillar"}:
            geometry = UsdGeom.Cylinder.Define(self._stage, path)
            geometry.CreateRadiusAttr(0.5)
            geometry.CreateHeightAttr(1.0)
            geometry.CreateAxisAttr("Z")
        else:
            geometry = UsdGeom.Cube.Define(self._stage, path)
            geometry.CreateSizeAttr(1.0)
        prim = geometry.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in position]))
        xform.AddScaleOp().Set(Gf.Vec3d(*[float(v) for v in size]))
        material = self._create_material(path, color, material_cfg, Gf, Sdf, UsdShade)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)

    def _create_material(self, path, color, material_cfg, Gf, Sdf, UsdShade):
        material_path = f"{path}_Material"
        material = UsdShade.Material.Define(self._stage, material_path)
        shader = UsdShade.Shader.Define(self._stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        rgb = [max(0.0, min(1.0, float(v))) for v in color]
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(material_cfg.get("opacity", 1.0)))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material_cfg.get("roughness", 0.65)))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(material_cfg.get("metallic", 0.0)))
        reflectance = max(0.0, min(1.0, float(material_cfg.get("reflectance", 0.25))))
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(reflectance, reflectance, reflectance))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _box(self, path, position, size, color, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade) -> None:
        self._shape(path, "box", position, size, color, {}, True, Gf, Sdf, UsdGeom, UsdPhysics, UsdShade)

    def _configure_preview_camera(self, Gf, UsdGeom) -> None:
        preview = self.cfg.get("scenario_preview", {}).get("camera", {})
        room_size = [float(v) for v in self.cfg["environment"]["room"]["size_m"]]
        look_at = [float(v) for v in preview.get("look_at_m", [0.0, 0.0, room_size[2] * 0.4])]
        azimuth = math.radians(float(preview.get("azimuth_deg", 45.0)))
        elevation = math.radians(float(preview.get("elevation_deg", 28.0)))
        extent = max(room_size[0], room_size[1], room_size[2])
        distance = extent * float(preview.get("distance_scale", 1.25)) + float(preview.get("padding_m", 0.5))
        horizontal = distance * math.cos(elevation)
        eye = Gf.Vec3d(
            look_at[0] + horizontal * math.cos(azimuth),
            look_at[1] + horizontal * math.sin(azimuth),
            look_at[2] + distance * math.sin(elevation),
        )
        target = Gf.Vec3d(*look_at)
        camera_path = "/World/ScenarioPreviewCamera"
        camera = UsdGeom.Camera.Define(self._stage, camera_path)
        camera.CreateFocalLengthAttr(float(preview.get("focal_length_mm", 35.0)))
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
        self._preview_camera_path = camera_path
        try:
            from omni.kit.viewport.utility import get_active_viewport
            viewport = get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
        except Exception as exc:
            print(f"WARNING: could not select preview camera: {exc}")

    def _resolve_crazyflie_asset(self, omni_client, get_assets_root_path) -> str:
        explicit = str(self.cfg["environment"]["crazyflie"].get("usd_path", "")).strip()
        if explicit:
            path = Path(explicit)
            if not path.is_absolute():
                path = self.project_root / path
            if path.exists():
                return str(path.resolve())
            raise FileNotFoundError(f"Configured Crazyflie USD not found: {path}")
        assets_root = get_assets_root_path()
        candidates = []
        if assets_root:
            candidates.extend([
                f"{assets_root}/Isaac/Robots/Bitcraze/Crazyflie/cf2x.usd",
                f"{assets_root}/Isaac/Bitcraze/Crazyflie/cf2x.usd",
                f"{assets_root}/Bitcraze/Crazyflie/cf2x.usd",
                f"{assets_root}/Isaac/Robots/Crazyflie/cf2x.usd",
            ])
        candidates.append(str((self.project_root / "assets" / "robots" / "crazyflie" / "cf2x.usd").resolve()))
        for candidate in candidates:
            if candidate.startswith("omniverse://") or candidate.startswith("http"):
                result, _ = omni_client.stat(candidate)
                if result == omni_client.Result.OK:
                    return candidate
            elif Path(candidate).exists():
                return candidate
        raise FileNotFoundError(
            "Crazyflie cf2x.usd was not found. Set environment.crazyflie.usd_path or place it at "
            "assets/robots/crazyflie/cf2x.usd. Tried:\n" + "\n".join(candidates)
        )

    def _sensor_readings(self, slot: IsaacSlot, pose: Pose) -> tuple[dict[str, float], bool]:
        requested = self.cfg["environment"]["sensors"].get("range_backend", "analytic")
        if requested != "physx_raycast" or self._scene_query is None or self._carb is None:
            return self.geometry.ranges(pose), False
        directions = {
            "front": (math.cos(pose.yaw_rad), math.sin(pose.yaw_rad), 0.0),
            "back": (-math.cos(pose.yaw_rad), -math.sin(pose.yaw_rad), 0.0),
            "left": (-math.sin(pose.yaw_rad), math.cos(pose.yaw_rad), 0.0),
            "right": (math.sin(pose.yaw_rad), -math.cos(pose.yaw_rad), 0.0),
            "up": (0.0, 0.0, 1.0),
            "down": (0.0, 0.0, -1.0),
        }
        max_range = float(self.cfg["environment"]["sensors"]["max_range_m"])
        start_offset = float(self.cfg["environment"]["crazyflie"]["collision_radius_m"]) + 0.02
        base = (
            slot.world_origin[0] + pose.x,
            slot.world_origin[1] + pose.y,
            slot.world_origin[2] + pose.z,
        )
        readings: dict[str, float] = {}
        try:
            for name, direction in directions.items():
                origin = self._carb.Float3(
                    base[0] + direction[0] * start_offset,
                    base[1] + direction[1] * start_offset,
                    base[2] + direction[2] * start_offset,
                )
                ray_direction = self._carb.Float3(*direction)
                hit = self._scene_query.raycast_closest(origin, ray_direction, max_range)
                if hit and bool(hit.get("hit", False)):
                    readings[name] = max(0.0, min(max_range, float(hit.get("distance", max_range))))
                else:
                    readings[name] = max_range
            return readings, False
        except Exception:
            # Training must not halt because one Isaac version changed its query API.
            # The fallback is recorded in each evaluation's extra_json.
            return self.geometry.ranges(pose), True

    def _apply_pose(self, slot: IsaacSlot, pose: Pose) -> None:
        from pxr import Gf

        slot.translate_op.Set(Gf.Vec3d(pose.x, pose.y, pose.z))
        slot.rotate_op.Set(Gf.Vec3f(0.0, 0.0, math.degrees(pose.yaw_rad)))

    def _render_updates(self, count: int) -> None:
        for _ in range(max(1, count)):
            # Replicator annotators are updated by the orchestrator. Falling back to
            # SimulationApp.update keeps compatibility with older Isaac builds.
            try:
                self._rep.orchestrator.step(rt_subframes=1)
            except TypeError:
                self._rep.orchestrator.step()
            except Exception:
                self.app.update()

    @staticmethod
    def _camera_image(slot: IsaacSlot) -> np.ndarray | None:
        try:
            data = slot.annotator.get_data()
            if isinstance(data, dict):
                data = data.get("data", data.get("rgb"))
            if data is None:
                return None
            array = np.asarray(data)
            if array.ndim != 3 or array.shape[2] < 3 or array.size == 0:
                return None
            return np.ascontiguousarray(array[:, :, :3].astype(np.uint8, copy=False))
        except Exception:
            return None
