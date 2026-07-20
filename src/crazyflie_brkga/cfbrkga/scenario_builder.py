from __future__ import annotations

import copy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .gui_widgets import (
    ColorEntry,
    ScrollableFrame,
    canvas_to_room,
    distance,
    format_number_list,
    hex_to_rgb01,
    hex_to_rgb255,
    parse_number_list,
    rgb01_to_hex,
    rgb255_to_hex,
    room_to_canvas,
)


OBSTACLE_KINDS = ("box", "wall", "sphere", "cylinder", "pillar", "floor_patch")


class ScenarioBuilderPanel(ttk.Frame):
    """Scenario editor shared by preview, integrity tests and normal training."""

    def __init__(self, parent: tk.Widget, project_root: Path, on_changed: Callable[[], None] | None = None) -> None:
        super().__init__(parent)
        self.project_root = project_root
        self.on_changed = on_changed
        self.obstacles: list[dict[str, Any]] = []
        self.custom_assets: list[dict[str, Any]] = []
        self.selected_obstacle: int | None = None
        self.selected_asset: int | None = None
        self.selected_map_element = ""
        self._build_variables()
        self._build()
        self._install_traces()

    def _changed(self) -> None:
        if self.on_changed:
            self.on_changed()
        self.refresh_maps()

    def _build_variables(self) -> None:
        # Room
        self.room_x = tk.DoubleVar(value=4.0)
        self.room_y = tk.DoubleVar(value=3.0)
        self.room_z = tk.DoubleVar(value=2.0)
        self.wall_thickness = tk.DoubleVar(value=0.05)
        self.has_roof = tk.BooleanVar(value=False)
        self.floor_color = tk.StringVar(value="#404047")
        self.wall_color = tk.StringVar(value="#a6adb8")
        self.floor_opacity = tk.DoubleVar(value=1.0)
        self.floor_roughness = tk.DoubleVar(value=0.75)
        self.floor_metallic = tk.DoubleVar(value=0.0)
        self.floor_reflectance = tk.DoubleVar(value=0.25)
        self.wall_opacity = tk.DoubleVar(value=1.0)
        self.wall_roughness = tk.DoubleVar(value=0.75)
        self.wall_metallic = tk.DoubleVar(value=0.0)
        self.wall_reflectance = tk.DoubleVar(value=0.25)
        self.ambient_light = tk.DoubleVar(value=350.0)
        self.env_spacing = tk.DoubleVar(value=6.0)

        # Crazyflie
        self.cf_x = tk.DoubleVar(value=-1.4)
        self.cf_y = tk.DoubleVar(value=-0.9)
        self.cf_z = tk.DoubleVar(value=0.7)
        self.cf_yaw = tk.DoubleVar(value=0.0)
        self.cf_collision_radius = tk.DoubleVar(value=0.12)
        self.cf_usd_path = tk.StringVar(value="")
        self.cf_spawn_randomization = tk.StringVar(value="0, 0, 0")

        # Landmark
        self.landmark_enabled = tk.BooleanVar(value=True)
        self.landmark_x = tk.DoubleVar(value=1.55)
        self.landmark_y = tk.DoubleVar(value=0.0)
        self.landmark_z = tk.DoubleVar(value=0.8)
        self.landmark_radius = tk.DoubleVar(value=0.1)
        self.landmark_color = tk.StringVar(value="#ffe114")
        self.landmark_emissive = tk.DoubleVar(value=8.0)
        self.landmark_light = tk.DoubleVar(value=2500.0)
        self.landmark_exposure = tk.DoubleVar(value=0.0)

        # Camera
        self.camera_width = tk.IntVar(value=320)
        self.camera_height = tk.IntVar(value=240)
        self.camera_focal = tk.DoubleVar(value=24.0)
        self.camera_aperture = tk.DoubleVar(value=20.955)
        self.camera_position = tk.StringVar(value="0.08, 0, 0.03")
        self.camera_rotation = tk.StringVar(value="0, 0, 0")
        self.preview_auto_frame = tk.BooleanVar(value=True)
        self.preview_azimuth = tk.DoubleVar(value=45.0)
        self.preview_elevation = tk.DoubleVar(value=28.0)
        self.preview_distance_scale = tk.DoubleVar(value=1.25)
        self.preview_padding = tk.DoubleVar(value=0.5)
        self.preview_focal = tk.DoubleVar(value=35.0)
        self.preview_look_at = tk.StringVar(value="0, 0, 0.8")

        # Sensors
        self.range_backend = tk.StringVar(value="physx_raycast")
        self.max_range = tk.DoubleVar(value=4.0)
        self.horizontal_collision = tk.DoubleVar(value=0.15)
        self.top_collision_altitude = tk.DoubleVar(value=1.82)

        # Obstacle editor
        self.obstacle_name = tk.StringVar(value="")
        self.obstacle_enabled = tk.BooleanVar(value=True)
        self.obstacle_kind = tk.StringVar(value="box")
        self.obstacle_position = tk.StringVar(value="0, 0, 0.25")
        self.obstacle_size = tk.StringVar(value="0.4, 0.4, 0.5")
        self.obstacle_color = tk.StringVar(value="#cc3333")
        self.obstacle_collision = tk.BooleanVar(value=True)
        self.obstacle_opacity = tk.DoubleVar(value=1.0)
        self.obstacle_roughness = tk.DoubleVar(value=0.65)
        self.obstacle_metallic = tk.DoubleVar(value=0.0)
        self.obstacle_reflectance = tk.DoubleVar(value=0.25)
        self.map_add_kind = tk.StringVar(value="box")

        # Asset editor
        self.asset_name = tk.StringVar(value="")
        self.asset_enabled = tk.BooleanVar(value=True)
        self.asset_path = tk.StringVar(value="")
        self.asset_position = tk.StringVar(value="0, 0, 0")
        self.asset_rotation = tk.StringVar(value="0, 0, 0")
        self.asset_scale = tk.StringVar(value="1, 1, 1")
        self.asset_collision_size = tk.StringVar(value="")

    def _install_traces(self) -> None:
        for variable in (
            self.room_x,
            self.room_y,
            self.floor_color,
            self.wall_color,
            self.cf_x,
            self.cf_y,
            self.landmark_x,
            self.landmark_y,
            self.landmark_radius,
            self.landmark_color,
        ):
            variable.trace_add("write", lambda *_args: self.refresh_maps())

    def _build(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        self.tabs: dict[str, ScrollableFrame] = {}
        for name in ("Room", "Scene map", "Obstacles", "Custom assets", "Landmark", "Crazyflie", "Cameras & sensors"):
            frame = ScrollableFrame(notebook)
            notebook.add(frame, text=name)
            self.tabs[name] = frame
        self._build_room(self.tabs["Room"].content)
        self._build_map(self.tabs["Scene map"].content)
        self._build_obstacles(self.tabs["Obstacles"].content)
        self._build_assets(self.tabs["Custom assets"].content)
        self._build_landmark(self.tabs["Landmark"].content)
        self._build_crazyflie(self.tabs["Crazyflie"].content)
        self._build_camera_sensors(self.tabs["Cameras & sensors"].content)

    @staticmethod
    def _entry_row(parent: ttk.Frame, row: int, label: str, variable: tk.Variable, width: int = 24) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        parent.columnconfigure(1, weight=1)
        return entry

    def _color_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ColorEntry(parent, variable, changed=self._changed).grid(row=row, column=1, sticky="w", padx=4, pady=4)

    def _build_room(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        row = 0
        ttk.Label(parent, text="Room geometry", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1
        for label, variable in (
            ("Size X [m]", self.room_x),
            ("Size Y [m]", self.room_y),
            ("Size Z [m]", self.room_z),
            ("Wall thickness [m]", self.wall_thickness),
            ("Environment spacing [m]", self.env_spacing),
            ("Ambient light intensity", self.ambient_light),
        ):
            self._entry_row(parent, row, label, variable)
            row += 1
        ttk.Checkbutton(parent, text="Add roof", variable=self.has_roof, command=self._changed).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        row += 1
        self._color_row(parent, row, "Floor color", self.floor_color); row += 1
        self._color_row(parent, row, "Wall color", self.wall_color); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Floor material", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        for label, variable in (
            ("Opacity [0-1]", self.floor_opacity),
            ("Roughness [0-1]", self.floor_roughness),
            ("Metallic [0-1]", self.floor_metallic),
            ("Reflectance [0-1]", self.floor_reflectance),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        ttk.Label(parent, text="Wall material", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0)); row += 1
        for label, variable in (
            ("Opacity [0-1]", self.wall_opacity),
            ("Roughness [0-1]", self.wall_roughness),
            ("Metallic [0-1]", self.wall_metallic),
            ("Reflectance [0-1]", self.wall_reflectance),
        ):
            self._entry_row(parent, row, label, variable); row += 1

    def _build_map(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        left = ttk.LabelFrame(parent, text="Top-down scene", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.LabelFrame(parent, text="Scene elements", padding=6)
        right.grid(row=0, column=1, sticky="nsew")
        self.map_canvas = tk.Canvas(left, width=600, height=430, background="#f5f5f5", highlightthickness=1, highlightbackground="#888")
        self.map_canvas.pack(fill="both", expand=True)
        self.map_canvas.bind("<Configure>", lambda _event: self.refresh_maps())
        self.map_canvas.bind("<Button-1>", self._map_click)
        add_bar = ttk.Frame(left)
        add_bar.pack(fill="x", pady=(6, 0))
        ttk.Label(add_bar, text="New obstacle:").pack(side="left")
        ttk.Combobox(add_bar, textvariable=self.map_add_kind, values=OBSTACLE_KINDS, state="readonly", width=14).pack(side="left", padx=6)
        ttk.Button(add_bar, text="Add at room center", command=self._add_obstacle_from_map_center).pack(side="left")
        ttk.Button(add_bar, text="Refresh views", command=self.refresh_all).pack(side="right")
        ttk.Label(
            left,
            text="Select an element, then click another map position to move it in X/Y. Click empty space with no movable selection to add an obstacle.",
            foreground="gray",
            wraplength=590,
        ).pack(anchor="w", pady=(5, 0))
        columns = ("kind", "position", "editable")
        self.scene_tree = ttk.Treeview(right, columns=columns, show="tree headings", height=18)
        self.scene_tree.heading("#0", text="Element")
        self.scene_tree.heading("kind", text="Kind")
        self.scene_tree.heading("position", text="Position")
        self.scene_tree.heading("editable", text="Move")
        self.scene_tree.column("#0", width=155)
        self.scene_tree.column("kind", width=95)
        self.scene_tree.column("position", width=160)
        self.scene_tree.column("editable", width=55, anchor="center")
        self.scene_tree.pack(fill="both", expand=True)
        self.scene_tree.bind("<<TreeviewSelect>>", self._scene_tree_selected)

    def _build_obstacles(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.LabelFrame(parent, text="Selected obstacle", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        columns = ("on", "kind", "position", "size", "color")
        self.obstacle_tree = ttk.Treeview(left, columns=columns, show="tree headings", height=17)
        self.obstacle_tree.heading("#0", text="Name")
        for key, label in (("on", "On"), ("kind", "Kind"), ("position", "Position"), ("size", "Size"), ("color", "Color")):
            self.obstacle_tree.heading(key, text=label)
        self.obstacle_tree.column("#0", width=130)
        self.obstacle_tree.column("on", width=40, anchor="center")
        self.obstacle_tree.column("kind", width=80)
        self.obstacle_tree.column("position", width=145)
        self.obstacle_tree.column("size", width=145)
        self.obstacle_tree.column("color", width=75)
        self.obstacle_tree.pack(fill="both", expand=True)
        self.obstacle_tree.bind("<<TreeviewSelect>>", self._obstacle_selected)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Add obstacle", command=self.add_obstacle).pack(side="left")
        ttk.Button(buttons, text="Duplicate", command=self.duplicate_obstacle).pack(side="left", padx=5)
        ttk.Button(buttons, text="Remove", command=self.remove_obstacle).pack(side="left")
        row = 0
        self._entry_row(right, row, "Name", self.obstacle_name); row += 1
        ttk.Checkbutton(right, text="Enabled", variable=self.obstacle_enabled, command=self._changed).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        ttk.Label(right, text="Kind").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(right, textvariable=self.obstacle_kind, values=OBSTACLE_KINDS, state="readonly").grid(row=row, column=1, sticky="ew", padx=4, pady=4); row += 1
        self._entry_row(right, row, "Position X, Y, Z", self.obstacle_position); row += 1
        self._entry_row(right, row, "Size X, Y, Z", self.obstacle_size); row += 1
        self._color_row(right, row, "Color", self.obstacle_color); row += 1
        ttk.Checkbutton(right, text="Collision", variable=self.obstacle_collision, command=self._changed).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        for label, variable in (
            ("Opacity [0-1]", self.obstacle_opacity),
            ("Roughness [0-1]", self.obstacle_roughness),
            ("Metallic [0-1]", self.obstacle_metallic),
            ("Reflectance [0-1]", self.obstacle_reflectance),
        ):
            self._entry_row(right, row, label, variable); row += 1
        ttk.Button(right, text="Apply obstacle edits", command=self.apply_obstacle).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(10, 4))

    def _build_assets(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right = ttk.LabelFrame(parent, text="Selected custom asset", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        ttk.Label(
            left,
            text="USD, USDA and USDC files under assets/ appear in the dropdown. Imported geometry can replace code-generated walls or rooms.",
            foreground="gray",
            wraplength=540,
        ).pack(anchor="w", pady=(0, 8))
        columns = ("on", "path", "position")
        self.asset_tree = ttk.Treeview(left, columns=columns, show="tree headings", height=16)
        self.asset_tree.heading("#0", text="Name")
        self.asset_tree.heading("on", text="On")
        self.asset_tree.heading("path", text="USD")
        self.asset_tree.heading("position", text="Position")
        self.asset_tree.column("#0", width=130)
        self.asset_tree.column("on", width=40, anchor="center")
        self.asset_tree.column("path", width=280)
        self.asset_tree.column("position", width=140)
        self.asset_tree.pack(fill="both", expand=True)
        self.asset_tree.bind("<<TreeviewSelect>>", self._asset_selected)
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Add asset", command=self.add_asset).pack(side="left")
        ttk.Button(buttons, text="Remove", command=self.remove_asset).pack(side="left", padx=5)
        ttk.Button(buttons, text="Rescan assets/", command=self._scan_assets).pack(side="left")
        row = 0
        self._entry_row(right, row, "Name", self.asset_name); row += 1
        ttk.Checkbutton(right, text="Enabled", variable=self.asset_enabled, command=self._changed).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=4); row += 1
        ttk.Label(right, text="USD path").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        self.asset_combo = ttk.Combobox(right, textvariable=self.asset_path, values=self._available_assets())
        self.asset_combo.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(right, text="Browse", command=self.browse_asset).grid(row=row, column=2, padx=4, pady=4); row += 1
        self._entry_row(right, row, "Position X, Y, Z", self.asset_position); row += 1
        self._entry_row(right, row, "Rotation roll, pitch, yaw", self.asset_rotation); row += 1
        self._entry_row(right, row, "Scale X, Y, Z", self.asset_scale); row += 1
        self._entry_row(right, row, "Collision AABB X, Y, Z (blank=none)", self.asset_collision_size); row += 1
        ttk.Button(right, text="Apply asset edits", command=self.apply_asset).grid(row=row, column=0, columnspan=3, sticky="w", padx=4, pady=(10, 4))

    def _build_landmark(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        row = 0
        ttk.Checkbutton(parent, text="Enabled", variable=self.landmark_enabled, command=self._changed).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        for label, variable in (
            ("Position X [m]", self.landmark_x),
            ("Position Y [m]", self.landmark_y),
            ("Position Z [m]", self.landmark_z),
            ("Radius [m]", self.landmark_radius),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        self._color_row(parent, row, "Color", self.landmark_color); row += 1
        for label, variable in (
            ("Emissive intensity", self.landmark_emissive),
            ("Light intensity", self.landmark_light),
            ("Exposure", self.landmark_exposure),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        ttk.Label(parent, text="Use Update scenario in the main window to rebuild the live Isaac preview after changing these fields.", foreground="gray", wraplength=700).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(10, 4))

    def _build_crazyflie(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        row = 0
        for label, variable in (
            ("Initial X [m]", self.cf_x),
            ("Initial Y [m]", self.cf_y),
            ("Initial Z [m]", self.cf_z),
            ("Initial yaw [deg]", self.cf_yaw),
            ("Collision radius [m]", self.cf_collision_radius),
            ("Spawn randomization X, Y, Z [m]", self.cf_spawn_randomization),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        ttk.Label(parent, text="Crazyflie USD (blank = Isaac assets)").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(parent, textvariable=self.cf_usd_path).grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(parent, text="Browse", command=self.browse_crazyflie).grid(row=row, column=2, padx=4, pady=4)
        parent.columnconfigure(1, weight=1)

    def _build_camera_sensors(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        row = 0
        ttk.Label(parent, text="Onboard training camera", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        for label, variable in (
            ("Width", self.camera_width),
            ("Height", self.camera_height),
            ("Focal length [mm]", self.camera_focal),
            ("Horizontal aperture [mm]", self.camera_aperture),
            ("Local position X, Y, Z", self.camera_position),
            ("Local rotation roll, pitch, yaw", self.camera_rotation),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Isometric scenario preview camera", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        ttk.Checkbutton(parent, text="Auto-frame room", variable=self.preview_auto_frame, command=self._changed).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        for label, variable in (
            ("Azimuth [deg]", self.preview_azimuth),
            ("Elevation [deg]", self.preview_elevation),
            ("Distance scale", self.preview_distance_scale),
            ("Padding [m]", self.preview_padding),
            ("Focal length [mm]", self.preview_focal),
            ("Look-at X, Y, Z", self.preview_look_at),
        ):
            self._entry_row(parent, row, label, variable); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Range and collision sensors", font=("Segoe UI", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w"); row += 1
        ttk.Label(parent, text="Range backend").grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(parent, textvariable=self.range_backend, values=("physx_raycast", "analytic"), state="readonly").grid(row=row, column=1, sticky="ew", padx=4, pady=4); row += 1
        for label, variable in (
            ("Maximum range [m]", self.max_range),
            ("Horizontal collision threshold [m]", self.horizontal_collision),
            ("Top collision altitude [m]", self.top_collision_altitude),
        ):
            self._entry_row(parent, row, label, variable); row += 1

    # ------------------------ config transfer ------------------------
    def load_from_config(self, cfg: dict[str, Any]) -> None:
        env = cfg["environment"]
        room = env["room"]
        self.room_x.set(room["size_m"][0]); self.room_y.set(room["size_m"][1]); self.room_z.set(room["size_m"][2])
        self.wall_thickness.set(room["wall_thickness_m"]); self.has_roof.set(room.get("has_roof", False))
        self.floor_color.set(rgb01_to_hex(room["floor_color"])); self.wall_color.set(rgb01_to_hex(room["wall_color"]))
        floor = room["floor_material"]; wall = room["wall_material"]
        self.floor_opacity.set(floor["opacity"]); self.floor_roughness.set(floor["roughness"]); self.floor_metallic.set(floor["metallic"]); self.floor_reflectance.set(floor["reflectance"])
        self.wall_opacity.set(wall["opacity"]); self.wall_roughness.set(wall["roughness"]); self.wall_metallic.set(wall["metallic"]); self.wall_reflectance.set(wall["reflectance"])
        self.ambient_light.set(env["ambient_light_intensity"]); self.env_spacing.set(env["env_spacing_m"])
        cf = env["crazyflie"]
        pos = cf["initial_pose"]["position_m"]
        self.cf_x.set(pos[0]); self.cf_y.set(pos[1]); self.cf_z.set(pos[2]); self.cf_yaw.set(cf["initial_pose"]["yaw_deg"])
        self.cf_collision_radius.set(cf["collision_radius_m"]); self.cf_usd_path.set(cf.get("usd_path", "")); self.cf_spawn_randomization.set(format_number_list(cf.get("spawn_randomization_m", [0, 0, 0])))
        lm = env["landmark"]
        self.landmark_enabled.set(lm.get("enabled", True)); self.landmark_x.set(lm["position_m"][0]); self.landmark_y.set(lm["position_m"][1]); self.landmark_z.set(lm["position_m"][2])
        self.landmark_radius.set(lm["radius_m"]); self.landmark_color.set(rgb255_to_hex(lm["rgb_255"])); self.landmark_emissive.set(lm["emissive_intensity"]); self.landmark_light.set(lm["light_intensity"]); self.landmark_exposure.set(lm.get("exposure", 0.0))
        cam = cfg["camera"]
        self.camera_width.set(cam["width"]); self.camera_height.set(cam["height"]); self.camera_focal.set(cam["focal_length_mm"]); self.camera_aperture.set(cam["horizontal_aperture_mm"]); self.camera_position.set(format_number_list(cam["local_position_m"])); self.camera_rotation.set(format_number_list(cam.get("local_rotation_deg", [0, 0, 0])))
        preview = cfg["scenario_preview"]["camera"]
        self.preview_auto_frame.set(preview["auto_frame_room"]); self.preview_azimuth.set(preview["azimuth_deg"]); self.preview_elevation.set(preview["elevation_deg"]); self.preview_distance_scale.set(preview["distance_scale"]); self.preview_padding.set(preview["padding_m"]); self.preview_focal.set(preview["focal_length_mm"]); self.preview_look_at.set(format_number_list(preview["look_at_m"]))
        sensors = env["sensors"]
        self.range_backend.set(sensors["range_backend"]); self.max_range.set(sensors["max_range_m"]); self.horizontal_collision.set(sensors["horizontal_collision_threshold_m"]); self.top_collision_altitude.set(sensors["top_collision_altitude_m"])
        self.obstacles = copy.deepcopy(env.get("obstacles", [])); self.custom_assets = copy.deepcopy(env.get("custom_assets", []))
        self.selected_obstacle = None; self.selected_asset = None; self.selected_map_element = ""
        self.refresh_all()

    def apply_to_config(self, cfg: dict[str, Any]) -> None:
        self.apply_obstacle(silent=True)
        self.apply_asset(silent=True)
        env = cfg["environment"]
        room = env["room"]
        room["size_m"] = [float(self.room_x.get()), float(self.room_y.get()), float(self.room_z.get())]
        room["wall_thickness_m"] = float(self.wall_thickness.get()); room["has_roof"] = bool(self.has_roof.get())
        room["floor_color"] = hex_to_rgb01(self.floor_color.get()); room["wall_color"] = hex_to_rgb01(self.wall_color.get())
        room["floor_material"] = {"opacity": float(self.floor_opacity.get()), "roughness": float(self.floor_roughness.get()), "metallic": float(self.floor_metallic.get()), "reflectance": float(self.floor_reflectance.get())}
        room["wall_material"] = {"opacity": float(self.wall_opacity.get()), "roughness": float(self.wall_roughness.get()), "metallic": float(self.wall_metallic.get()), "reflectance": float(self.wall_reflectance.get())}
        env["ambient_light_intensity"] = float(self.ambient_light.get()); env["env_spacing_m"] = float(self.env_spacing.get())
        cf = env["crazyflie"]
        cf["initial_pose"]["position_m"] = [float(self.cf_x.get()), float(self.cf_y.get()), float(self.cf_z.get())]
        cf["initial_pose"]["yaw_deg"] = float(self.cf_yaw.get()); cf["collision_radius_m"] = float(self.cf_collision_radius.get()); cf["usd_path"] = self.cf_usd_path.get().strip(); cf["spawn_randomization_m"] = parse_number_list(self.cf_spawn_randomization.get(), 3, float)
        lm = env["landmark"]
        lm["enabled"] = bool(self.landmark_enabled.get()); lm["position_m"] = [float(self.landmark_x.get()), float(self.landmark_y.get()), float(self.landmark_z.get())]; lm["radius_m"] = float(self.landmark_radius.get()); lm["rgb_255"] = list(hex_to_rgb255(self.landmark_color.get())); lm["emissive_intensity"] = float(self.landmark_emissive.get()); lm["light_intensity"] = float(self.landmark_light.get()); lm["exposure"] = float(self.landmark_exposure.get())
        cam = cfg["camera"]
        cam["width"] = int(self.camera_width.get()); cam["height"] = int(self.camera_height.get()); cam["focal_length_mm"] = float(self.camera_focal.get()); cam["horizontal_aperture_mm"] = float(self.camera_aperture.get()); cam["local_position_m"] = parse_number_list(self.camera_position.get(), 3, float); cam["local_rotation_deg"] = parse_number_list(self.camera_rotation.get(), 3, float)
        preview = cfg["scenario_preview"]["camera"]
        preview["auto_frame_room"] = bool(self.preview_auto_frame.get()); preview["azimuth_deg"] = float(self.preview_azimuth.get()); preview["elevation_deg"] = float(self.preview_elevation.get()); preview["distance_scale"] = float(self.preview_distance_scale.get()); preview["padding_m"] = float(self.preview_padding.get()); preview["focal_length_mm"] = float(self.preview_focal.get()); preview["look_at_m"] = parse_number_list(self.preview_look_at.get(), 3, float)
        sensors = env["sensors"]
        sensors["range_backend"] = self.range_backend.get(); sensors["max_range_m"] = float(self.max_range.get()); sensors["horizontal_collision_threshold_m"] = float(self.horizontal_collision.get()); sensors["top_collision_altitude_m"] = float(self.top_collision_altitude.get())
        env["obstacles"] = copy.deepcopy(self.obstacles); env["custom_assets"] = copy.deepcopy(self.custom_assets)

    # ------------------------ obstacles ------------------------
    def _new_obstacle(self, kind: str, position: list[float] | None = None) -> dict[str, Any]:
        index = len(self.obstacles)
        sizes = {
            "box": [0.4, 0.4, 0.5],
            "wall": [0.1, 1.2, 1.0],
            "sphere": [0.4, 0.4, 0.4],
            "cylinder": [0.35, 0.35, 0.8],
            "pillar": [0.3, 0.3, 1.4],
            "floor_patch": [0.8, 0.8, 0.03],
        }
        size = sizes.get(kind, sizes["box"])
        pos = position or [0.0, 0.0, size[2] / 2.0]
        return {"name": f"{kind}_{index + 1:02d}", "enabled": True, "kind": kind, "position_m": pos, "size_m": size, "color": [0.8, 0.2, 0.2], "collision": True, "opacity": 1.0, "roughness": 0.65, "metallic": 0.0, "reflectance": 0.25}

    def add_obstacle(self) -> None:
        self.obstacles.append(self._new_obstacle(self.map_add_kind.get()))
        self.selected_obstacle = len(self.obstacles) - 1
        self._load_obstacle_editor(self.selected_obstacle)
        self.refresh_all(); self._changed()

    def _add_obstacle_from_map_center(self) -> None:
        self.obstacles.append(self._new_obstacle(self.map_add_kind.get(), [0.0, 0.0, 0.25]))
        self.selected_obstacle = len(self.obstacles) - 1
        self.refresh_all(); self._changed()

    def duplicate_obstacle(self) -> None:
        if self.selected_obstacle is None:
            return
        item = copy.deepcopy(self.obstacles[self.selected_obstacle])
        item["name"] = f"{item['name']}_copy"
        item["position_m"][0] += 0.2
        item["position_m"][1] += 0.2
        self.obstacles.append(item)
        self.selected_obstacle = len(self.obstacles) - 1
        self.refresh_all(); self._changed()

    def remove_obstacle(self) -> None:
        if self.selected_obstacle is None:
            return
        del self.obstacles[self.selected_obstacle]
        self.selected_obstacle = None
        self.refresh_all(); self._changed()

    def apply_obstacle(self, silent: bool = False) -> None:
        if self.selected_obstacle is None or self.selected_obstacle >= len(self.obstacles):
            return
        try:
            item = self.obstacles[self.selected_obstacle]
            item.update({
                "name": self.obstacle_name.get().strip() or f"obstacle_{self.selected_obstacle:03d}",
                "enabled": bool(self.obstacle_enabled.get()),
                "kind": self.obstacle_kind.get(),
                "position_m": parse_number_list(self.obstacle_position.get(), 3, float),
                "size_m": parse_number_list(self.obstacle_size.get(), 3, float),
                "color": hex_to_rgb01(self.obstacle_color.get()),
                "collision": bool(self.obstacle_collision.get()),
                "opacity": float(self.obstacle_opacity.get()),
                "roughness": float(self.obstacle_roughness.get()),
                "metallic": float(self.obstacle_metallic.get()),
                "reflectance": float(self.obstacle_reflectance.get()),
            })
            self.refresh_all(); self._changed()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Obstacle", str(exc), parent=self)
            else:
                raise

    def _obstacle_selected(self, _event: tk.Event) -> None:
        selection = self.obstacle_tree.selection()
        if not selection:
            return
        self.selected_obstacle = int(selection[0].split(":", 1)[1])
        self.selected_map_element = f"obstacle:{self.selected_obstacle}"
        self._load_obstacle_editor(self.selected_obstacle)
        self.refresh_maps()

    def _load_obstacle_editor(self, index: int) -> None:
        item = self.obstacles[index]
        self.obstacle_name.set(item["name"]); self.obstacle_enabled.set(item.get("enabled", True)); self.obstacle_kind.set(item.get("kind", "box")); self.obstacle_position.set(format_number_list(item["position_m"])); self.obstacle_size.set(format_number_list(item["size_m"])); self.obstacle_color.set(rgb01_to_hex(item.get("color", [0.8, 0.2, 0.2]))); self.obstacle_collision.set(item.get("collision", True)); self.obstacle_opacity.set(item.get("opacity", 1.0)); self.obstacle_roughness.set(item.get("roughness", 0.65)); self.obstacle_metallic.set(item.get("metallic", 0.0)); self.obstacle_reflectance.set(item.get("reflectance", 0.25))

    # ------------------------ custom assets ------------------------
    def _available_assets(self) -> list[str]:
        root = self.project_root / "assets"
        root.mkdir(parents=True, exist_ok=True)
        return [str(path.relative_to(self.project_root)).replace("\\", "/") for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}]

    def _scan_assets(self) -> None:
        self.asset_combo["values"] = self._available_assets()

    def add_asset(self) -> None:
        paths = self._available_assets()
        self.custom_assets.append({"name": f"asset_{len(self.custom_assets) + 1:02d}", "enabled": True, "usd_path": paths[0] if paths else "", "position_m": [0.0, 0.0, 0.0], "rotation_deg": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0], "collision_aabb_size_m": None})
        self.selected_asset = len(self.custom_assets) - 1
        self._load_asset_editor(self.selected_asset)
        self.refresh_all(); self._changed()

    def remove_asset(self) -> None:
        if self.selected_asset is None:
            return
        del self.custom_assets[self.selected_asset]
        self.selected_asset = None
        self.refresh_all(); self._changed()

    def browse_asset(self) -> None:
        path = filedialog.askopenfilename(parent=self, initialdir=str(self.project_root / "assets"), filetypes=[("USD files", "*.usd *.usda *.usdc"), ("All files", "*.*")])
        if not path:
            return
        candidate = Path(path).resolve()
        try:
            relative = candidate.relative_to(self.project_root)
            self.asset_path.set(str(relative).replace("\\", "/"))
        except ValueError:
            self.asset_path.set(str(candidate))

    def apply_asset(self, silent: bool = False) -> None:
        if self.selected_asset is None or self.selected_asset >= len(self.custom_assets):
            return
        try:
            collision_text = self.asset_collision_size.get().strip()
            item = self.custom_assets[self.selected_asset]
            item.update({"name": self.asset_name.get().strip() or f"asset_{self.selected_asset:03d}", "enabled": bool(self.asset_enabled.get()), "usd_path": self.asset_path.get().strip(), "position_m": parse_number_list(self.asset_position.get(), 3, float), "rotation_deg": parse_number_list(self.asset_rotation.get(), 3, float), "scale": parse_number_list(self.asset_scale.get(), 3, float), "collision_aabb_size_m": parse_number_list(collision_text, 3, float) if collision_text else None})
            self.refresh_all(); self._changed()
        except Exception as exc:
            if not silent:
                messagebox.showerror("Custom asset", str(exc), parent=self)
            else:
                raise

    def _asset_selected(self, _event: tk.Event) -> None:
        selection = self.asset_tree.selection()
        if not selection:
            return
        self.selected_asset = int(selection[0].split(":", 1)[1])
        self.selected_map_element = f"asset:{self.selected_asset}"
        self._load_asset_editor(self.selected_asset)
        self.refresh_maps()

    def _load_asset_editor(self, index: int) -> None:
        item = self.custom_assets[index]
        self.asset_name.set(item.get("name", "")); self.asset_enabled.set(item.get("enabled", True)); self.asset_path.set(item.get("usd_path", "")); self.asset_position.set(format_number_list(item.get("position_m", [0, 0, 0]))); self.asset_rotation.set(format_number_list(item.get("rotation_deg", [0, 0, 0]))); self.asset_scale.set(format_number_list(item.get("scale", [1, 1, 1]))); collision = item.get("collision_aabb_size_m"); self.asset_collision_size.set(format_number_list(collision) if collision else "")

    def browse_crazyflie(self) -> None:
        path = filedialog.askopenfilename(parent=self, initialdir=str(self.project_root / "assets"), filetypes=[("USD files", "*.usd *.usda *.usdc"), ("All files", "*.*")])
        if path:
            self.cf_usd_path.set(path)

    # ------------------------ refresh and map ------------------------
    def refresh_all(self) -> None:
        self._refresh_obstacle_tree(); self._refresh_asset_tree(); self._refresh_scene_tree(); self.refresh_maps()

    def _refresh_obstacle_tree(self) -> None:
        for item in self.obstacle_tree.get_children():
            self.obstacle_tree.delete(item)
        for index, obstacle in enumerate(self.obstacles):
            iid = f"obstacle:{index}"
            self.obstacle_tree.insert("", "end", iid=iid, text=obstacle.get("name", iid), values=("yes" if obstacle.get("enabled", True) else "no", obstacle.get("kind", "box"), format_number_list(obstacle.get("position_m", [0, 0, 0])), format_number_list(obstacle.get("size_m", [1, 1, 1])), rgb01_to_hex(obstacle.get("color", [0.8, 0.2, 0.2]))))
        if self.selected_obstacle is not None and self.selected_obstacle < len(self.obstacles):
            iid = f"obstacle:{self.selected_obstacle}"
            self.obstacle_tree.selection_set(iid); self.obstacle_tree.see(iid)

    def _refresh_asset_tree(self) -> None:
        for item in self.asset_tree.get_children():
            self.asset_tree.delete(item)
        for index, asset in enumerate(self.custom_assets):
            iid = f"asset:{index}"
            self.asset_tree.insert("", "end", iid=iid, text=asset.get("name", iid), values=("yes" if asset.get("enabled", True) else "no", asset.get("usd_path", ""), format_number_list(asset.get("position_m", [0, 0, 0]))))
        if self.selected_asset is not None and self.selected_asset < len(self.custom_assets):
            iid = f"asset:{self.selected_asset}"
            self.asset_tree.selection_set(iid); self.asset_tree.see(iid)

    def _refresh_scene_tree(self) -> None:
        for item in self.scene_tree.get_children():
            self.scene_tree.delete(item)
        self.scene_tree.insert("", "end", iid="room", text="Room", values=("room", f"{self.room_x.get():.2f} × {self.room_y.get():.2f}", "no"))
        self.scene_tree.insert("", "end", iid="crazyflie", text="Crazyflie", values=("robot", f"{self.cf_x.get():.2f}, {self.cf_y.get():.2f}, {self.cf_z.get():.2f}", "yes"))
        self.scene_tree.insert("", "end", iid="landmark", text="Landmark", values=("light", f"{self.landmark_x.get():.2f}, {self.landmark_y.get():.2f}, {self.landmark_z.get():.2f}", "yes"))
        for index, obstacle in enumerate(self.obstacles):
            self.scene_tree.insert("", "end", iid=f"obstacle:{index}", text=obstacle.get("name", f"Obstacle {index}"), values=(obstacle.get("kind", "box"), format_number_list(obstacle.get("position_m", [0, 0, 0])), "yes"))
        for index, asset in enumerate(self.custom_assets):
            self.scene_tree.insert("", "end", iid=f"asset:{index}", text=asset.get("name", f"Asset {index}"), values=("USD", format_number_list(asset.get("position_m", [0, 0, 0])), "yes"))
        if self.selected_map_element and self.scene_tree.exists(self.selected_map_element):
            self.scene_tree.selection_set(self.selected_map_element)

    def _scene_tree_selected(self, _event: tk.Event) -> None:
        selection = self.scene_tree.selection()
        if not selection:
            return
        self.selected_map_element = selection[0]
        if self.selected_map_element.startswith("obstacle:"):
            self.selected_obstacle = int(self.selected_map_element.split(":", 1)[1]); self._load_obstacle_editor(self.selected_obstacle)
        elif self.selected_map_element.startswith("asset:"):
            self.selected_asset = int(self.selected_map_element.split(":", 1)[1]); self._load_asset_editor(self.selected_asset)
        self.refresh_maps()

    def refresh_maps(self) -> None:
        if not hasattr(self, "map_canvas") or not self.map_canvas.winfo_exists():
            return
        canvas = self.map_canvas
        canvas.delete("all")
        width = max(200, canvas.winfo_width()); height = max(200, canvas.winfo_height())
        room_x = max(0.1, float(self.room_x.get())); room_y = max(0.1, float(self.room_y.get()))
        x0, y0, scale = room_to_canvas(-room_x / 2, room_y / 2, room_x, room_y, width, height)
        x1, y1, _ = room_to_canvas(room_x / 2, -room_y / 2, room_x, room_y, width, height)
        canvas.create_rectangle(x0, y0, x1, y1, fill=self.floor_color.get(), outline=self.wall_color.get(), width=5, tags=("room",))
        # Grid and axes.
        for fraction in (0.25, 0.5, 0.75):
            gx = x0 + (x1 - x0) * fraction; gy = y0 + (y1 - y0) * fraction
            canvas.create_line(gx, y0, gx, y1, fill="#888888", dash=(2, 4))
            canvas.create_line(x0, gy, x1, gy, fill="#888888", dash=(2, 4))
        self._draw_circle(canvas, "crazyflie", self.cf_x.get(), self.cf_y.get(), 0.10, "#2a76d2", room_x, room_y, width, height, "CF")
        if self.landmark_enabled.get():
            self._draw_circle(canvas, "landmark", self.landmark_x.get(), self.landmark_y.get(), max(0.06, self.landmark_radius.get()), self.landmark_color.get(), room_x, room_y, width, height, "L")
        for index, obstacle in enumerate(self.obstacles):
            if not obstacle.get("enabled", True):
                continue
            px, py, _pz = obstacle.get("position_m", [0, 0, 0]); sx, sy, _sz = obstacle.get("size_m", [0.4, 0.4, 0.4])
            cx, cy, _ = room_to_canvas(px, py, room_x, room_y, width, height)
            color = rgb01_to_hex(obstacle.get("color", [0.8, 0.2, 0.2]))
            if obstacle.get("kind") in {"sphere", "cylinder", "pillar"}:
                radius_px = max(4, min(sx, sy) * scale / 2)
                canvas.create_oval(cx - radius_px, cy - radius_px, cx + radius_px, cy + radius_px, fill=color, outline="#111", tags=(f"obstacle:{index}",))
            else:
                canvas.create_rectangle(cx - sx * scale / 2, cy - sy * scale / 2, cx + sx * scale / 2, cy + sy * scale / 2, fill=color, outline="#111", tags=(f"obstacle:{index}",))
        for index, asset in enumerate(self.custom_assets):
            if not asset.get("enabled", True):
                continue
            px, py, _pz = asset.get("position_m", [0, 0, 0]); sx, sy, _sz = asset.get("collision_aabb_size_m") or [0.35, 0.35, 0.35]
            cx, cy, _ = room_to_canvas(px, py, room_x, room_y, width, height)
            canvas.create_rectangle(cx - sx * scale / 2, cy - sy * scale / 2, cx + sx * scale / 2, cy + sy * scale / 2, fill="#8a62b5", outline="#111", dash=(4, 2), tags=(f"asset:{index}",))
        if self.selected_map_element:
            bbox = canvas.bbox(self.selected_map_element)
            if bbox:
                canvas.create_rectangle(bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4, outline="#00ff4c", width=3)

    @staticmethod
    def _draw_circle(canvas: tk.Canvas, tag: str, x: float, y: float, radius: float, color: str, room_x: float, room_y: float, width: int, height: int, label: str) -> None:
        cx, cy, scale = room_to_canvas(x, y, room_x, room_y, width, height)
        r = max(6, radius * scale)
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="#111", width=2, tags=(tag,))
        canvas.create_text(cx, cy, text=label, tags=(tag,))

    def _map_click(self, event: tk.Event) -> None:
        canvas = self.map_canvas
        current = canvas.find_withtag("current")
        if current:
            tags = canvas.gettags(current[0])
            for tag in tags:
                if tag in {"crazyflie", "landmark", "room"} or tag.startswith("obstacle:") or tag.startswith("asset:"):
                    self.selected_map_element = tag
                    self._refresh_scene_tree(); self.refresh_maps(); return
        room_x = float(self.room_x.get()); room_y = float(self.room_y.get())
        x, y = canvas_to_room(event.x, event.y, room_x, room_y, max(200, canvas.winfo_width()), max(200, canvas.winfo_height()))
        selected = self.selected_map_element
        if selected == "crazyflie":
            self.cf_x.set(x); self.cf_y.set(y)
        elif selected == "landmark":
            self.landmark_x.set(x); self.landmark_y.set(y)
        elif selected.startswith("obstacle:"):
            index = int(selected.split(":", 1)[1]); self.obstacles[index]["position_m"][0] = x; self.obstacles[index]["position_m"][1] = y; self.selected_obstacle = index; self._load_obstacle_editor(index)
        elif selected.startswith("asset:"):
            index = int(selected.split(":", 1)[1]); self.custom_assets[index]["position_m"][0] = x; self.custom_assets[index]["position_m"][1] = y; self.selected_asset = index; self._load_asset_editor(index)
        else:
            obstacle = self._new_obstacle(self.map_add_kind.get(), [x, y, 0.25]); self.obstacles.append(obstacle); self.selected_obstacle = len(self.obstacles) - 1; self.selected_map_element = f"obstacle:{self.selected_obstacle}"; self._load_obstacle_editor(self.selected_obstacle)
        self.refresh_all(); self._changed()
