# gui/scenario_launcher_gui.py
#
# Scenario launcher GUI.
#
# This GUI does not import Isaac Sim. It is safe to run with normal Python.
# The launcher returns a scenario_id + JSON profile. The caller starts Isaac Sim afterwards.

from __future__ import annotations

import copy
import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk
from typing import Any

from scenario.scenario_registry import ScenarioSpec, list_scenarios
from utils.crazyflie_profile_io import (
    CONTROL_MODES,
    OBSTACLE_KINDS,
    default_crazyflie_room_profile,
    load_crazyflie_room_profile,
    save_crazyflie_room_profile,
    validate_crazyflie_room_profile,
)


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self.content_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _on_content_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.content_window, width=event.width)

    def _bind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class ScenarioLauncherGui:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self.result: dict[str, Any] | None = None
        self.scenarios = list_scenarios()
        self.spec_by_label = {spec.label: spec for spec in self.scenarios}
        self.spec_by_id = {spec.scenario_id: spec for spec in self.scenarios}

        self.current_profile: dict[str, Any] | None = None
        self.current_spec: ScenarioSpec = self.spec_by_id["crazyflie_room_basic"]
        self.selected_obstacle_index: int | None = None

        self.root = tk.Tk()
        self.root.title("AWES Isaac Scenario Launcher")
        self.root.geometry("980x760")
        self.root.minsize(900, 640)

        self._build_common_variables()
        self._build_crazyflie_variables()
        self._install_map_redraw_traces()
        self._build_layout()
        self._select_scenario(self.current_spec.scenario_id)

    def run(self) -> dict[str, Any] | None:
        self.root.mainloop()
        return self.result

    def _build_common_variables(self) -> None:
        self.scenario_label = tk.StringVar()
        self.profile_path = tk.StringVar()
        self.status_text = tk.StringVar()

    def _build_crazyflie_variables(self) -> None:
        self.scene_name = tk.StringVar()

        self.room_size_x = tk.DoubleVar()
        self.room_size_y = tk.DoubleVar()
        self.room_size_z = tk.DoubleVar()
        self.room_wall_thickness = tk.DoubleVar()
        self.room_has_roof = tk.BooleanVar()
        self.room_floor_color = tk.StringVar()
        self.room_wall_color = tk.StringVar()
        self.room_floor_opacity = tk.DoubleVar()
        self.room_floor_roughness = tk.DoubleVar()
        self.room_floor_metallic = tk.DoubleVar()
        self.room_floor_reflectance = tk.DoubleVar()
        self.room_wall_opacity = tk.DoubleVar()
        self.room_wall_roughness = tk.DoubleVar()
        self.room_wall_metallic = tk.DoubleVar()
        self.room_wall_reflectance = tk.DoubleVar()

        self.selected_scene_element = tk.StringVar(value="")
        self.map_add_kind = tk.StringVar(value="box")
        self._scene_map_canvas: tk.Canvas | None = None
        self._scene_element_tree: ttk.Treeview | None = None

        self.obs_name = tk.StringVar()
        self.obs_enabled = tk.BooleanVar()
        self.obs_kind = tk.StringVar()
        self.obs_x = tk.DoubleVar()
        self.obs_y = tk.DoubleVar()
        self.obs_z = tk.DoubleVar()
        self.obs_sx = tk.DoubleVar()
        self.obs_sy = tk.DoubleVar()
        self.obs_sz = tk.DoubleVar()
        self.obs_color = tk.StringVar()
        self.obs_collision = tk.BooleanVar()
        self.obs_negative = tk.BooleanVar()
        self.obs_opacity = tk.DoubleVar()
        self.obs_roughness = tk.DoubleVar()
        self.obs_metallic = tk.DoubleVar()
        self.obs_reflectance = tk.DoubleVar()
        self._color_entries: dict[int, tk.Entry] = {}
        self._obstacle_map_canvas: tk.Canvas | None = None

        self.landmark_enabled = tk.BooleanVar()
        self.landmark_x = tk.DoubleVar()
        self.landmark_y = tk.DoubleVar()
        self.landmark_z = tk.DoubleVar()
        self.landmark_radius = tk.DoubleVar()
        self.landmark_color = tk.StringVar()
        self.landmark_intensity = tk.DoubleVar()
        self.landmark_exposure = tk.DoubleVar()

        self.asset_mode = tk.StringVar()
        self.local_usd_path = tk.StringVar()
        self.allow_placeholder_if_missing = tk.BooleanVar()
        self.cf_x = tk.DoubleVar()
        self.cf_y = tk.DoubleVar()
        self.cf_z = tk.DoubleVar()
        self.cf_roll = tk.DoubleVar()
        self.cf_pitch = tk.DoubleVar()
        self.cf_yaw = tk.DoubleVar()
        self.max_vx = tk.DoubleVar()
        self.max_vy = tk.DoubleVar()
        self.max_vz = tk.DoubleVar()
        self.max_yaw_rate = tk.DoubleVar()
        self.collision_radius = tk.DoubleVar()
        self.propellers_animate = tk.BooleanVar()

        self.onboard_enabled = tk.BooleanVar()
        self.onboard_x = tk.DoubleVar()
        self.onboard_y = tk.DoubleVar()
        self.onboard_z = tk.DoubleVar()
        self.onboard_roll = tk.DoubleVar()
        self.onboard_pitch = tk.DoubleVar()
        self.onboard_yaw = tk.DoubleVar()
        self.onboard_focal = tk.DoubleVar()
        self.onboard_width = tk.IntVar()
        self.onboard_height = tk.IntVar()
        self.onboard_capture_rgb = tk.BooleanVar()

        self.iso_enabled = tk.BooleanVar()
        self.iso_x = tk.DoubleVar()
        self.iso_y = tk.DoubleVar()
        self.iso_z = tk.DoubleVar()
        self.iso_look_x = tk.DoubleVar()
        self.iso_look_y = tk.DoubleVar()
        self.iso_look_z = tk.DoubleVar()
        self.iso_auto_frame = tk.BooleanVar()
        self.iso_azimuth = tk.DoubleVar()
        self.iso_elevation = tk.DoubleVar()
        self.iso_distance_scale = tk.DoubleVar()
        self.iso_padding = tk.DoubleVar()
        self.iso_focal = tk.DoubleVar()
        self.iso_width = tk.IntVar()
        self.iso_height = tk.IntVar()
        self.iso_capture_rgb = tk.BooleanVar()

        self.runtime_fps = tk.DoubleVar()
        self.runtime_control_mode = tk.StringVar()
        self.runtime_duration_s = tk.DoubleVar()
        self.telemetry_enabled = tk.BooleanVar()
        self.telemetry_output_root = tk.StringVar()
        self.write_latest_json = tk.BooleanVar()
        self.write_history_csv = tk.BooleanVar()
        self.print_every_s = tk.DoubleVar()
        self.csv_every_s = tk.DoubleVar()
        self.save_usd_path = tk.StringVar()

    def _install_map_redraw_traces(self) -> None:
        for variable in [
            self.room_size_x,
            self.room_size_y,
            self.room_floor_color,
            self.room_wall_color,
            self.room_floor_opacity,
            self.room_floor_roughness,
            self.room_floor_metallic,
            self.room_floor_reflectance,
            self.room_wall_opacity,
            self.room_wall_roughness,
            self.room_wall_metallic,
            self.room_wall_reflectance,
            self.landmark_x,
            self.landmark_y,
            self.landmark_radius,
            self.landmark_color,
            self.cf_x,
            self.cf_y,
        ]:
            variable.trace_add("write", lambda *_args: self._redraw_all_maps())

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="AWES Isaac Scenario Launcher", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w", pady=(0, 4))

        description = ttk.Label(
            main,
            text=(
                "Select the scenario context first. The selected scenario controls which tabs, "
                "fields, profile validator, and Isaac runner are used."
            ),
            wraplength=900,
        )
        description.pack(anchor="w", pady=(0, 8))

        selector = ttk.LabelFrame(main, text="Scenario and profile", padding=8)
        selector.pack(fill="x", pady=(0, 8))
        selector.columnconfigure(1, weight=1)

        ttk.Label(selector, text="Scenario").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        scenario_box = ttk.Combobox(
            selector,
            textvariable=self.scenario_label,
            values=[spec.label for spec in self.scenarios],
            state="readonly",
            width=42,
        )
        scenario_box.grid(row=0, column=1, sticky="ew", pady=4)
        scenario_box.bind("<<ComboboxSelected>>", self._on_scenario_selected)

        ttk.Label(selector, text="Profile JSON").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(selector, textvariable=self.profile_path).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(selector, text="Browse", command=self._on_browse_profile).grid(row=1, column=2, padx=(8, 0), pady=4)
        ttk.Button(selector, text="Load", command=self._on_load_profile).grid(row=1, column=3, padx=(8, 0), pady=4)

        self.notebook_holder = ttk.Frame(main)
        self.notebook_holder.pack(fill="both", expand=True)

        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Label(bottom, textvariable=self.status_text, foreground="gray", wraplength=880).pack(anchor="w", pady=(0, 8))

        buttons = ttk.Frame(bottom)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Save", command=self._on_save).pack(side="left")
        ttk.Button(buttons, text="Save As", command=self._on_save_as).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Validate", command=self._on_validate).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Refresh views", command=self._on_refresh_views).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Cancel", command=self._on_cancel).pack(side="right", padx=(8, 0))
        ttk.Button(buttons, text="Run Isaac Sim", command=self._on_run).pack(side="right")

    def _select_scenario(self, scenario_id: str) -> None:
        spec = self.spec_by_id[scenario_id]
        self.current_spec = spec
        self.scenario_label.set(spec.label)
        default_path = spec.default_profile_path(self.project_root)
        self.profile_path.set(str(default_path))
        self._load_profile_from_path(default_path)
        self._rebuild_scenario_tabs()

    def _on_scenario_selected(self, _event: tk.Event | None = None) -> None:
        label = self.scenario_label.get()
        spec = self.spec_by_label[label]
        self._select_scenario(spec.scenario_id)

    def _on_browse_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Select profile JSON",
            initialdir=str(self.project_root / "profiles"),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.profile_path.set(path)

    def _on_load_profile(self) -> None:
        try:
            self._load_profile_from_path(Path(self.profile_path.get()).resolve())
            self._rebuild_scenario_tabs()
            self.status_text.set(f"Loaded profile: {self.profile_path.get()}")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _load_profile_from_path(self, path: Path) -> None:
        if self.current_spec.scenario_id == "crazyflie_room_basic":
            if path.exists():
                profile = load_crazyflie_room_profile(path)
            else:
                profile = default_crazyflie_room_profile()
            self.current_profile = profile
            self._load_crazyflie_profile_into_gui(profile)
        else:
            if not path.exists():
                raise FileNotFoundError(f"Profile does not exist: {path}")
            with path.open("r", encoding="utf-8") as file:
                profile = json.load(file)
            self.current_profile = profile

    def _rebuild_scenario_tabs(self) -> None:
        for child in self.notebook_holder.winfo_children():
            child.destroy()

        if self.current_spec.scenario_id == "crazyflie_room_basic":
            notebook = ttk.Notebook(self.notebook_holder)
            notebook.pack(fill="both", expand=True)

            scenario_tab = ScrollableFrame(notebook)
            room_tab = ScrollableFrame(notebook)
            scene_map_tab = ScrollableFrame(notebook)
            obstacles_tab = ScrollableFrame(notebook)
            landmark_tab = ScrollableFrame(notebook)
            crazyflie_tab = ScrollableFrame(notebook)
            cameras_tab = ScrollableFrame(notebook)
            runtime_tab = ScrollableFrame(notebook)

            notebook.add(scenario_tab, text="Scenario")
            notebook.add(room_tab, text="Room")
            notebook.add(scene_map_tab, text="Scene Map")
            notebook.add(obstacles_tab, text="Obstacles")
            notebook.add(landmark_tab, text="Landmark")
            notebook.add(crazyflie_tab, text="Crazyflie")
            notebook.add(cameras_tab, text="Cameras")
            notebook.add(runtime_tab, text="Runtime")

            self._build_scenario_tab(scenario_tab.content)
            self._build_room_tab(room_tab.content)
            self._build_scene_map_tab(scene_map_tab.content)
            self._build_obstacles_tab(obstacles_tab.content)
            self._build_landmark_tab(landmark_tab.content)
            self._build_crazyflie_tab(crazyflie_tab.content)
            self._build_cameras_tab(cameras_tab.content)
            self._build_runtime_tab(runtime_tab.content)
            self._refresh_obstacle_tree()
            self._refresh_scene_element_tree()
            self.status_text.set("Crazyflie profile ready.")
            return

        frame = ttk.Frame(self.notebook_holder, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="Legacy scenario registered but not yet edited by the new launcher.",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        ttk.Label(
            frame,
            text=(
                "This keeps the architecture open for the Bixler/glider path without forcing "
                "Crazyflie fields into a glider form. You can still run the selected glider "
                "profile from here, but use the old main.py GUI when you need the rich glider editor."
            ),
            wraplength=840,
        ).pack(anchor="w")
        self.status_text.set("Legacy glider profile loaded. Editing is intentionally not implemented in this Step 1 launcher.")

    def _build_scenario_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Scene name").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.scene_name, width=36).grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        row += 1
        ttk.Label(
            parent,
            text=(
                "Step 1 builds a single Crazyflie room scenario. The profile is intentionally "
                "structured so later dynamic scenarios and multiple robots can be added without "
                "rewriting the launcher."
            ),
            wraplength=820,
            foreground="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_room_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        self._add_float_row(parent, row, "Room size X [m]", self.room_size_x); row += 1
        self._add_float_row(parent, row, "Room size Y [m]", self.room_size_y); row += 1
        self._add_float_row(parent, row, "Room size Z [m]", self.room_size_z); row += 1
        self._add_float_row(parent, row, "Wall thickness [m]", self.room_wall_thickness); row += 1
        ttk.Checkbutton(parent, text="Add roof", variable=self.room_has_roof).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_color_row(parent, row, "Floor color", self.room_floor_color); row += 1
        self._add_color_row(parent, row, "Wall color", self.room_wall_color); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Room visual material", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        self._add_float_row(parent, row, "Floor opacity [0-1]", self.room_floor_opacity); row += 1
        self._add_float_row(parent, row, "Floor roughness [0-1]", self.room_floor_roughness); row += 1
        self._add_float_row(parent, row, "Floor metallic [0-1]", self.room_floor_metallic); row += 1
        self._add_float_row(parent, row, "Floor reflectance [0-1]", self.room_floor_reflectance); row += 1
        self._add_float_row(parent, row, "Wall opacity [0-1]", self.room_wall_opacity); row += 1
        self._add_float_row(parent, row, "Wall roughness [0-1]", self.room_wall_roughness); row += 1
        self._add_float_row(parent, row, "Wall metallic [0-1]", self.room_wall_metallic); row += 1
        self._add_float_row(parent, row, "Wall reflectance [0-1]", self.room_wall_reflectance); row += 1

    def _build_scene_map_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.LabelFrame(parent, text="Scene elements", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        map_box = ttk.LabelFrame(left, text="Top-down scene map", padding=6)
        map_box.pack(fill="x", pady=(0, 8))
        self._scene_map_canvas = tk.Canvas(
            map_box, width=520, height=360, background="#f6f6f6", highlightthickness=1, highlightbackground="#999999"
        )
        self._scene_map_canvas.pack(fill="x", expand=False)
        self._scene_map_canvas.bind("<Configure>", lambda _event: self._redraw_scene_map())
        self._scene_map_canvas.bind("<Button-1>", self._on_scene_map_click)
        ttk.Label(
            map_box,
            text=(
                "Click an element to select it. If Crazyflie, landmark, or an obstacle is selected, "
                "click empty room space to move it in X/Y. Room walls/floor are selectable for visual-material editing, not movement."
            ),
            foreground="gray",
            wraplength=520,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(map_box, text="Refresh map from all tabs", command=self._on_refresh_views).pack(anchor="w", pady=(6, 0))

        add_row = ttk.Frame(left)
        add_row.pack(fill="x", pady=(0, 8))
        ttk.Label(add_row, text="New obstacle kind").pack(side="left")
        ttk.Combobox(add_row, textvariable=self.map_add_kind, values=OBSTACLE_KINDS, state="readonly", width=16).pack(side="left", padx=(8, 0))
        ttk.Label(add_row, text="Click empty room space with no movable element selected to add this obstacle.", foreground="gray").pack(side="left", padx=(8, 0))

        columns = ("type", "xy", "z", "editable")
        self._scene_element_tree = ttk.Treeview(right, columns=columns, show="tree headings", height=14)
        self._scene_element_tree.heading("#0", text="Element")
        self._scene_element_tree.heading("type", text="Type")
        self._scene_element_tree.heading("xy", text="X/Y")
        self._scene_element_tree.heading("z", text="Z")
        self._scene_element_tree.heading("editable", text="Map move")
        self._scene_element_tree.column("#0", width=150)
        self._scene_element_tree.column("type", width=90)
        self._scene_element_tree.column("xy", width=120)
        self._scene_element_tree.column("z", width=60)
        self._scene_element_tree.column("editable", width=80, anchor="center")
        self._scene_element_tree.pack(fill="both", expand=True)
        self._scene_element_tree.bind("<<TreeviewSelect>>", self._on_scene_element_selected)

        ttk.Label(
            right,
            text=(
                "Detailed fields stay in their own tabs: Room, Obstacles, Landmark, Crazyflie, Cameras. "
                "This map is for fast selection and X/Y alignment."
            ),
            foreground="gray",
            wraplength=420,
        ).pack(anchor="w", pady=(8, 0))
        right.columnconfigure(0, weight=1)

    def _build_obstacles_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.LabelFrame(parent, text="Selected obstacle", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        map_box = ttk.LabelFrame(left, text="Top-down obstacle map", padding=6)
        map_box.pack(fill="x", pady=(0, 8))
        self._obstacle_map_canvas = tk.Canvas(
            map_box, width=430, height=280, background="#f6f6f6", highlightthickness=1, highlightbackground="#999999"
        )
        self._obstacle_map_canvas.pack(fill="x", expand=False)
        self._obstacle_map_canvas.bind("<Configure>", lambda _event: self._redraw_obstacle_map())
        self._obstacle_map_canvas.bind("<Button-1>", self._on_obstacle_map_click)
        ttk.Label(
            map_box,
            text="Click inside the room to add a new obstacle. If an obstacle is selected, click to move it in X/Y.",
            foreground="gray",
            wraplength=430,
        ).pack(anchor="w", pady=(4, 0))
        ttk.Button(map_box, text="Refresh map from all tabs", command=self._on_refresh_views).pack(anchor="w", pady=(6, 0))

        columns = ("enabled", "kind", "position", "size", "color", "negative", "collision")
        self.obstacle_tree = ttk.Treeview(left, columns=columns, show="tree headings", height=10)
        self.obstacle_tree.heading("#0", text="Name")
        self.obstacle_tree.heading("enabled", text="On")
        self.obstacle_tree.heading("kind", text="Kind")
        self.obstacle_tree.heading("position", text="Position")
        self.obstacle_tree.heading("size", text="Size")
        self.obstacle_tree.heading("color", text="Color")
        self.obstacle_tree.heading("negative", text="Negative")
        self.obstacle_tree.heading("collision", text="Collision")
        self.obstacle_tree.column("#0", width=110)
        self.obstacle_tree.column("enabled", width=42, anchor="center")
        self.obstacle_tree.column("kind", width=80)
        self.obstacle_tree.column("position", width=125)
        self.obstacle_tree.column("size", width=125)
        self.obstacle_tree.column("color", width=78)
        self.obstacle_tree.column("negative", width=72, anchor="center")
        self.obstacle_tree.column("collision", width=70, anchor="center")
        self.obstacle_tree.pack(fill="both", expand=True)
        self.obstacle_tree.bind("<<TreeviewSelect>>", self._on_obstacle_selected)

        obstacle_buttons = ttk.Frame(left)
        obstacle_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(obstacle_buttons, text="Add box", command=lambda: self._on_add_obstacle("box")).pack(side="left")
        ttk.Button(obstacle_buttons, text="Add wall", command=lambda: self._on_add_obstacle("wall")).pack(side="left", padx=(8, 0))
        ttk.Button(obstacle_buttons, text="Remove selected", command=self._on_remove_obstacle).pack(side="left", padx=(8, 0))

        row = 0
        ttk.Label(right, text="Name").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(right, textvariable=self.obs_name).grid(row=row, column=1, sticky="ew", padx=8, pady=4); row += 1
        ttk.Checkbutton(right, text="Enabled", variable=self.obs_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Label(right, text="Kind").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(right, textvariable=self.obs_kind, values=OBSTACLE_KINDS, state="readonly", width=16).grid(row=row, column=1, sticky="w", padx=8, pady=4); row += 1
        self._add_float_row(right, row, "Position X [m]", self.obs_x); row += 1
        self._add_float_row(right, row, "Position Y [m]", self.obs_y); row += 1
        self._add_float_row(right, row, "Position Z [m]", self.obs_z); row += 1
        self._add_float_row(right, row, "Size X [m]", self.obs_sx); row += 1
        self._add_float_row(right, row, "Size Y [m]", self.obs_sy); row += 1
        self._add_float_row(right, row, "Size Z [m]", self.obs_sz); row += 1
        self._add_color_row(right, row, "Color", self.obs_color); row += 1
        ttk.Checkbutton(right, text="Collision enabled", variable=self.obs_collision).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Checkbutton(
            right,
            text="Negative / wall cutout request",
            variable=self.obs_negative,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Label(right, text="Visual material", font=("Segoe UI", 9, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4)); row += 1
        self._add_float_row(right, row, "Opacity [0-1]", self.obs_opacity); row += 1
        self._add_float_row(right, row, "Roughness [0-1]", self.obs_roughness); row += 1
        self._add_float_row(right, row, "Metallic [0-1]", self.obs_metallic); row += 1
        self._add_float_row(right, row, "Reflectance [0-1]", self.obs_reflectance); row += 1
        ttk.Button(right, text="Apply obstacle edits", command=self._on_apply_obstacle).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0)); row += 1
        ttk.Label(
            right,
            text="Negative obstacles are invisible in Isaac. They become rectangular wall cutouts only when they touch or nearly touch a room wall; otherwise they are ignored by geometry/collision.",
            foreground="gray",
            wraplength=360,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        right.columnconfigure(1, weight=1)

    def _build_landmark_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Checkbutton(parent, text="Enable luminous landmark", variable=self.landmark_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Position X [m]", self.landmark_x); row += 1
        self._add_float_row(parent, row, "Position Y [m]", self.landmark_y); row += 1
        self._add_float_row(parent, row, "Position Z [m]", self.landmark_z); row += 1
        self._add_float_row(parent, row, "Radius [m]", self.landmark_radius); row += 1
        self._add_color_row(parent, row, "Color", self.landmark_color); row += 1
        self._add_float_row(parent, row, "Intensity", self.landmark_intensity); row += 1
        self._add_float_row(parent, row, "Exposure", self.landmark_exposure); row += 1

    def _build_crazyflie_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Asset", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        ttk.Label(parent, text="Asset mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.asset_mode, values=["isaac_builtin", "local_usd"], state="readonly", width=18).grid(row=row, column=1, sticky="w", padx=8, pady=4); row += 1
        ttk.Label(parent, text="Local USD path").grid(row=row, column=0, sticky="w", pady=4)
        path_row = ttk.Frame(parent)
        path_row.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        path_row.columnconfigure(0, weight=1)
        ttk.Entry(path_row, textvariable=self.local_usd_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(path_row, text="Browse", command=self._on_browse_local_usd).grid(row=0, column=1, padx=(8, 0)); row += 1
        ttk.Checkbutton(parent, text="Allow placeholder if asset is missing - debug only", variable=self.allow_placeholder_if_missing).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Initial pose", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        self._add_float_row(parent, row, "Position X [m]", self.cf_x); row += 1
        self._add_float_row(parent, row, "Position Y [m]", self.cf_y); row += 1
        self._add_float_row(parent, row, "Position Z [m]", self.cf_z); row += 1
        self._add_float_row(parent, row, "Roll [deg]", self.cf_roll); row += 1
        self._add_float_row(parent, row, "Pitch [deg]", self.cf_pitch); row += 1
        self._add_float_row(parent, row, "Yaw [deg]", self.cf_yaw); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Limits", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        self._add_float_row(parent, row, "Max vx [m/s]", self.max_vx); row += 1
        self._add_float_row(parent, row, "Max vy [m/s]", self.max_vy); row += 1
        self._add_float_row(parent, row, "Max vz [m/s]", self.max_vz); row += 1
        self._add_float_row(parent, row, "Max yaw rate [deg/s]", self.max_yaw_rate); row += 1
        self._add_float_row(parent, row, "Collision radius [m]", self.collision_radius); row += 1
        ttk.Checkbutton(parent, text="Animate propellers if prims exist", variable=self.propellers_animate).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

    def _build_cameras_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Onboard camera", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        ttk.Checkbutton(parent, text="Enable onboard camera prim", variable=self.onboard_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Onboard X [m]", self.onboard_x); row += 1
        self._add_float_row(parent, row, "Onboard Y [m]", self.onboard_y); row += 1
        self._add_float_row(parent, row, "Onboard Z [m]", self.onboard_z); row += 1
        self._add_float_row(parent, row, "Onboard roll [deg]", self.onboard_roll); row += 1
        self._add_float_row(parent, row, "Onboard pitch [deg]", self.onboard_pitch); row += 1
        self._add_float_row(parent, row, "Onboard yaw [deg]", self.onboard_yaw); row += 1
        self._add_float_row(parent, row, "Onboard focal length [mm]", self.onboard_focal); row += 1
        self._add_int_row(parent, row, "Onboard width", self.onboard_width); row += 1
        self._add_int_row(parent, row, "Onboard height", self.onboard_height); row += 1
        ttk.Checkbutton(parent, text="Capture RGB to outputs/<scene>/camera_onboard", variable=self.onboard_capture_rgb).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Isometric camera", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        ttk.Checkbutton(parent, text="Enable isometric camera prim", variable=self.iso_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Eye X [m]", self.iso_x); row += 1
        self._add_float_row(parent, row, "Eye Y [m]", self.iso_y); row += 1
        self._add_float_row(parent, row, "Eye Z [m]", self.iso_z); row += 1
        self._add_float_row(parent, row, "Look-at X [m]", self.iso_look_x); row += 1
        self._add_float_row(parent, row, "Look-at Y [m]", self.iso_look_y); row += 1
        self._add_float_row(parent, row, "Look-at Z [m]", self.iso_look_z); row += 1
        ttk.Checkbutton(parent, text="Auto-frame entire room from angle", variable=self.iso_auto_frame).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Auto-frame azimuth [deg]", self.iso_azimuth); row += 1
        self._add_float_row(parent, row, "Auto-frame elevation [deg]", self.iso_elevation); row += 1
        self._add_float_row(parent, row, "Auto-frame distance scale", self.iso_distance_scale); row += 1
        self._add_float_row(parent, row, "Auto-frame padding [m]", self.iso_padding); row += 1
        self._add_float_row(parent, row, "Isometric focal length [mm]", self.iso_focal); row += 1
        self._add_int_row(parent, row, "Isometric width", self.iso_width); row += 1
        self._add_int_row(parent, row, "Isometric height", self.iso_height); row += 1
        ttk.Checkbutton(parent, text="Capture RGB to outputs/<scene>/camera_isometric", variable=self.iso_capture_rgb).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

    def _build_runtime_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(1, weight=1)
        row = 0
        self._add_float_row(parent, row, "FPS", self.runtime_fps); row += 1
        ttk.Label(parent, text="Control mode").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=self.runtime_control_mode, values=CONTROL_MODES, state="readonly", width=22).grid(row=row, column=1, sticky="w", padx=8, pady=4); row += 1
        self._add_float_row(parent, row, "Duration [s], 0 = until closed", self.runtime_duration_s); row += 1
        ttk.Checkbutton(parent, text="Enable telemetry", variable=self.telemetry_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Label(parent, text="Telemetry output root").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=self.telemetry_output_root).grid(row=row, column=1, sticky="ew", padx=8, pady=4); row += 1
        ttk.Checkbutton(parent, text="Write state_latest.json", variable=self.write_latest_json).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Checkbutton(parent, text="Write state_history.csv", variable=self.write_history_csv).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Print every [s]", self.print_every_s); row += 1
        self._add_float_row(parent, row, "CSV every [s]", self.csv_every_s); row += 1
        ttk.Label(parent, text="Save generated USD path").grid(row=row, column=0, sticky="w", pady=4)
        save_row = ttk.Frame(parent)
        save_row.grid(row=row, column=1, sticky="ew", padx=8, pady=4)
        save_row.columnconfigure(0, weight=1)
        ttk.Entry(save_row, textvariable=self.save_usd_path).grid(row=0, column=0, sticky="ew")
        ttk.Button(save_row, text="Browse", command=self._on_browse_save_usd).grid(row=0, column=1, padx=(8, 0)); row += 1
        ttk.Label(
            parent,
            text="RGB camera capture is now available per camera. Keep it off while debugging geometry. Turn on only one camera first if Isaac Sim becomes unstable.",
            foreground="gray",
            wraplength=820,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _load_crazyflie_profile_into_gui(self, profile: dict[str, Any]) -> None:
        self.scene_name.set(profile["scene_name"])

        room = profile["room"]
        self.room_size_x.set(room["size_m"][0])
        self.room_size_y.set(room["size_m"][1])
        self.room_size_z.set(room["size_m"][2])
        self.room_wall_thickness.set(room["wall_thickness_m"])
        self.room_has_roof.set(room["has_roof"])
        self.room_floor_color.set(rgb_to_hex(room["floor_color"]))
        self.room_wall_color.set(rgb_to_hex(room["wall_color"]))
        room_visual = room.get("visual", {})
        floor_visual = room_visual.get("floor", {})
        wall_visual = room_visual.get("walls", {})
        self.room_floor_opacity.set(float(floor_visual.get("opacity", 1.0)))
        self.room_floor_roughness.set(float(floor_visual.get("roughness", 0.65)))
        self.room_floor_metallic.set(float(floor_visual.get("metallic", 0.0)))
        self.room_floor_reflectance.set(float(floor_visual.get("reflectance", 0.18)))
        self.room_wall_opacity.set(float(wall_visual.get("opacity", 1.0)))
        self.room_wall_roughness.set(float(wall_visual.get("roughness", 0.55)))
        self.room_wall_metallic.set(float(wall_visual.get("metallic", 0.0)))
        self.room_wall_reflectance.set(float(wall_visual.get("reflectance", 0.35)))

        landmark = profile["landmark"]
        self.landmark_enabled.set(landmark["enabled"])
        self.landmark_x.set(landmark["position_m"][0])
        self.landmark_y.set(landmark["position_m"][1])
        self.landmark_z.set(landmark["position_m"][2])
        self.landmark_radius.set(landmark["radius_m"])
        self.landmark_color.set(rgb_to_hex(landmark["color"]))
        self.landmark_intensity.set(landmark["intensity"])
        self.landmark_exposure.set(landmark["exposure"])

        cf = profile["crazyflie"]
        self.asset_mode.set(cf["asset"]["mode"])
        self.local_usd_path.set(cf["asset"]["local_usd_path"])
        self.allow_placeholder_if_missing.set(cf["asset"]["allow_placeholder_if_missing"])
        self.cf_x.set(cf["initial_pose"]["position_m"][0])
        self.cf_y.set(cf["initial_pose"]["position_m"][1])
        self.cf_z.set(cf["initial_pose"]["position_m"][2])
        self.cf_roll.set(cf["initial_pose"]["rotation_deg"][0])
        self.cf_pitch.set(cf["initial_pose"]["rotation_deg"][1])
        self.cf_yaw.set(cf["initial_pose"]["rotation_deg"][2])
        self.max_vx.set(cf["limits"]["max_vx_m_s"])
        self.max_vy.set(cf["limits"]["max_vy_m_s"])
        self.max_vz.set(cf["limits"]["max_vz_m_s"])
        self.max_yaw_rate.set(cf["limits"]["max_yaw_rate_deg_s"])
        self.collision_radius.set(cf["limits"]["collision_radius_m"])
        self.propellers_animate.set(cf["propellers"]["animate"])

        onboard = profile["cameras"]["onboard"]
        self.onboard_enabled.set(onboard["enabled"])
        self.onboard_x.set(onboard["position_m"][0])
        self.onboard_y.set(onboard["position_m"][1])
        self.onboard_z.set(onboard["position_m"][2])
        self.onboard_roll.set(onboard["rotation_deg"][0])
        self.onboard_pitch.set(onboard["rotation_deg"][1])
        self.onboard_yaw.set(onboard["rotation_deg"][2])
        self.onboard_focal.set(onboard["focal_length_mm"])
        self.onboard_width.set(onboard["resolution"][0])
        self.onboard_height.set(onboard["resolution"][1])
        self.onboard_capture_rgb.set(onboard["capture_rgb"])

        iso = profile["cameras"]["isometric"]
        self.iso_enabled.set(iso["enabled"])
        self.iso_x.set(iso["position_m"][0])
        self.iso_y.set(iso["position_m"][1])
        self.iso_z.set(iso["position_m"][2])
        self.iso_look_x.set(iso["look_at_m"][0])
        self.iso_look_y.set(iso["look_at_m"][1])
        self.iso_look_z.set(iso["look_at_m"][2])
        self.iso_auto_frame.set(bool(iso.get("auto_frame_room", True)))
        self.iso_azimuth.set(float(iso.get("azimuth_deg", -45.0)))
        self.iso_elevation.set(float(iso.get("elevation_deg", 35.0)))
        self.iso_distance_scale.set(float(iso.get("distance_scale", 1.85)))
        self.iso_padding.set(float(iso.get("padding_m", 0.35)))
        self.iso_focal.set(iso["focal_length_mm"])
        self.iso_width.set(iso["resolution"][0])
        self.iso_height.set(iso["resolution"][1])
        self.iso_capture_rgb.set(iso["capture_rgb"])

        runtime = profile["runtime"]
        self.runtime_fps.set(runtime["fps"])
        self.runtime_control_mode.set(runtime["control_mode"])
        self.runtime_duration_s.set(runtime["duration_s"])
        self.telemetry_enabled.set(runtime["telemetry_enabled"])
        self.telemetry_output_root.set(runtime["telemetry_output_root"])
        self.write_latest_json.set(runtime["write_latest_state_json"])
        self.write_history_csv.set(runtime["write_state_history_csv"])
        self.print_every_s.set(runtime["print_every_s"])
        self.csv_every_s.set(runtime["csv_every_s"])
        self.save_usd_path.set(runtime["save_usd_path"])

    def _sync_selected_obstacle_widgets_into_profile(self, profile: dict[str, Any]) -> None:
        """Copy the obstacle edit panel into the profile before save/run/refresh.

        Without this, a user can change Kind/Negative/Size in the right-hand editor,
        move the object on the map, then press Run and still get the old obstacle.
        That was the cause of the visible-cube symptom when the GUI showed a sphere/cutout.
        """
        index = self.selected_obstacle_index
        if index is None:
            return
        obstacles = profile.get("obstacles", [])
        if not (0 <= index < len(obstacles)):
            return
        # Only sync when the editor has actually been initialized with an obstacle name.
        if not self.obs_name.get().strip():
            return
        obstacles[index] = {
            "name": self.obs_name.get().strip(),
            "enabled": self.obs_enabled.get(),
            "kind": self.obs_kind.get(),
            "position_m": [self.obs_x.get(), self.obs_y.get(), self.obs_z.get()],
            "size_m": [self.obs_sx.get(), self.obs_sy.get(), self.obs_sz.get()],
            "color": hex_to_rgb(self.obs_color.get()),
            "collision": self.obs_collision.get(),
            "negative": self.obs_negative.get(),
            "visual": {
                "opacity": self.obs_opacity.get(),
                "roughness": self.obs_roughness.get(),
                "metallic": self.obs_metallic.get(),
                "reflectance": self.obs_reflectance.get(),
            },
        }

    def _on_refresh_views(self) -> None:
        try:
            profile = self._collect_profile_from_gui()
            if self.current_spec.scenario_id == "crazyflie_room_basic":
                validate_crazyflie_room_profile(profile)
            self.current_profile = profile
            self._refresh_obstacle_tree()
            self._refresh_scene_element_tree()
            self._redraw_all_maps()
            self.status_text.set("Views refreshed from all tabs. Pending obstacle edits were also applied.")
        except Exception as exc:
            messagebox.showerror("Refresh failed", str(exc))

    def _collect_profile_from_gui(self) -> dict[str, Any]:
        if self.current_spec.scenario_id != "crazyflie_room_basic":
            if self.current_profile is None:
                raise ValueError("No profile loaded.")
            return copy.deepcopy(self.current_profile)

        profile = copy.deepcopy(self.current_profile or default_crazyflie_room_profile())
        self._sync_selected_obstacle_widgets_into_profile(profile)
        profile["scene_name"] = self.scene_name.get().strip()
        profile["scenario_id"] = "crazyflie_room_basic"

        profile["room"].update(
            {
                "enabled": True,
                "size_m": [self.room_size_x.get(), self.room_size_y.get(), self.room_size_z.get()],
                "wall_thickness_m": self.room_wall_thickness.get(),
                "has_roof": self.room_has_roof.get(),
                "floor_color": hex_to_rgb(self.room_floor_color.get()),
                "wall_color": hex_to_rgb(self.room_wall_color.get()),
                "visual": {
                    "floor": {
                        "opacity": self.room_floor_opacity.get(),
                        "roughness": self.room_floor_roughness.get(),
                        "metallic": self.room_floor_metallic.get(),
                        "reflectance": self.room_floor_reflectance.get(),
                    },
                    "walls": {
                        "opacity": self.room_wall_opacity.get(),
                        "roughness": self.room_wall_roughness.get(),
                        "metallic": self.room_wall_metallic.get(),
                        "reflectance": self.room_wall_reflectance.get(),
                    },
                },
            }
        )

        profile["landmark"].update(
            {
                "enabled": self.landmark_enabled.get(),
                "kind": "sphere_light",
                "position_m": [self.landmark_x.get(), self.landmark_y.get(), self.landmark_z.get()],
                "radius_m": self.landmark_radius.get(),
                "color": hex_to_rgb(self.landmark_color.get()),
                "intensity": self.landmark_intensity.get(),
                "exposure": self.landmark_exposure.get(),
            }
        )

        cf = profile["crazyflie"]
        cf["asset"]["mode"] = self.asset_mode.get()
        cf["asset"]["local_usd_path"] = self.local_usd_path.get().strip()
        cf["asset"]["allow_placeholder_if_missing"] = self.allow_placeholder_if_missing.get()
        cf["initial_pose"]["position_m"] = [self.cf_x.get(), self.cf_y.get(), self.cf_z.get()]
        cf["initial_pose"]["rotation_deg"] = [self.cf_roll.get(), self.cf_pitch.get(), self.cf_yaw.get()]
        cf["limits"].update(
            {
                "max_vx_m_s": self.max_vx.get(),
                "max_vy_m_s": self.max_vy.get(),
                "max_vz_m_s": self.max_vz.get(),
                "max_yaw_rate_deg_s": self.max_yaw_rate.get(),
                "collision_radius_m": self.collision_radius.get(),
            }
        )
        cf["propellers"]["animate"] = self.propellers_animate.get()

        onboard = profile["cameras"]["onboard"]
        onboard.update(
            {
                "enabled": self.onboard_enabled.get(),
                "position_m": [self.onboard_x.get(), self.onboard_y.get(), self.onboard_z.get()],
                "rotation_deg": [self.onboard_roll.get(), self.onboard_pitch.get(), self.onboard_yaw.get()],
                "focal_length_mm": self.onboard_focal.get(),
                "resolution": [self.onboard_width.get(), self.onboard_height.get()],
                "capture_rgb": self.onboard_capture_rgb.get(),
                "capture_every_n_frames": int(onboard.get("capture_every_n_frames", 1)),
            }
        )

        iso = profile["cameras"]["isometric"]
        iso.update(
            {
                "enabled": self.iso_enabled.get(),
                "position_m": [self.iso_x.get(), self.iso_y.get(), self.iso_z.get()],
                "look_at_m": [self.iso_look_x.get(), self.iso_look_y.get(), self.iso_look_z.get()],
                "auto_frame_room": self.iso_auto_frame.get(),
                "azimuth_deg": self.iso_azimuth.get(),
                "elevation_deg": self.iso_elevation.get(),
                "distance_scale": self.iso_distance_scale.get(),
                "padding_m": self.iso_padding.get(),
                "focal_length_mm": self.iso_focal.get(),
                "resolution": [self.iso_width.get(), self.iso_height.get()],
                "capture_rgb": self.iso_capture_rgb.get(),
                "capture_every_n_frames": int(iso.get("capture_every_n_frames", 1)),
            }
        )

        profile["runtime"].update(
            {
                "fps": self.runtime_fps.get(),
                "control_mode": self.runtime_control_mode.get(),
                "duration_s": self.runtime_duration_s.get(),
                "telemetry_enabled": self.telemetry_enabled.get(),
                "telemetry_output_root": self.telemetry_output_root.get().strip(),
                "write_latest_state_json": self.write_latest_json.get(),
                "write_state_history_csv": self.write_history_csv.get(),
                "print_every_s": self.print_every_s.get(),
                "csv_every_s": self.csv_every_s.get(),
                "save_usd_path": self.save_usd_path.get().strip(),
            }
        )

        return profile

    def _refresh_obstacle_tree(self) -> None:
        if not hasattr(self, "obstacle_tree"):
            return
        for item in self.obstacle_tree.get_children():
            self.obstacle_tree.delete(item)
        if not self.current_profile:
            return
        for index, obstacle in enumerate(self.current_profile.get("obstacles", [])):
            self.obstacle_tree.insert(
                "",
                "end",
                iid=str(index),
                text=obstacle["name"],
                values=(
                    "yes" if obstacle.get("enabled", True) else "no",
                    obstacle.get("kind", "box"),
                    fmt_vec(obstacle.get("position_m", [])),
                    fmt_vec(obstacle.get("size_m", [])),
                    rgb_to_hex(obstacle.get("color", [0.8, 0.25, 0.15])),
                    "yes" if obstacle.get("negative", False) else "no",
                    "yes" if obstacle.get("collision", True) else "no",
                ),
            )
        self._redraw_all_maps()

    def _on_obstacle_selected(self, _event: tk.Event | None = None) -> None:
        selection = self.obstacle_tree.selection()
        if not selection or self.current_profile is None:
            self.selected_obstacle_index = None
            return
        index = int(selection[0])
        self.selected_obstacle_index = index
        self.selected_scene_element.set(f"obstacle:{index}")
        obstacle = self.current_profile["obstacles"][index]
        self.obs_name.set(obstacle["name"])
        self.obs_enabled.set(obstacle["enabled"])
        self.obs_kind.set(obstacle["kind"])
        self.obs_x.set(obstacle["position_m"][0])
        self.obs_y.set(obstacle["position_m"][1])
        self.obs_z.set(obstacle["position_m"][2])
        self.obs_sx.set(obstacle["size_m"][0])
        self.obs_sy.set(obstacle["size_m"][1])
        self.obs_sz.set(obstacle["size_m"][2])
        self.obs_color.set(rgb_to_hex(obstacle["color"]))
        self.obs_collision.set(obstacle.get("collision", True))
        self.obs_negative.set(obstacle.get("negative", False))
        visual = obstacle.get("visual", {})
        self.obs_opacity.set(float(visual.get("opacity", 1.0)))
        self.obs_roughness.set(float(visual.get("roughness", 0.55)))
        self.obs_metallic.set(float(visual.get("metallic", 0.0)))
        self.obs_reflectance.set(float(visual.get("reflectance", 0.35)))
        self._redraw_all_maps()

    def _on_add_obstacle(self, kind: str = "box", x: float = 0.0, y: float = 0.0) -> None:
        if self.current_profile is None:
            return
        kind = kind if kind in OBSTACLE_KINDS else "box"
        count = len(self.current_profile.setdefault("obstacles", [])) + 1
        if kind == "wall":
            name = f"wall_{count:02d}"
            size = [0.08, 1.0, 1.0]
            z = 0.5
            color = [0.65, 0.65, 0.70]
            reflectance = 0.45
        elif kind == "sphere":
            name = f"sphere_{count:02d}"
            size = [0.3, 0.3, 0.3]
            z = 0.15
            color = [0.8, 0.25, 0.15]
            reflectance = 0.35
        elif kind in {"cylinder", "pillar"}:
            name = f"{kind}_{count:02d}"
            size = [0.25, 0.25, 0.8]
            z = 0.4
            color = [0.45, 0.45, 0.45]
            reflectance = 0.4
        elif kind == "floor_patch":
            name = f"floor_patch_{count:02d}"
            size = [0.6, 0.6, 0.02]
            z = 0.01
            color = [0.15, 0.55, 0.2]
            reflectance = 0.2
        else:
            name = f"box_{count:02d}"
            size = [0.3, 0.3, 0.5]
            z = 0.25
            color = [0.8, 0.25, 0.15]
            reflectance = 0.35

        obstacle = {
            "name": name,
            "enabled": True,
            "kind": kind,
            "position_m": [float(x), float(y), float(z)],
            "size_m": size,
            "color": color,
            "collision": True,
            "negative": False,
            "visual": {
                "opacity": 1.0,
                "roughness": 0.55,
                "metallic": 0.0,
                "reflectance": reflectance,
            },
        }
        self.current_profile["obstacles"].append(obstacle)
        self.selected_obstacle_index = len(self.current_profile["obstacles"]) - 1
        self._refresh_obstacle_tree()
        self._refresh_scene_element_tree()
        self.selected_scene_element.set(f"obstacle:{self.selected_obstacle_index}")
        if hasattr(self, "obstacle_tree"):
            self.obstacle_tree.selection_set(str(self.selected_obstacle_index))
            self.obstacle_tree.see(str(self.selected_obstacle_index))
            self._on_obstacle_selected()
        self.status_text.set(f"Added {kind} obstacle at x={x:.3f}, y={y:.3f}.")

    def _on_remove_obstacle(self) -> None:
        if self.current_profile is None or self.selected_obstacle_index is None:
            messagebox.showinfo("No obstacle selected", "Select an obstacle first.")
            return
        del self.current_profile["obstacles"][self.selected_obstacle_index]
        self.selected_obstacle_index = None
        self._refresh_obstacle_tree()
        self._refresh_scene_element_tree()
        self._redraw_all_maps()

    def _on_apply_obstacle(self) -> None:
        if self.current_profile is None or self.selected_obstacle_index is None:
            messagebox.showinfo("No obstacle selected", "Select an obstacle first.")
            return
        self.current_profile["obstacles"][self.selected_obstacle_index] = {
            "name": self.obs_name.get().strip(),
            "enabled": self.obs_enabled.get(),
            "kind": self.obs_kind.get(),
            "position_m": [self.obs_x.get(), self.obs_y.get(), self.obs_z.get()],
            "size_m": [self.obs_sx.get(), self.obs_sy.get(), self.obs_sz.get()],
            "color": hex_to_rgb(self.obs_color.get()),
            "collision": self.obs_collision.get(),
            "negative": self.obs_negative.get(),
            "visual": {
                "opacity": self.obs_opacity.get(),
                "roughness": self.obs_roughness.get(),
                "metallic": self.obs_metallic.get(),
                "reflectance": self.obs_reflectance.get(),
            },
        }
        try:
            profile = self._collect_profile_from_gui()
            validate_crazyflie_room_profile(profile, project_root=self.project_root)
            self.current_profile = profile
        except Exception as exc:
            messagebox.showerror("Invalid obstacle", str(exc))
            return
        self._refresh_obstacle_tree()
        self._refresh_scene_element_tree()
        self._redraw_all_maps()
        self.status_text.set("Obstacle edits applied.")

    def _on_obstacle_map_click(self, event: tk.Event) -> None:
        if self.current_profile is None or self._obstacle_map_canvas is None:
            return
        coords = self._canvas_to_room_xy(float(event.x), float(event.y))
        if coords is None:
            return
        x, y = coords
        if self.selected_obstacle_index is None:
            self._on_add_obstacle("box", x=x, y=y)
            return

        obstacles = self.current_profile.get("obstacles", [])
        if not (0 <= self.selected_obstacle_index < len(obstacles)):
            self.selected_obstacle_index = None
            self._on_add_obstacle("box", x=x, y=y)
            return

        obstacle = obstacles[self.selected_obstacle_index]
        obstacle["position_m"][0] = x
        obstacle["position_m"][1] = y
        self.obs_x.set(x)
        self.obs_y.set(y)
        self._refresh_obstacle_tree()
        self._refresh_scene_element_tree()
        self.obstacle_tree.selection_set(str(self.selected_obstacle_index))
        self.status_text.set(f"Moved {obstacle.get('name', 'obstacle')} to x={x:.3f}, y={y:.3f}.")

    def _canvas_to_room_xy(self, canvas_x: float, canvas_y: float) -> tuple[float, float] | None:
        if self._obstacle_map_canvas is None:
            return None
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(self._obstacle_map_canvas.winfo_width()), 1.0)
        height = max(float(self._obstacle_map_canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        if not (x0 <= canvas_x <= x0 + room_w and y0 <= canvas_y <= y0 + room_h):
            return None
        x = (canvas_x - x0) / scale - 0.5 * sx
        y = 0.5 * sy - (canvas_y - y0) / scale
        return (x, y)

    def _room_xy_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        assert self._obstacle_map_canvas is not None
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(self._obstacle_map_canvas.winfo_width()), 1.0)
        height = max(float(self._obstacle_map_canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        cx = x0 + (x + 0.5 * sx) * scale
        cy = y0 + (0.5 * sy - y) * scale
        return cx, cy

    def _redraw_obstacle_map(self) -> None:
        canvas = self._obstacle_map_canvas
        if canvas is None:
            return
        canvas.delete("all")
        if self.current_profile is None:
            return

        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(canvas.winfo_width()), 1.0)
        height = max(float(canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        x1 = x0 + room_w
        y1 = y0 + room_h

        floor_color = self.room_floor_color.get() or "#ffffff"
        wall_color = self.room_wall_color.get() or "#d0d0d0"
        canvas.create_rectangle(x0, y0, x1, y1, fill=floor_color, outline=wall_color, width=4)

        # 1 m grid, centered at room origin.
        grid_step = 1.0
        gx = -0.5 * sx
        while gx <= 0.5 * sx + 1e-9:
            cx0, cy0 = self._room_xy_to_canvas(gx, -0.5 * sy)
            cx1, cy1 = self._room_xy_to_canvas(gx, 0.5 * sy)
            canvas.create_line(cx0, cy0, cx1, cy1, fill="#d9d9d9")
            gx += grid_step
        gy = -0.5 * sy
        while gy <= 0.5 * sy + 1e-9:
            cx0, cy0 = self._room_xy_to_canvas(-0.5 * sx, gy)
            cx1, cy1 = self._room_xy_to_canvas(0.5 * sx, gy)
            canvas.create_line(cx0, cy0, cx1, cy1, fill="#d9d9d9")
            gy += grid_step

        cx0, cy0 = self._room_xy_to_canvas(-0.5 * sx, 0.0)
        cx1, cy1 = self._room_xy_to_canvas(0.5 * sx, 0.0)
        canvas.create_line(cx0, cy0, cx1, cy1, fill="#7aa37a")
        cx0, cy0 = self._room_xy_to_canvas(0.0, -0.5 * sy)
        cx1, cy1 = self._room_xy_to_canvas(0.0, 0.5 * sy)
        canvas.create_line(cx0, cy0, cx1, cy1, fill="#b28282")

        # Landmark.
        landmark = self.current_profile.get("landmark", {})
        if landmark.get("enabled", True):
            lx, ly, _lz = [float(v) for v in landmark.get("position_m", [0.0, 0.0, 0.0])]
            lr = max(float(landmark.get("radius_m", 0.08)) * scale, 4.0)
            lc = rgb_to_hex(landmark.get("color", [1.0, 0.85, 0.1]))
            cx, cy = self._room_xy_to_canvas(lx, ly)
            canvas.create_oval(cx - lr, cy - lr, cx + lr, cy + lr, fill=lc, outline="#777777")

        # Crazyflie.
        cf_pose = self.current_profile.get("crazyflie", {}).get("initial_pose", {})
        fx, fy, _fz = [float(v) for v in cf_pose.get("position_m", [0.0, 0.0, 0.0])]
        cx, cy = self._room_xy_to_canvas(fx, fy)
        canvas.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill="#222222", outline="#000000")
        canvas.create_text(cx + 8, cy - 8, text="CF", anchor="w", fill="#222222")

        for index, obstacle in enumerate(self.current_profile.get("obstacles", [])):
            if not obstacle.get("enabled", True):
                continue
            px, py, _pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
            sx_o, sy_o, _sz_o = [float(v) for v in obstacle.get("size_m", [0.3, 0.3, 0.3])]
            ox0, oy0 = self._room_xy_to_canvas(px - 0.5 * sx_o, py + 0.5 * sy_o)
            ox1, oy1 = self._room_xy_to_canvas(px + 0.5 * sx_o, py - 0.5 * sy_o)
            color = rgb_to_hex(obstacle.get("color", [0.8, 0.25, 0.15]))
            selected = index == self.selected_obstacle_index
            outline = "#000000" if selected else "#555555"
            width_px = 3 if selected else 1
            dash = (5, 3) if obstacle.get("negative", False) else None
            kind = obstacle.get("kind", "box")
            if kind in {"sphere", "cylinder", "pillar"}:
                canvas.create_oval(ox0, oy0, ox1, oy1, fill="" if obstacle.get("negative", False) else color, outline=outline, width=width_px, dash=dash)
            else:
                canvas.create_rectangle(ox0, oy0, ox1, oy1, fill="" if obstacle.get("negative", False) else color, outline=outline, width=width_px, dash=dash)
            if obstacle.get("negative", False):
                canvas.create_text((ox0 + ox1) * 0.5, (oy0 + oy1) * 0.5, text=self._negative_obstacle_map_label(obstacle), fill="#333333")

    def _redraw_all_maps(self) -> None:
        self._redraw_obstacle_map()
        self._redraw_scene_map()

    def _refresh_scene_element_tree(self) -> None:
        tree = self._scene_element_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        if self.current_profile is None:
            return
        for element_id, label, element_type, xy, z_value, movable in self._iter_scene_elements():
            tree.insert(
                "",
                "end",
                iid=element_id,
                text=label,
                values=(element_type, xy, z_value, "yes" if movable else "no"),
            )
        selected = self.selected_scene_element.get()
        if selected and tree.exists(selected):
            tree.selection_set(selected)

    def _iter_scene_elements(self) -> list[tuple[str, str, str, str, str, bool]]:
        if self.current_profile is None:
            return []
        elements: list[tuple[str, str, str, str, str, bool]] = []
        elements.append(("room:floor", "Room floor", "room", "origin", "0", False))
        elements.append(("room:walls", "Room walls", "room", "boundary", "full", False))
        cf_pose = self.current_profile.get("crazyflie", {}).get("initial_pose", {})
        cfp = cf_pose.get("position_m", [0.0, 0.0, 0.0])
        elements.append(("crazyflie:main", "Crazyflie", "crazyflie", f"{float(cfp[0]):+.2f}, {float(cfp[1]):+.2f}", f"{float(cfp[2]):.2f}", True))
        landmark = self.current_profile.get("landmark", {})
        lp = landmark.get("position_m", [0.0, 0.0, 0.0])
        elements.append(("landmark:main", "Luminous landmark", "landmark", f"{float(lp[0]):+.2f}, {float(lp[1]):+.2f}", f"{float(lp[2]):.2f}", True))
        for index, obstacle in enumerate(self.current_profile.get("obstacles", [])):
            pos = obstacle.get("position_m", [0.0, 0.0, 0.0])
            elements.append((
                f"obstacle:{index}",
                str(obstacle.get("name", f"obstacle_{index + 1:02d}")),
                str(obstacle.get("kind", "box")),
                f"{float(pos[0]):+.2f}, {float(pos[1]):+.2f}",
                f"{float(pos[2]):.2f}",
                True,
            ))
        return elements

    def _on_scene_element_selected(self, _event: tk.Event | None = None) -> None:
        tree = self._scene_element_tree
        if tree is None:
            return
        selection = tree.selection()
        if not selection:
            self.selected_scene_element.set("")
            self._redraw_scene_map()
            return
        element_id = str(selection[0])
        self.selected_scene_element.set(element_id)
        if element_id.startswith("obstacle:"):
            index = int(element_id.split(":", 1)[1])
            self.selected_obstacle_index = index
            if hasattr(self, "obstacle_tree") and self.obstacle_tree.exists(str(index)):
                self.obstacle_tree.selection_set(str(index))
                self.obstacle_tree.see(str(index))
                self._on_obstacle_selected()
        self._redraw_scene_map()
        self.status_text.set(self._scene_selection_hint(element_id))

    def _scene_selection_hint(self, element_id: str) -> str:
        if element_id == "room:floor":
            return "Selected room floor. Edit floor visual material in the Room tab."
        if element_id == "room:walls":
            return "Selected room walls. Edit wall color/reflectance in the Room tab."
        if element_id == "crazyflie:main":
            return "Selected Crazyflie. Click empty map space to move X/Y; edit Z/angles/limits in the Crazyflie tab."
        if element_id == "landmark:main":
            return "Selected luminous landmark. Click empty map space to move X/Y; edit Z/light settings in the Landmark tab."
        if element_id.startswith("obstacle:"):
            return "Selected obstacle. Click empty map space to move X/Y; edit dimensions/material/negative in the Obstacles tab."
        return "Selected scene element."

    def _on_scene_map_click(self, event: tk.Event) -> None:
        if self.current_profile is None or self._scene_map_canvas is None:
            return
        coords = self._canvas_to_room_xy_for_canvas(self._scene_map_canvas, float(event.x), float(event.y))
        if coords is None:
            return
        hit = self._hit_test_scene_map(float(event.x), float(event.y))
        if hit is not None:
            self.selected_scene_element.set(hit)
            if self._scene_element_tree is not None and self._scene_element_tree.exists(hit):
                self._scene_element_tree.selection_set(hit)
                self._scene_element_tree.see(hit)
            if hit.startswith("obstacle:"):
                self.selected_obstacle_index = int(hit.split(":", 1)[1])
                if hasattr(self, "obstacle_tree") and self.obstacle_tree.exists(str(self.selected_obstacle_index)):
                    self.obstacle_tree.selection_set(str(self.selected_obstacle_index))
                    self._on_obstacle_selected()
            self._redraw_all_maps()
            self.status_text.set(self._scene_selection_hint(hit))
            return

        selected = self.selected_scene_element.get()
        if self._move_selected_scene_element(selected, coords[0], coords[1]):
            self._refresh_scene_element_tree()
            self._refresh_obstacle_tree()
            if self._scene_element_tree is not None and self._scene_element_tree.exists(selected):
                self._scene_element_tree.selection_set(selected)
            self.status_text.set(f"Moved {selected} to x={coords[0]:.3f}, y={coords[1]:.3f}.")
            return

        self._on_add_obstacle(self.map_add_kind.get() or "box", x=coords[0], y=coords[1])
        self.selected_scene_element.set(f"obstacle:{self.selected_obstacle_index}")
        self._refresh_scene_element_tree()

    def _move_selected_scene_element(self, element_id: str, x: float, y: float) -> bool:
        if self.current_profile is None or not element_id:
            return False
        if element_id == "crazyflie:main":
            pose = self.current_profile["crazyflie"]["initial_pose"]["position_m"]
            pose[0] = x
            pose[1] = y
            self.cf_x.set(x)
            self.cf_y.set(y)
            return True
        if element_id == "landmark:main":
            pos = self.current_profile["landmark"]["position_m"]
            pos[0] = x
            pos[1] = y
            self.landmark_x.set(x)
            self.landmark_y.set(y)
            return True
        if element_id.startswith("obstacle:"):
            index = int(element_id.split(":", 1)[1])
            obstacles = self.current_profile.get("obstacles", [])
            if not (0 <= index < len(obstacles)):
                return False
            obstacles[index]["position_m"][0] = x
            obstacles[index]["position_m"][1] = y
            if self.selected_obstacle_index == index:
                self.obs_x.set(x)
                self.obs_y.set(y)
            return True
        return False

    def _negative_obstacle_map_label(self, obstacle: dict[str, Any]) -> str:
        return "wall cut" if self._negative_obstacle_near_room_wall(obstacle) else "neg"

    def _negative_obstacle_near_room_wall(self, obstacle: dict[str, Any]) -> bool:
        try:
            sx = max(float(self.room_size_x.get()), 1e-6)
            sy = max(float(self.room_size_y.get()), 1e-6)
            sz = max(float(self.room_size_z.get()), 1e-6)
            tolerance = max(float(self.room_wall_thickness.get()) * 2.0, 0.10)
            px, py, pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
            ox, oy, oz = [float(v) for v in obstacle.get("size_m", [0.0, 0.0, 0.0])]
            x0 = px - 0.5 * ox
            x1 = px + 0.5 * ox
            y0 = py - 0.5 * oy
            y1 = py + 0.5 * oy
            z0 = pz - 0.5 * oz
            z1 = pz + 0.5 * oz
            if not (z1 > 0.0 and z0 < sz):
                return False
            return (
                x1 >= 0.5 * sx - tolerance
                or x0 <= -0.5 * sx + tolerance
                or y1 >= 0.5 * sy - tolerance
                or y0 <= -0.5 * sy + tolerance
            )
        except Exception:
            return False

    def _hit_test_scene_map(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.current_profile is None or self._scene_map_canvas is None:
            return None
        candidates: list[tuple[float, str]] = []
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        x_left, y_top = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, -0.5 * sx, 0.5 * sy)
        x_right, y_bottom = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, 0.5 * sx, -0.5 * sy)
        if min(abs(canvas_x - x_left), abs(canvas_x - x_right), abs(canvas_y - y_top), abs(canvas_y - y_bottom)) <= 8.0:
            candidates.append((0.0, "room:walls"))
        if x_left + 8.0 < canvas_x < x_right - 8.0 and y_top + 8.0 < canvas_y < y_bottom - 8.0:
            candidates.append((500.0, "room:floor"))

        landmark = self.current_profile.get("landmark", {})
        if landmark.get("enabled", True):
            lx, ly, _lz = [float(v) for v in landmark.get("position_m", [0.0, 0.0, 0.0])]
            cx, cy = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, lx, ly)
            dist = ((canvas_x - cx) ** 2 + (canvas_y - cy) ** 2) ** 0.5
            if dist <= 14.0:
                candidates.append((dist, "landmark:main"))

        cf_pose = self.current_profile.get("crazyflie", {}).get("initial_pose", {})
        fx, fy, _fz = [float(v) for v in cf_pose.get("position_m", [0.0, 0.0, 0.0])]
        cx, cy = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, fx, fy)
        dist = ((canvas_x - cx) ** 2 + (canvas_y - cy) ** 2) ** 0.5
        if dist <= 14.0:
            candidates.append((dist, "crazyflie:main"))

        for index, obstacle in enumerate(self.current_profile.get("obstacles", [])):
            if not obstacle.get("enabled", True):
                continue
            px, py, _pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
            ox, oy, _oz = [float(v) for v in obstacle.get("size_m", [0.3, 0.3, 0.3])]
            x0, y0 = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, px - 0.5 * ox, py + 0.5 * oy)
            x1, y1 = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, px + 0.5 * ox, py - 0.5 * oy)
            if min(x0, x1) - 4.0 <= canvas_x <= max(x0, x1) + 4.0 and min(y0, y1) - 4.0 <= canvas_y <= max(y0, y1) + 4.0:
                center_x, center_y = self._room_xy_to_canvas_for_canvas(self._scene_map_canvas, px, py)
                dist = ((canvas_x - center_x) ** 2 + (canvas_y - center_y) ** 2) ** 0.5
                candidates.append((dist, f"obstacle:{index}"))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _canvas_to_room_xy_for_canvas(self, canvas: tk.Canvas, canvas_x: float, canvas_y: float) -> tuple[float, float] | None:
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(canvas.winfo_width()), 1.0)
        height = max(float(canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        if not (x0 <= canvas_x <= x0 + room_w and y0 <= canvas_y <= y0 + room_h):
            return None
        x = (canvas_x - x0) / scale - 0.5 * sx
        y = 0.5 * sy - (canvas_y - y0) / scale
        return (x, y)

    def _room_xy_to_canvas_for_canvas(self, canvas: tk.Canvas, x: float, y: float) -> tuple[float, float]:
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(canvas.winfo_width()), 1.0)
        height = max(float(canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        cx = x0 + (x + 0.5 * sx) * scale
        cy = y0 + (0.5 * sy - y) * scale
        return cx, cy

    def _draw_scene_map_content(self, canvas: tk.Canvas) -> None:
        if self.current_profile is None:
            return
        sx = max(float(self.room_size_x.get()), 1e-6)
        sy = max(float(self.room_size_y.get()), 1e-6)
        width = max(float(canvas.winfo_width()), 1.0)
        height = max(float(canvas.winfo_height()), 1.0)
        margin = 24.0
        scale = min((width - 2.0 * margin) / sx, (height - 2.0 * margin) / sy)
        room_w = sx * scale
        room_h = sy * scale
        x0 = 0.5 * (width - room_w)
        y0 = 0.5 * (height - room_h)
        x1 = x0 + room_w
        y1 = y0 + room_h
        selected = self.selected_scene_element.get()
        floor_color = self.room_floor_color.get() or "#ffffff"
        wall_color = self.room_wall_color.get() or "#d0d0d0"
        wall_width = 7 if selected == "room:walls" else 4
        canvas.create_rectangle(x0, y0, x1, y1, fill=floor_color, outline=wall_color, width=wall_width)
        if selected == "room:floor":
            canvas.create_rectangle(x0 + 5, y0 + 5, x1 - 5, y1 - 5, outline="#000000", dash=(4, 3), width=2)

        grid_step = 1.0
        gx = -0.5 * sx
        while gx <= 0.5 * sx + 1e-9:
            cx0, cy0 = self._room_xy_to_canvas_for_canvas(canvas, gx, -0.5 * sy)
            cx1, cy1 = self._room_xy_to_canvas_for_canvas(canvas, gx, 0.5 * sy)
            canvas.create_line(cx0, cy0, cx1, cy1, fill="#d9d9d9")
            gx += grid_step
        gy = -0.5 * sy
        while gy <= 0.5 * sy + 1e-9:
            cx0, cy0 = self._room_xy_to_canvas_for_canvas(canvas, -0.5 * sx, gy)
            cx1, cy1 = self._room_xy_to_canvas_for_canvas(canvas, 0.5 * sx, gy)
            canvas.create_line(cx0, cy0, cx1, cy1, fill="#d9d9d9")
            gy += grid_step
        cx0, cy0 = self._room_xy_to_canvas_for_canvas(canvas, -0.5 * sx, 0.0)
        cx1, cy1 = self._room_xy_to_canvas_for_canvas(canvas, 0.5 * sx, 0.0)
        canvas.create_line(cx0, cy0, cx1, cy1, fill="#7aa37a")
        cx0, cy0 = self._room_xy_to_canvas_for_canvas(canvas, 0.0, -0.5 * sy)
        cx1, cy1 = self._room_xy_to_canvas_for_canvas(canvas, 0.0, 0.5 * sy)
        canvas.create_line(cx0, cy0, cx1, cy1, fill="#b28282")

        for index, obstacle in enumerate(self.current_profile.get("obstacles", [])):
            if not obstacle.get("enabled", True):
                continue
            px, py, _pz = [float(v) for v in obstacle.get("position_m", [0.0, 0.0, 0.0])]
            sx_o, sy_o, _sz_o = [float(v) for v in obstacle.get("size_m", [0.3, 0.3, 0.3])]
            ox0, oy0 = self._room_xy_to_canvas_for_canvas(canvas, px - 0.5 * sx_o, py + 0.5 * sy_o)
            ox1, oy1 = self._room_xy_to_canvas_for_canvas(canvas, px + 0.5 * sx_o, py - 0.5 * sy_o)
            color = rgb_to_hex(obstacle.get("color", [0.8, 0.25, 0.15]))
            element_id = f"obstacle:{index}"
            selected_obstacle = selected == element_id
            outline = "#000000" if selected_obstacle else "#555555"
            width_px = 3 if selected_obstacle else 1
            dash = (5, 3) if obstacle.get("negative", False) else None
            kind = obstacle.get("kind", "box")
            fill_color = "" if obstacle.get("negative", False) else color
            if kind in {"sphere", "cylinder", "pillar"}:
                canvas.create_oval(ox0, oy0, ox1, oy1, fill=fill_color, outline=outline, width=width_px, dash=dash)
            else:
                canvas.create_rectangle(ox0, oy0, ox1, oy1, fill=fill_color, outline=outline, width=width_px, dash=dash)
            if obstacle.get("negative", False):
                canvas.create_text((ox0 + ox1) * 0.5, (oy0 + oy1) * 0.5, text=self._negative_obstacle_map_label(obstacle), fill="#333333")

        landmark = self.current_profile.get("landmark", {})
        if landmark.get("enabled", True):
            lx, ly, _lz = [float(v) for v in landmark.get("position_m", [0.0, 0.0, 0.0])]
            lr = max(float(landmark.get("radius_m", 0.08)) * scale, 5.0)
            lc = rgb_to_hex(landmark.get("color", [1.0, 0.85, 0.1]))
            cx, cy = self._room_xy_to_canvas_for_canvas(canvas, lx, ly)
            outline = "#000000" if selected == "landmark:main" else "#777777"
            canvas.create_oval(cx - lr, cy - lr, cx + lr, cy + lr, fill=lc, outline=outline, width=3 if selected == "landmark:main" else 1)
            canvas.create_text(cx + 8, cy - 8, text="LM", anchor="w", fill="#222222")

        cf_pose = self.current_profile.get("crazyflie", {}).get("initial_pose", {})
        fx, fy, _fz = [float(v) for v in cf_pose.get("position_m", [0.0, 0.0, 0.0])]
        cx, cy = self._room_xy_to_canvas_for_canvas(canvas, fx, fy)
        outline = "#000000" if selected == "crazyflie:main" else "#222222"
        canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#222222", outline=outline, width=3 if selected == "crazyflie:main" else 1)
        yaw_deg = float(cf_pose.get("rotation_deg", [0.0, 0.0, 0.0])[2])
        yaw_rad = math.radians(yaw_deg)
        canvas.create_line(cx, cy, cx + 18.0 * math.cos(yaw_rad), cy - 18.0 * math.sin(yaw_rad), fill="#222222", width=2)
        canvas.create_text(cx + 10, cy - 10, text="CF", anchor="w", fill="#222222")

    def _redraw_scene_map(self) -> None:
        canvas = self._scene_map_canvas
        if canvas is None:
            return
        canvas.delete("all")
        self._draw_scene_map_content(canvas)

    def _on_browse_local_usd(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Crazyflie USD",
            initialdir=str(self.project_root / "assets"),
            filetypes=[("USD files", "*.usd *.usda *.usdc"), ("All files", "*.*")],
        )
        if path:
            try:
                relative = Path(path).resolve().relative_to(self.project_root)
                self.local_usd_path.set(str(relative))
            except ValueError:
                self.local_usd_path.set(path)

    def _on_browse_save_usd(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save generated USD as",
            initialdir=str(self.project_root / "outputs"),
            defaultextension=".usd",
            filetypes=[("USD files", "*.usd *.usda"), ("All files", "*.*")],
        )
        if path:
            try:
                relative = Path(path).resolve().relative_to(self.project_root)
                self.save_usd_path.set(str(relative))
            except ValueError:
                self.save_usd_path.set(path)

    def _on_save(self) -> None:
        try:
            profile = self._collect_profile_from_gui()
            path = Path(self.profile_path.get()).resolve()
            if self.current_spec.scenario_id == "crazyflie_room_basic":
                save_crazyflie_room_profile(path, profile)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as file:
                    json.dump(profile, file, indent=2)
            self.current_profile = profile
            self.status_text.set(f"Saved profile: {path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _on_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save profile as",
            initialdir=str(self.project_root / "profiles"),
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.profile_path.set(path)
            self._on_save()

    def _on_validate(self) -> None:
        try:
            profile = self._collect_profile_from_gui()
            if self.current_spec.scenario_id == "crazyflie_room_basic":
                validate_crazyflie_room_profile(profile, project_root=self.project_root)
            self.status_text.set("Validation passed.")
            messagebox.showinfo("Validation", "Profile validation passed.")
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))

    def _on_run(self) -> None:
        try:
            profile = self._collect_profile_from_gui()
            if self.current_spec.scenario_id == "crazyflie_room_basic":
                validate_crazyflie_room_profile(profile, project_root=self.project_root)
            self.current_profile = profile
            self.result = {
                "run_requested": True,
                "scenario_id": self.current_spec.scenario_id,
                "profile": profile,
                "profile_path": Path(self.profile_path.get()).resolve(),
            }
            self.root.destroy()
        except Exception as exc:
            messagebox.showerror("Cannot run", str(exc))

    def _on_cancel(self) -> None:
        self.result = None
        self.root.destroy()

    def _add_float_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.DoubleVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="w", padx=8, pady=4)

    def _add_int_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.IntVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=18).grid(row=row, column=1, sticky="w", padx=8, pady=4)

    def _add_color_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="w", padx=8, pady=4)
        entry = tk.Entry(holder, textvariable=variable, width=12, relief="solid", borderwidth=1, justify="center")
        entry.pack(side="left")
        self._color_entries[id(variable)] = entry
        self._sync_color_entry(variable)
        variable.trace_add("write", lambda *_args, v=variable: self._sync_color_entry(v))
        ttk.Button(holder, text="Pick", command=lambda: self._pick_color(variable)).pack(side="left", padx=(8, 0))

    def _sync_color_entry(self, variable: tk.StringVar) -> None:
        entry = self._color_entries.get(id(variable))
        if entry is None:
            return
        value = normalize_hex_color(variable.get())
        if value is None:
            entry.configure(background="#ffffff", foreground="#000000", insertbackground="#000000")
            return
        r, g, b = hex_to_rgb(value)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        fg = "#000000" if luminance > 0.55 else "#ffffff"
        entry.configure(background=value, foreground=fg, insertbackground=fg)
        self._redraw_all_maps()

    def _pick_color(self, variable: tk.StringVar) -> None:
        initial = variable.get() or "#ffffff"
        _rgb, hex_color = colorchooser.askcolor(color=initial)
        if hex_color:
            variable.set(hex_color.lower())


def run_scenario_launcher_gui(project_root: Path) -> dict[str, Any] | None:
    gui = ScenarioLauncherGui(project_root=project_root)
    return gui.run()


def rgb_to_hex(rgb: list[float] | tuple[float, float, float]) -> str:
    r = clamp_int(round(float(rgb[0]) * 255.0), 0, 255)
    g = clamp_int(round(float(rgb[1]) * 255.0), 0, 255)
    b = clamp_int(round(float(rgb[2]) * 255.0), 0, 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def normalize_hex_color(value: str) -> str | None:
    text = value.strip().lower()
    if not text:
        return None
    if not text.startswith("#"):
        text = "#" + text
    if len(text) != 7:
        return None
    try:
        int(text[1:], 16)
    except ValueError:
        return None
    return text


def hex_to_rgb(value: str) -> list[float]:
    text = value.strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if len(text) != 6:
        raise ValueError(f"Invalid color '{value}'. Expected #rrggbb.")
    try:
        r = int(text[0:2], 16) / 255.0
        g = int(text[2:4], 16) / 255.0
        b = int(text[4:6], 16) / 255.0
    except ValueError as exc:
        raise ValueError(f"Invalid color '{value}'. Expected #rrggbb.") from exc
    return [r, g, b]


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def fmt_vec(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    return "[" + ", ".join(f"{float(v):.2f}" for v in value) + "]"
