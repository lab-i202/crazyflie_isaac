# gui/scenario_launcher_gui.py
#
# Scenario launcher GUI.
#
# This GUI does not import Isaac Sim. It is safe to run with normal Python.
# The launcher returns a scenario_id + JSON profile. The caller starts Isaac Sim afterwards.

from __future__ import annotations

import copy
import json
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
            obstacles_tab = ScrollableFrame(notebook)
            landmark_tab = ScrollableFrame(notebook)
            crazyflie_tab = ScrollableFrame(notebook)
            cameras_tab = ScrollableFrame(notebook)
            runtime_tab = ScrollableFrame(notebook)

            notebook.add(scenario_tab, text="Scenario")
            notebook.add(room_tab, text="Room")
            notebook.add(obstacles_tab, text="Obstacles")
            notebook.add(landmark_tab, text="Landmark")
            notebook.add(crazyflie_tab, text="Crazyflie")
            notebook.add(cameras_tab, text="Cameras")
            notebook.add(runtime_tab, text="Runtime")

            self._build_scenario_tab(scenario_tab.content)
            self._build_room_tab(room_tab.content)
            self._build_obstacles_tab(obstacles_tab.content)
            self._build_landmark_tab(landmark_tab.content)
            self._build_crazyflie_tab(crazyflie_tab.content)
            self._build_cameras_tab(cameras_tab.content)
            self._build_runtime_tab(runtime_tab.content)
            self._refresh_obstacle_tree()
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

    def _build_obstacles_tab(self, parent: ttk.Frame) -> None:
        parent.configure(padding=12)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        left = ttk.Frame(parent)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.LabelFrame(parent, text="Selected obstacle", padding=8)
        right.grid(row=0, column=1, sticky="nsew")

        columns = ("enabled", "kind", "position", "size", "color", "collision")
        self.obstacle_tree = ttk.Treeview(left, columns=columns, show="tree headings", height=12)
        self.obstacle_tree.heading("#0", text="Name")
        self.obstacle_tree.heading("enabled", text="On")
        self.obstacle_tree.heading("kind", text="Kind")
        self.obstacle_tree.heading("position", text="Position")
        self.obstacle_tree.heading("size", text="Size")
        self.obstacle_tree.heading("color", text="Color")
        self.obstacle_tree.heading("collision", text="Collision")
        self.obstacle_tree.column("#0", width=120)
        self.obstacle_tree.column("enabled", width=42, anchor="center")
        self.obstacle_tree.column("kind", width=80)
        self.obstacle_tree.column("position", width=130)
        self.obstacle_tree.column("size", width=130)
        self.obstacle_tree.column("color", width=80)
        self.obstacle_tree.column("collision", width=70, anchor="center")
        self.obstacle_tree.pack(fill="both", expand=True)
        self.obstacle_tree.bind("<<TreeviewSelect>>", self._on_obstacle_selected)

        obstacle_buttons = ttk.Frame(left)
        obstacle_buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(obstacle_buttons, text="Add box", command=self._on_add_obstacle).pack(side="left")
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
        ttk.Button(right, text="Apply obstacle edits", command=self._on_apply_obstacle).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 0))
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
        ttk.Checkbutton(parent, text="Capture RGB - disabled by runner in Step 1", variable=self.onboard_capture_rgb).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10); row += 1
        ttk.Label(parent, text="Isometric camera", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)); row += 1
        ttk.Checkbutton(parent, text="Enable isometric camera prim", variable=self.iso_enabled).grid(row=row, column=0, columnspan=2, sticky="w", pady=4); row += 1
        self._add_float_row(parent, row, "Eye X [m]", self.iso_x); row += 1
        self._add_float_row(parent, row, "Eye Y [m]", self.iso_y); row += 1
        self._add_float_row(parent, row, "Eye Z [m]", self.iso_z); row += 1
        self._add_float_row(parent, row, "Look-at X [m]", self.iso_look_x); row += 1
        self._add_float_row(parent, row, "Look-at Y [m]", self.iso_look_y); row += 1
        self._add_float_row(parent, row, "Look-at Z [m]", self.iso_look_z); row += 1
        self._add_float_row(parent, row, "Isometric focal length [mm]", self.iso_focal); row += 1
        self._add_int_row(parent, row, "Isometric width", self.iso_width); row += 1
        self._add_int_row(parent, row, "Isometric height", self.iso_height); row += 1
        ttk.Checkbutton(parent, text="Capture RGB - disabled by runner in Step 1", variable=self.iso_capture_rgb).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

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
            text="RGB camera capture remains disabled in this Step 1 runner to avoid reintroducing the previous Hydra/render-texture shutdown instability.",
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

    def _collect_profile_from_gui(self) -> dict[str, Any]:
        if self.current_spec.scenario_id != "crazyflie_room_basic":
            if self.current_profile is None:
                raise ValueError("No profile loaded.")
            return copy.deepcopy(self.current_profile)

        profile = copy.deepcopy(self.current_profile or default_crazyflie_room_profile())
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
            }
        )

        iso = profile["cameras"]["isometric"]
        iso.update(
            {
                "enabled": self.iso_enabled.get(),
                "position_m": [self.iso_x.get(), self.iso_y.get(), self.iso_z.get()],
                "look_at_m": [self.iso_look_x.get(), self.iso_look_y.get(), self.iso_look_z.get()],
                "focal_length_mm": self.iso_focal.get(),
                "resolution": [self.iso_width.get(), self.iso_height.get()],
                "capture_rgb": self.iso_capture_rgb.get(),
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
                    "yes" if obstacle.get("collision", True) else "no",
                ),
            )

    def _on_obstacle_selected(self, _event: tk.Event | None = None) -> None:
        selection = self.obstacle_tree.selection()
        if not selection or self.current_profile is None:
            self.selected_obstacle_index = None
            return
        index = int(selection[0])
        self.selected_obstacle_index = index
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
        self.obs_collision.set(obstacle["collision"])

    def _on_add_obstacle(self) -> None:
        if self.current_profile is None:
            return
        count = len(self.current_profile.setdefault("obstacles", [])) + 1
        self.current_profile["obstacles"].append(
            {
                "name": f"box_{count:02d}",
                "enabled": True,
                "kind": "box",
                "position_m": [0.0, 0.0, 0.25],
                "size_m": [0.3, 0.3, 0.5],
                "color": [0.8, 0.25, 0.15],
                "collision": True,
            }
        )
        self._refresh_obstacle_tree()

    def _on_remove_obstacle(self) -> None:
        if self.current_profile is None or self.selected_obstacle_index is None:
            messagebox.showinfo("No obstacle selected", "Select an obstacle first.")
            return
        del self.current_profile["obstacles"][self.selected_obstacle_index]
        self.selected_obstacle_index = None
        self._refresh_obstacle_tree()

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
        }
        try:
            profile = self._collect_profile_from_gui()
            validate_crazyflie_room_profile(profile, project_root=self.project_root)
            self.current_profile = profile
        except Exception as exc:
            messagebox.showerror("Invalid obstacle", str(exc))
            return
        self._refresh_obstacle_tree()
        self.status_text.set("Obstacle edits applied.")

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
        preview = tk.Label(holder, textvariable=variable, width=12, relief="solid", borderwidth=1)
        preview.pack(side="left")
        ttk.Button(holder, text="Pick", command=lambda: self._pick_color(variable)).pack(side="left", padx=(8, 0))

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
