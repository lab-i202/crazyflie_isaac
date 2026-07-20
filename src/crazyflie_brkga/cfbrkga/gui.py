from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, save_config, validate_config


class TrainingGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Crazyflie Isaac BRKGA")
        self.root.geometry("1180x920")
        self.config_path = DEFAULT_CONFIG_PATH
        self.cfg = load_config(self.config_path)
        self.vars: dict[str, tk.Variable] = {}
        self.status = tk.StringVar(value="Ready")
        self._build()
        self._load_to_form()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Configuration:").pack(side="left")
        self.path_var = tk.StringVar(value=str(self.config_path))
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Open", command=self.open_config).pack(side="left", padx=2)
        ttk.Button(top, text="Save as", command=self.save_as).pack(side="left", padx=2)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)
        self.tabs = {name: ttk.Frame(self.notebook, padding=10) for name in [
            "Project", "Environment", "Camera & Vision", "BRKGA", "Mission", "Data"
        ]}
        for name, frame in self.tabs.items():
            self.notebook.add(frame, text=name)

        self._project_tab()
        self._environment_tab()
        self._camera_tab()
        self._brkga_tab()
        self._mission_tab()
        self._data_tab()

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Validate", command=self.validate).pack(side="left", padx=3)
        ttk.Button(bottom, text="Run mock integrity test", command=self.run_mock).pack(side="left", padx=3)
        ttk.Button(bottom, text="Start Isaac training", command=self.run_isaac).pack(side="left", padx=3)
        ttk.Button(bottom, text="Open dashboard", command=self.open_dashboard).pack(side="left", padx=3)
        ttk.Button(bottom, text="Open outputs", command=self.open_outputs).pack(side="left", padx=3)
        ttk.Label(bottom, textvariable=self.status).pack(side="right")

    def _field(self, parent, row, label, key, kind="str", width=18, values=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        var_cls = {"int": tk.IntVar, "float": tk.DoubleVar, "bool": tk.BooleanVar}.get(kind, tk.StringVar)
        var = var_cls()
        self.vars[key] = var
        if kind == "bool":
            widget = ttk.Checkbutton(parent, variable=var)
        elif values:
            widget = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        else:
            widget = ttk.Entry(parent, textvariable=var, width=width)
        widget.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        parent.columnconfigure(1, weight=1)
        return widget

    def _project_tab(self):
        tab = self.tabs["Project"]
        self._field(tab, 0, "Project name", "project.name")
        self._field(tab, 1, "Note", "project.note")
        self._field(tab, 2, "Headless Isaac", "app.headless", "bool")
        self._field(tab, 3, "Window width", "app.width", "int")
        self._field(tab, 4, "Window height", "app.height", "int")
        self._field(tab, 5, "Renderer", "app.renderer", values=["RayTracedLighting", "RealTimePathTracing"])
        self._field(tab, 6, "Parallel environments", "environment.num_parallel_envs", "int")

    def _environment_tab(self):
        tab = self.tabs["Environment"]
        self._field(tab, 0, "Room size X, Y, Z (m)", "environment.room.size_m")
        self._field(tab, 1, "Environment spacing (m)", "environment.env_spacing_m", "float")
        self._field(tab, 2, "Crazyflie initial X, Y, Z", "environment.crazyflie.initial_pose.position_m")
        self._field(tab, 3, "Crazyflie initial yaw (deg)", "environment.crazyflie.initial_pose.yaw_deg", "float")
        self._field(tab, 4, "Crazyflie USD path (blank = Isaac assets)", "environment.crazyflie.usd_path")
        self._field(tab, 5, "Landmark X, Y, Z", "environment.landmark.position_m")
        self._field(tab, 6, "Landmark RGB 0-255", "environment.landmark.rgb_255")
        self._field(tab, 7, "Landmark radius (m)", "environment.landmark.radius_m", "float")
        self._field(tab, 8, "Landmark light intensity", "environment.landmark.light_intensity", "float")

        obstacle_frame = ttk.LabelFrame(tab, text="Obstacles JSON (box/wall)", padding=6)
        obstacle_frame.grid(row=9, column=0, columnspan=2, sticky="nsew", padx=4, pady=6)
        self.obstacles_text = tk.Text(obstacle_frame, height=7, wrap="none")
        self.obstacles_text.pack(fill="both", expand=True)

        asset_frame = ttk.LabelFrame(tab, text="Custom USD assets", padding=6)
        asset_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", padx=4, pady=6)
        asset_bar = ttk.Frame(asset_frame)
        asset_bar.pack(fill="x", pady=(0, 4))
        self.asset_scan_var = tk.StringVar()
        self.asset_combo = ttk.Combobox(asset_bar, textvariable=self.asset_scan_var, state="readonly")
        self.asset_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(asset_bar, text="Rescan assets/", command=self._scan_assets).pack(side="left", padx=3)
        ttk.Button(asset_bar, text="Add selected", command=self._add_scanned_asset).pack(side="left", padx=3)
        self.custom_assets_text = tk.Text(asset_frame, height=8, wrap="none")
        self.custom_assets_text.pack(fill="both", expand=True)
        ttk.Label(
            asset_frame,
            text="Optional collision_aabb_size_m lets the baseline range/collision model include imported geometry.",
        ).pack(anchor="w", pady=(4, 0))
        tab.rowconfigure(9, weight=1)
        tab.rowconfigure(10, weight=1)
        self._scan_assets()

    def _camera_tab(self):
        tab = self.tabs["Camera & Vision"]
        self._field(tab, 0, "Camera width", "camera.width", "int")
        self._field(tab, 1, "Camera height", "camera.height", "int")
        self._field(tab, 2, "Horizontal FOV (deg, mock projection)", "camera.horizontal_fov_deg", "float")
        self._field(tab, 3, "Focal length (mm)", "camera.focal_length_mm", "float")
        self._field(tab, 4, "Camera local X, Y, Z", "camera.local_position_m")
        self._field(tab, 5, "HSV lower", "vision.hsv_lower")
        self._field(tab, 6, "HSV upper", "vision.hsv_upper")
        self._field(tab, 7, "Minimum blob area (px)", "vision.min_area_px", "float")
        self._field(tab, 8, "Minimum circularity", "vision.min_circularity", "float")
        self._field(tab, 9, "Morphology kernel", "vision.morphology_kernel", "int")

    def _brkga_tab(self):
        tab = self.tabs["BRKGA"]
        self._field(tab, 0, "Population size", "brkga.population_size", "int")
        self._field(tab, 1, "Generations", "brkga.generations", "int")
        self._field(tab, 2, "Elite fraction", "brkga.elite_fraction", "float")
        self._field(tab, 3, "Mutant fraction", "brkga.mutant_fraction", "float")
        self._field(tab, 4, "Elite inheritance probability", "brkga.elite_inheritance_probability", "float")
        self._field(tab, 5, "Random seed", "brkga.random_seed", "int")
        self._field(tab, 6, "Episode seeds", "brkga.episode_seeds")
        self._field(tab, 7, "Weight lower bound", "policy.decoder.lower_bound", "float")
        self._field(tab, 8, "Weight upper bound", "policy.decoder.upper_bound", "float")
        self._field(tab, 9, "Torch device", "policy.device", values=["cuda", "cuda:0", "cpu"])
        ttk.Label(tab, text="Fixed baseline network: 2 → 128 → 128 → 5 (17,541 evolved values).").grid(row=10, column=0, columnspan=2, sticky="w", pady=10)

    def _mission_tab(self):
        tab = self.tabs["Mission"]
        self._field(tab, 0, "Control step (s)", "mission.control_step_s", "float")
        self._field(tab, 1, "Maximum steps", "mission.max_steps", "int")
        self._field(tab, 2, "Forward speed (m/s)", "mission.actions.forward_speed_m_s", "float")
        self._field(tab, 3, "Vertical speed (m/s)", "mission.actions.vertical_speed_m_s", "float")
        self._field(tab, 4, "Yaw rate (rad/s)", "mission.actions.yaw_rate_rad_s", "float")
        self._field(tab, 5, "Success front ToF (m)", "mission.success_front_tof_m", "float")
        self._field(tab, 6, "Lost landmark max steps", "mission.lost_landmark_max_steps", "int")
        self._field(tab, 7, "Range sensor backend", "environment.sensors.range_backend", values=["physx_raycast", "analytic"])
        self._field(tab, 8, "Horizontal collision threshold", "environment.sensors.horizontal_collision_threshold_m", "float")
        self._field(tab, 9, "Success reward", "mission.reward.success_reward", "float")
        self._field(tab, 10, "Collision penalty", "mission.reward.collision_penalty", "float")

    def _data_tab(self):
        tab = self.tabs["Data"]
        self._field(tab, 0, "Output root", "data.output_root")
        self._field(tab, 1, "Execution mode", "data.execution_mode", values=["training", "integrity"])
        self._field(tab, 2, "Save raw integrity data", "data.integrity.save_raw_data", "bool")
        self._field(tab, 3, "Save integrity frames", "data.integrity.save_frames", "bool")
        self._field(tab, 4, "Frame stride", "data.integrity.frame_stride", "int")
        self._field(tab, 5, "Store integrity steps in SQLite", "data.integrity.store_steps_in_database", "bool")
        self._field(tab, 6, "Database heartbeat interval (steps)", "data.heartbeat_steps", "int")
        ttk.Label(tab, text="SQLite is authoritative. CSV files are regenerated after every generation.").grid(row=7, column=0, columnspan=2, sticky="w", pady=10)

    def _load_to_form(self):
        for key, var in self.vars.items():
            value = self._get(key)
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            var.set(value)
        if hasattr(self, "obstacles_text"):
            self.obstacles_text.delete("1.0", "end")
            self.obstacles_text.insert("1.0", json.dumps(self.cfg["environment"].get("obstacles", []), indent=2))
        if hasattr(self, "custom_assets_text"):
            self.custom_assets_text.delete("1.0", "end")
            self.custom_assets_text.insert("1.0", json.dumps(self.cfg["environment"].get("custom_assets", []), indent=2))

    def _collect(self):
        for key, var in self.vars.items():
            value = var.get()
            original = self._get(key)
            if isinstance(original, list):
                parts = [part.strip() for part in str(value).split(",") if part.strip()]
                if original and isinstance(original[0], int):
                    value = [int(part) for part in parts]
                else:
                    value = [float(part) for part in parts]
            self._set(key, value)
        if hasattr(self, "obstacles_text"):
            self.cfg["environment"]["obstacles"] = json.loads(self.obstacles_text.get("1.0", "end").strip() or "[]")
        if hasattr(self, "custom_assets_text"):
            self.cfg["environment"]["custom_assets"] = json.loads(self.custom_assets_text.get("1.0", "end").strip() or "[]")
        validate_config(self.cfg)

    def _scan_assets(self):
        extensions = {".usd", ".usda", ".usdc"}
        assets_root = PROJECT_ROOT / "assets"
        assets_root.mkdir(parents=True, exist_ok=True)
        values = [
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            for path in sorted(assets_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in extensions
        ]
        if hasattr(self, "asset_combo"):
            self.asset_combo["values"] = values
            self.asset_scan_var.set(values[0] if values else "")

    def _add_scanned_asset(self):
        selected = self.asset_scan_var.get().strip()
        if not selected:
            messagebox.showwarning("No USD asset", "Place a .usd, .usda or .usdc file under assets/ and rescan.")
            return
        try:
            items = json.loads(self.custom_assets_text.get("1.0", "end").strip() or "[]")
            if not isinstance(items, list):
                raise ValueError("Custom assets JSON must be a list.")
            stem = Path(selected).stem
            items.append({
                "name": stem,
                "enabled": True,
                "usd_path": selected,
                "position_m": [0.0, 0.0, 0.0],
                "rotation_deg": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
                "collision_aabb_size_m": None,
            })
            self.custom_assets_text.delete("1.0", "end")
            self.custom_assets_text.insert("1.0", json.dumps(items, indent=2))
        except Exception as exc:
            messagebox.showerror("Cannot add asset", str(exc))

    def _get(self, dotted):
        value = self.cfg
        for part in dotted.split("."):
            value = value[part]
        return value

    def _set(self, dotted, new_value):
        target = self.cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = new_value

    def open_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.config_path = Path(path)
            self.cfg = load_config(path)
            self.path_var.set(path)
            self._load_to_form()
            self.status.set("Configuration loaded")

    def save_as(self):
        try:
            self._collect()
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                self.config_path = save_config(self.cfg, path)
                self.path_var.set(str(self.config_path))
                self.status.set("Configuration saved")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _save_current(self):
        self._collect()
        self.config_path = save_config(self.cfg, self.path_var.get())
        return self.config_path

    def validate(self):
        try:
            self._save_current()
            pop = int(self.cfg["brkga"]["population_size"])
            genes = 17541
            estimated_gib = pop * int(self.cfg["brkga"]["generations"]) * genes * 8 / 1024**3
            messagebox.showinfo("Valid", f"Configuration is valid.\nApproximate uncompressed chromosome+weight data: {estimated_gib:.2f} GiB")
            self.status.set("Valid")
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))

    def _launch(self, script_name):
        path = self._save_current()
        env = os.environ.copy()
        env["CRAZYFLIE_BRKGA_CONFIG"] = str(path)
        process = subprocess.Popen([sys.executable, str(PROJECT_ROOT / script_name)], cwd=PROJECT_ROOT, env=env)
        self.status.set(f"Started PID {process.pid}")

    def run_mock(self):
        try:
            self._collect()
            smoke_cfg = json.loads(json.dumps(self.cfg))
            smoke_cfg["project"]["name"] = f"{self.cfg['project']['name']}_smoke"
            smoke_cfg["project"]["note"] = "GUI-generated software-only integrity smoke test."
            smoke_cfg["environment"]["num_parallel_envs"] = min(2, int(self.cfg["environment"]["num_parallel_envs"]))
            smoke_cfg["brkga"]["population_size"] = 4
            smoke_cfg["brkga"]["generations"] = 1
            smoke_cfg["brkga"]["episode_seeds"] = [int(self.cfg["brkga"]["episode_seeds"][0])]
            smoke_cfg["mission"]["max_steps"] = min(20, int(self.cfg["mission"]["max_steps"]))
            smoke_cfg["mission"]["lost_landmark_max_steps"] = smoke_cfg["mission"]["max_steps"]
            smoke_cfg["data"]["execution_mode"] = "integrity"
            smoke_cfg["data"]["integrity"]["save_raw_data"] = True
            smoke_cfg["data"]["integrity"]["save_frames"] = True
            smoke_cfg["data"]["integrity"]["frame_stride"] = 5
            smoke_cfg["data"]["integrity"]["store_steps_in_database"] = True
            smoke_path = PROJECT_ROOT / "config" / "_gui_smoke.json"
            save_config(smoke_cfg, smoke_path)
            env = os.environ.copy()
            env["CRAZYFLIE_BRKGA_CONFIG"] = str(smoke_path)
            process = subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "train_mock.py")],
                cwd=PROJECT_ROOT,
                env=env,
            )
            self.status.set(f"Started mock integrity PID {process.pid}")
        except Exception as exc:
            messagebox.showerror("Mock launch failed", str(exc))

    def run_isaac(self):
        try:
            path = self._save_current()
            env = os.environ.copy()
            env["CRAZYFLIE_BRKGA_CONFIG"] = str(path)
            bat = PROJECT_ROOT / "run_training.bat"
            if os.name == "nt" and bat.exists():
                process = subprocess.Popen(["cmd", "/c", str(bat)], cwd=PROJECT_ROOT, env=env)
            else:
                process = subprocess.Popen([sys.executable, str(PROJECT_ROOT / "train_isaac.py")], cwd=PROJECT_ROOT, env=env)
            self.status.set(f"Started Isaac PID {process.pid}")
        except Exception as exc:
            messagebox.showerror("Isaac launch failed", str(exc))

    def open_dashboard(self):
        try:
            output_root = (PROJECT_ROOT / self.cfg["data"]["output_root"]).resolve()
            databases = sorted(output_root.glob("*/run.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
            env = os.environ.copy()
            if databases:
                env["CRAZYFLIE_BRKGA_DB"] = str(databases[0])
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "dashboard" / "app.py")], cwd=PROJECT_ROOT, env=env)
        except Exception as exc:
            messagebox.showerror("Dashboard failed", "Install dashboard requirements first.\n\n" + str(exc))

    def open_outputs(self):
        path = (PROJECT_ROOT / self.cfg["data"]["output_root"]).resolve()
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])


def run_gui() -> None:
    root = tk.Tk()
    TrainingGui(root)
    root.mainloop()
