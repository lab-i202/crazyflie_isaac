from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, config_summary, load_config, save_config, validate_config
from .process_manager import ProcessManager
from .runtime_files import SCENARIO_COMMAND, write_command
from .scenario_builder import ScenarioBuilderPanel


class TrainingGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Crazyflie Isaac BRKGA")
        self.root.geometry("1480x940")
        self.root.minsize(1120, 760)
        self.config_path = DEFAULT_CONFIG_PATH.resolve()
        self.cfg = load_config(self.config_path)
        self.vars: dict[str, tk.Variable] = {}
        self.status = tk.StringVar(value="Ready")
        self.path_var = tk.StringVar(value=str(self.config_path))
        self.dirty = False
        self.process_manager = ProcessManager(PROJECT_ROOT, self._append_log, self.status.set)
        self._build()
        self._load_to_form()
        self.root.after(100, self._poll_processes)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------ UI build ------------------------
    def _build(self) -> None:
        self._configure_styles()
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Current configuration:").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Open", command=self.open_config).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self.save_current).pack(side="left", padx=2)
        ttk.Button(top, text="Save As", command=self.save_as).pack(side="left", padx=2)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=4)

        scenario_tab = ttk.Frame(self.notebook)
        training_tab = ttk.Frame(self.notebook, padding=10)
        data_tab = ttk.Frame(self.notebook, padding=10)
        tools_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(scenario_tab, text="Scenario builder")
        self.notebook.add(training_tab, text="Training configuration")
        self.notebook.add(data_tab, text="Data and outputs")
        self.notebook.add(tools_tab, text="Tools and logs")

        self.scenario_builder = ScenarioBuilderPanel(scenario_tab, PROJECT_ROOT, on_changed=self._mark_dirty)
        self.scenario_builder.pack(fill="both", expand=True)
        self._build_training_tab(training_tab)
        self._build_data_tab(data_tab)
        self._build_tools_tab(tools_tab)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Save current JSON", command=self.save_current).pack(side="left", padx=3)
        ttk.Button(bottom, text="Validate", command=self.validate).pack(side="left", padx=3)
        ttk.Button(bottom, text="Update scenario", command=self.update_scenario, style="Accent.TButton").pack(side="left", padx=3)
        ttk.Button(bottom, text="RUN TRAINING", command=self.run_training, style="Run.TButton").pack(side="left", padx=8)
        ttk.Button(bottom, text="Landmark detector", command=self.open_detector).pack(side="left", padx=3)
        ttk.Button(bottom, text="Open dashboard", command=self.open_dashboard).pack(side="left", padx=3)
        ttk.Label(bottom, textvariable=self.status).pack(side="right")

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 6))
            style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))
        except tk.TclError:
            pass

    def _build_training_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        project = ttk.LabelFrame(tab, text="Project and Isaac", padding=8)
        brkga = ttk.LabelFrame(tab, text="BRKGA and policy", padding=8)
        mission = ttk.LabelFrame(tab, text="Mission and reward", padding=8)
        project.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        brkga.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        mission.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(6, 0))

        row = 0
        row = self._field(project, row, "Project name", "project.name")
        row = self._field(project, row, "Description / note", "project.note")
        row = self._field(project, row, "Headless Isaac", "app.headless", "bool")
        row = self._field(project, row, "Window width", "app.width", "int")
        row = self._field(project, row, "Window height", "app.height", "int")
        row = self._field(project, row, "Renderer", "app.renderer", values=("RayTracedLighting", "RealTimePathTracing"))
        row = self._field(project, row, "Parallel environments", "environment.num_parallel_envs", "int")

        row = 0
        row = self._field(brkga, row, "Population size", "brkga.population_size", "int")
        row = self._field(brkga, row, "Generations", "brkga.generations", "int")
        row = self._field(brkga, row, "Elite fraction", "brkga.elite_fraction", "float")
        row = self._field(brkga, row, "Mutant fraction", "brkga.mutant_fraction", "float")
        row = self._field(brkga, row, "Elite inheritance probability", "brkga.elite_inheritance_probability", "float")
        row = self._field(brkga, row, "Random seed", "brkga.random_seed", "int")
        row = self._field(brkga, row, "Episode seeds", "brkga.episode_seeds")
        row = self._field(brkga, row, "Decoder lower bound", "policy.decoder.lower_bound", "float")
        row = self._field(brkga, row, "Decoder upper bound", "policy.decoder.upper_bound", "float")
        row = self._field(brkga, row, "Torch device", "policy.device", values=("cuda", "cuda:0", "cpu"))
        ttk.Label(brkga, text="Fixed baseline network: 2 → 128 → 128 → 5; BRKGA evolves 17,541 values.", foreground="gray", wraplength=520).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

        row = 0
        mission_fields = (
            ("Control step [s]", "mission.control_step_s", "float"),
            ("Maximum control steps", "mission.max_steps", "int"),
            ("Forward speed [m/s]", "mission.actions.forward_speed_m_s", "float"),
            ("Vertical speed [m/s]", "mission.actions.vertical_speed_m_s", "float"),
            ("Yaw rate [rad/s]", "mission.actions.yaw_rate_rad_s", "float"),
            ("Success front ToF [m]", "mission.success_front_tof_m", "float"),
            ("Lost landmark maximum steps", "mission.lost_landmark_max_steps", "int"),
            ("Step penalty scale", "mission.reward.step_penalty_scale", "float"),
            ("Time penalty", "mission.reward.time_penalty", "float"),
            ("Collision penalty", "mission.reward.collision_penalty", "float"),
            ("Lost landmark penalty", "mission.reward.lost_landmark_penalty", "float"),
            ("Timeout penalty", "mission.reward.timeout_penalty", "float"),
            ("Failure terminal penalty", "mission.reward.failure_terminal_penalty", "float"),
            ("Success reward", "mission.reward.success_reward", "float"),
            ("Near-target partial reward", "mission.reward.near_target_partial_reward", "float"),
            ("Worker error fitness", "mission.reward.worker_error_fitness", "float"),
        )
        for index, (label, key, kind) in enumerate(mission_fields):
            column = 0 if index < 8 else 2
            local_row = index if index < 8 else index - 8
            self._field_at(mission, local_row, column, label, key, kind)
        mission.columnconfigure(1, weight=1)
        mission.columnconfigure(3, weight=1)

    def _build_data_tab(self, tab: ttk.Frame) -> None:
        tab.columnconfigure(0, weight=1)
        general = ttk.LabelFrame(tab, text="Experiment output", padding=8)
        integrity = ttk.LabelFrame(tab, text="Integrity / debugging retention", padding=8)
        mirror = ttk.LabelFrame(tab, text="Stable latest-run mirror", padding=8)
        general.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        integrity.grid(row=1, column=0, sticky="ew", pady=8)
        mirror.grid(row=2, column=0, sticky="ew", pady=8)
        row = 0
        row = self._field(general, row, "Output root", "data.output_root")
        row = self._field(general, row, "Execution mode", "data.execution_mode", values=("training", "integrity"))
        row = self._field(general, row, "Database heartbeat interval [steps]", "data.heartbeat_steps", "int")
        row = 0
        row = self._field(integrity, row, "Save raw data", "data.integrity.save_raw_data", "bool")
        row = self._field(integrity, row, "Save camera frames", "data.integrity.save_frames", "bool")
        row = self._field(integrity, row, "Frame stride", "data.integrity.frame_stride", "int")
        row = self._field(integrity, row, "Store every step in SQLite", "data.integrity.store_steps_in_database", "bool")
        ttk.Label(integrity, text="Normal training should leave raw data and frame storage disabled.", foreground="gray").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row = 0
        row = self._field(mirror, row, "Enabled", "data.latest_mirror.enabled", "bool")
        row = self._field(mirror, row, "Folder name (blank = last_<project>)", "data.latest_mirror.folder_name")
        row = self._field(mirror, row, "Database synchronization interval [s]", "data.latest_mirror.database_sync_interval_s", "float")
        row = self._field(mirror, row, "Copy CSV exports", "data.latest_mirror.copy_csv", "bool")
        row = self._field(mirror, row, "Copy checkpoints", "data.latest_mirror.copy_checkpoints", "bool")
        ttk.Label(mirror, text="Keep DBeaver connected to outputs/experiments/last_<project>/run.sqlite.", foreground="gray").grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_tools_tab(self, tab: ttk.Frame) -> None:
        actions = ttk.LabelFrame(tab, text="Project scripts", padding=8)
        actions.pack(fill="x", pady=(0, 8))
        buttons = (
            ("Run mock integrity", self.run_mock_integrity),
            ("Run Isaac integrity", self.run_isaac_integrity),
            ("Start / update scenario preview", self.update_scenario),
            ("Stop scenario preview", self.stop_scenario),
            ("Landmark detector configurator", self.open_detector),
            ("Open dashboard", self.open_dashboard),
            ("Open output folder", self.open_outputs),
            ("Open latest database folder", self.open_latest_database_folder),
        )
        for index, (label, command) in enumerate(buttons):
            ttk.Button(actions, text=label, command=command).grid(row=index // 4, column=index % 4, sticky="ew", padx=4, pady=4)
        for column in range(4):
            actions.columnconfigure(column, weight=1)

        log_frame = ttk.LabelFrame(tab, text="Process output", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_frame, wrap="none", height=28, state="disabled", background="#101010", foreground="#e6e6e6", insertbackground="white")
        y_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        x_scroll = ttk.Scrollbar(log_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        ttk.Button(log_frame, text="Clear log", command=self._clear_log).grid(row=2, column=0, sticky="w", pady=(6, 0))

    def _field(self, parent: ttk.Frame, row: int, label: str, key: str, kind: str = "str", values: tuple[str, ...] | None = None) -> int:
        self._field_at(parent, row, 0, label, key, kind, values)
        parent.columnconfigure(1, weight=1)
        return row + 1

    def _field_at(self, parent: ttk.Frame, row: int, column: int, label: str, key: str, kind: str = "str", values: tuple[str, ...] | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=4, pady=4)
        cls = {"int": tk.IntVar, "float": tk.DoubleVar, "bool": tk.BooleanVar}.get(kind, tk.StringVar)
        variable = cls()
        self.vars[key] = variable
        variable.trace_add("write", lambda *_args: self._mark_dirty())
        if kind == "bool":
            widget = ttk.Checkbutton(parent, variable=variable)
        elif values:
            widget = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        else:
            widget = ttk.Entry(parent, textvariable=variable)
        widget.grid(row=row, column=column + 1, sticky="ew", padx=4, pady=4)

    # ------------------------ config transfer ------------------------
    def _get(self, dotted: str) -> Any:
        value: Any = self.cfg
        for part in dotted.split("."):
            value = value[part]
        return value

    def _set(self, dotted: str, value: Any) -> None:
        target = self.cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    def _load_to_form(self) -> None:
        self.scenario_builder.load_from_config(self.cfg)
        for key, variable in self.vars.items():
            value = self._get(key)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            variable.set(value)
        self.dirty = False
        self._update_title()
        self.status.set(config_summary(self.cfg))

    def _collect(self) -> None:
        self.scenario_builder.apply_to_config(self.cfg)
        for key, variable in self.vars.items():
            value = variable.get()
            original = self._get(key)
            if isinstance(original, list):
                parts = [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]
                if original and isinstance(original[0], int):
                    value = [int(part) for part in parts]
                else:
                    value = [float(part) for part in parts]
            self._set(key, value)
        validate_config(self.cfg)

    def _mark_dirty(self) -> None:
        self.dirty = True
        self._update_title()

    def _update_title(self) -> None:
        marker = " *" if self.dirty else ""
        self.root.title(f"Crazyflie Isaac BRKGA - {self.config_path.name}{marker}")

    # ------------------------ file actions ------------------------
    def open_config(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(PROJECT_ROOT / "config"), filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.config_path = Path(path).resolve()
            self.cfg = load_config(self.config_path)
            self.path_var.set(str(self.config_path))
            self._load_to_form()
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc), parent=self.root)

    def save_current(self) -> Path | None:
        try:
            self._collect()
            entered = Path(self.path_var.get()).expanduser()
            if not entered.is_absolute():
                entered = PROJECT_ROOT / entered
            self.config_path = save_config(self.cfg, entered)
            self.path_var.set(str(self.config_path))
            self.dirty = False
            self._update_title()
            self.status.set(f"Saved {self.config_path}")
            return self.config_path
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return None

    def save_as(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(PROJECT_ROOT / "config"), defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.path_var.set(path)
        self.config_path = Path(path).resolve()
        self.save_current()

    def validate(self) -> None:
        path = self.save_current()
        if path is None:
            return
        pop = int(self.cfg["brkga"]["population_size"])
        generations = int(self.cfg["brkga"]["generations"])
        waves = (pop + int(self.cfg["environment"]["num_parallel_envs"]) - 1) // int(self.cfg["environment"]["num_parallel_envs"])
        messagebox.showinfo("Configuration valid", f"{config_summary(self.cfg)}\nEvaluation waves per generation: {waves}\nTotal individual evaluations: {pop * generations * len(self.cfg['brkga']['episode_seeds'])}", parent=self.root)

    # ------------------------ launch helpers ------------------------
    def _bat_command(self, filename: str) -> list[str]:
        path = PROJECT_ROOT / filename
        if os.name == "nt":
            return ["cmd", "/c", str(path)]
        raise RuntimeError(f"{filename} is a Windows launcher. Run the corresponding Python script manually on this platform.")

    def _launch(self, name: str, command: list[str], config_path: Path | None = None) -> None:
        env: dict[str, str] = {}
        if config_path is not None:
            env["CRAZYFLIE_BRKGA_CONFIG"] = str(config_path)
        self.process_manager.start(name, command, env)

    def run_training(self) -> None:
        path = self.save_current()
        if path is None:
            return
        if self.cfg["data"]["execution_mode"] != "training":
            answer = messagebox.askyesno("Execution mode", "The current JSON is in integrity mode. Change it to training and save before starting?", parent=self.root)
            if answer:
                self.cfg["data"]["execution_mode"] = "training"
                self.vars["data.execution_mode"].set("training")
                path = self.save_current()
            else:
                return
        try:
            self._launch("training", self._bat_command("run_training.bat"), path)
        except Exception as exc:
            messagebox.showerror("Training launch failed", str(exc), parent=self.root)

    def run_isaac_integrity(self) -> None:
        try:
            self._collect()
            integrity = copy.deepcopy(self.cfg)
            integrity["project"]["name"] = f"{self.cfg['project']['name']}_integrity"
            integrity["project"]["note"] = "GUI-generated Isaac integrity run using the current scenario."
            integrity["data"]["execution_mode"] = "integrity"
            integrity["data"]["integrity"]["save_raw_data"] = True
            integrity["data"]["integrity"]["save_frames"] = True
            integrity["data"]["integrity"]["store_steps_in_database"] = True
            path = save_config(integrity, PROJECT_ROOT / "config" / "_gui_isaac_integrity.json")
            self._launch("isaac_integrity", self._bat_command("run_training.bat"), path)
        except Exception as exc:
            messagebox.showerror("Isaac integrity launch failed", str(exc), parent=self.root)

    def run_mock_integrity(self) -> None:
        try:
            self._collect()
            smoke = copy.deepcopy(self.cfg)
            smoke["project"]["name"] = f"{self.cfg['project']['name']}_mock"
            smoke["environment"]["num_parallel_envs"] = min(2, int(smoke["environment"]["num_parallel_envs"]))
            smoke["brkga"]["population_size"] = 4
            smoke["brkga"]["generations"] = 1
            smoke["mission"]["max_steps"] = min(20, int(smoke["mission"]["max_steps"]))
            smoke["mission"]["lost_landmark_max_steps"] = smoke["mission"]["max_steps"]
            smoke["data"]["execution_mode"] = "integrity"
            smoke["data"]["integrity"].update({"save_raw_data": True, "save_frames": True, "frame_stride": 5, "store_steps_in_database": True})
            path = save_config(smoke, PROJECT_ROOT / "config" / "_gui_mock_integrity.json")
            self._launch("mock_integrity", [sys.executable, str(PROJECT_ROOT / "train_mock.py")], path)
        except Exception as exc:
            messagebox.showerror("Mock integrity launch failed", str(exc), parent=self.root)

    def update_scenario(self) -> None:
        path = self.save_current()
        if path is None:
            return
        try:
            revision = write_command(SCENARIO_COMMAND, "reload", path)
            if not self.process_manager.is_running("scenario_preview"):
                self._launch("scenario_preview", self._bat_command("run_scenario_preview.bat"), path)
                self.status.set(f"Scenario preview starting; revision {revision}")
            else:
                self.status.set(f"Scenario update sent; revision {revision}")
        except Exception as exc:
            messagebox.showerror("Scenario preview failed", str(exc), parent=self.root)

    def stop_scenario(self) -> None:
        write_command(SCENARIO_COMMAND, "stop", self.config_path)
        self.status.set("Scenario preview stop requested")

    def open_detector(self) -> None:
        path = self.save_current()
        if path is None:
            return
        try:
            self._launch("detector_configurator", [sys.executable, str(PROJECT_ROOT / "landmark_detector_configurator.py")], path)
        except Exception as exc:
            messagebox.showerror("Detector configurator failed", str(exc), parent=self.root)

    def open_dashboard(self) -> None:
        try:
            output_root = (PROJECT_ROOT / self.cfg["data"]["output_root"]).resolve()
            databases = sorted(output_root.glob("*/run.sqlite"), key=lambda item: (not item.parent.name.startswith("last_"), -item.stat().st_mtime))
            env: dict[str, str] = {}
            if databases:
                env["CRAZYFLIE_BRKGA_DB"] = str(databases[0])
            self.process_manager.start("dashboard", [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "dashboard" / "app.py")], env)
        except Exception as exc:
            messagebox.showerror("Dashboard failed", "Install requirements-dashboard.txt first.\n\n" + str(exc), parent=self.root)

    def open_outputs(self) -> None:
        path = (PROJECT_ROOT / self.cfg["data"]["output_root"]).resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def open_latest_database_folder(self) -> None:
        path = (PROJECT_ROOT / self.cfg["data"]["output_root"]).resolve()
        candidates = sorted(path.glob("last_*/run.sqlite"), key=lambda item: item.stat().st_mtime, reverse=True)
        self._open_path(candidates[0].parent if candidates else path)

    @staticmethod
    def _open_path(path: Path) -> None:
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", str(path)])

    # ------------------------ logs and shutdown ------------------------
    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _poll_processes(self) -> None:
        self.process_manager.drain()
        self.root.after(100, self._poll_processes)

    def _close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Unsaved configuration", "Save current changes before closing?", parent=self.root)
            if answer is None:
                return
            if answer and self.save_current() is None:
                return
        # Do not kill training or dashboard. Only request preview workers to stop.
        try:
            write_command(SCENARIO_COMMAND, "stop", self.config_path)
        except Exception:
            pass
        self.root.destroy()


def run_gui() -> None:
    root = tk.Tk()
    TrainingGui(root)
    root.mainloop()
