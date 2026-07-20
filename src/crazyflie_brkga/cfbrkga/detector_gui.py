from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, save_config, validate_config
from .image_processing import LuminousBlobDetector
from .runtime_files import DETECTOR_COMMAND, DETECTOR_FRAME, DETECTOR_STATUS, read_json, write_command


class ImagePanel(ttk.LabelFrame):
    def __init__(self, parent: tk.Widget, title: str, max_width: int = 420, max_height: int = 290) -> None:
        super().__init__(parent, text=title, padding=4)
        self.max_width = max_width
        self.max_height = max_height
        self.label = ttk.Label(self, anchor="center")
        self.label.pack(fill="both", expand=True)
        self._photo: ImageTk.PhotoImage | None = None

    def set_rgb(self, rgb) -> None:
        image = Image.fromarray(rgb)
        image.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self._photo)

    def set_gray(self, gray) -> None:
        image = Image.fromarray(gray).convert("RGB")
        image.thumbnail((self.max_width, self.max_height), Image.Resampling.NEAREST)
        self._photo = ImageTk.PhotoImage(image)
        self.label.configure(image=self._photo)


class LandmarkDetectorConfigurator:
    def __init__(self, root: tk.Tk, config_path: str | Path | None = None) -> None:
        self.root = root
        self.root.title("Crazyflie Landmark Detector Configurator")
        self.root.geometry("1540x980")
        self.root.minsize(1180, 760)
        self.config_path = Path(config_path or os.environ.get("CRAZYFLIE_BRKGA_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
        self.cfg = load_config(self.config_path)
        self.worker: subprocess.Popen | None = None
        self.offline_frame_path: Path | None = None
        self.last_frame_mtime = 0.0
        self.current_frame_path: Path | None = None
        self._refresh_job: str | None = None
        self._live_refresh_enabled = False
        self.status = tk.StringVar(value="Ready")
        self.metrics = tk.StringVar(value="No frame")
        self.path_var = tk.StringVar(value=str(self.config_path))
        self._build_vars()
        self._build()
        self._load_vision()
        self._install_live_traces()
        self._live_refresh_enabled = True
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self._tick)


    def _install_live_traces(self) -> None:
        variables = (
            self.mode,
            self.h_low, self.s_low, self.v_low,
            self.h_high, self.s_high, self.v_high,
            self.brightness, self.gray_threshold,
            self.adaptive_block, self.adaptive_c,
            self.blur, self.open_kernel, self.close_kernel,
            self.erode, self.dilate, self.invert,
            self.min_area, self.max_area_fraction, self.circularity,
            self.prefer_bright,
        )
        for variable in variables:
            variable.trace_add("write", self._schedule_live_refresh)

    def _schedule_live_refresh(self, *_args) -> None:
        if not self._live_refresh_enabled or self.current_frame_path is None:
            return
        if self._refresh_job is not None:
            try:
                self.root.after_cancel(self._refresh_job)
            except tk.TclError:
                pass
        self._refresh_job = self.root.after(60, self._refresh_current_frame)

    def _refresh_current_frame(self) -> None:
        self._refresh_job = None
        path = self.current_frame_path
        if path is not None and path.exists():
            self._process_path(path)

    def _build_vars(self) -> None:
        self.mode = tk.StringVar(value="hsv")
        self.h_low = tk.IntVar(); self.s_low = tk.IntVar(); self.v_low = tk.IntVar()
        self.h_high = tk.IntVar(); self.s_high = tk.IntVar(); self.v_high = tk.IntVar()
        self.brightness = tk.IntVar(); self.gray_threshold = tk.IntVar()
        self.adaptive_block = tk.IntVar(); self.adaptive_c = tk.DoubleVar()
        self.blur = tk.IntVar(); self.open_kernel = tk.IntVar(); self.close_kernel = tk.IntVar()
        self.erode = tk.IntVar(); self.dilate = tk.IntVar(); self.invert = tk.BooleanVar()
        self.min_area = tk.DoubleVar(); self.max_area_fraction = tk.DoubleVar(); self.circularity = tk.DoubleVar()
        self.prefer_bright = tk.BooleanVar()

    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Configuration").pack(side="left")
        ttk.Entry(top, textvariable=self.path_var).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Open", command=self.open_config).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self.save_current).pack(side="left", padx=2)
        ttk.Button(top, text="Save As", command=self.save_as).pack(side="left", padx=2)
        ttk.Button(top, text="Open image", command=self.open_image).pack(side="left", padx=2)
        ttk.Button(top, text="Start / update Isaac camera", command=self.start_or_update_camera).pack(side="left", padx=2)
        ttk.Button(top, text="Stop camera", command=self.stop_camera).pack(side="left", padx=2)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)
        controls = ttk.Frame(body, padding=6)
        previews = ttk.Frame(body, padding=4)
        body.add(controls, weight=0)
        body.add(previews, weight=1)
        self._build_controls(controls)
        self._build_previews(previews)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status).pack(side="left")
        ttk.Label(bottom, textvariable=self.metrics, font=("Segoe UI", 10, "bold")).pack(side="right")

    def _build_controls(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, width=390, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
        row = 0
        ttk.Label(frame, text="Threshold mode", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ttk.Combobox(frame, textvariable=self.mode, values=("hsv", "brightness", "hsv_or_brightness", "grayscale", "adaptive"), state="readonly", width=25).grid(row=row, column=1, sticky="ew", padx=4, pady=4); row += 1
        row = self._scale(frame, row, "Hue lower", self.h_low, 0, 179)
        row = self._scale(frame, row, "Hue upper", self.h_high, 0, 179)
        row = self._scale(frame, row, "Saturation lower", self.s_low, 0, 255)
        row = self._scale(frame, row, "Saturation upper", self.s_high, 0, 255)
        row = self._scale(frame, row, "Value lower", self.v_low, 0, 255)
        row = self._scale(frame, row, "Value upper", self.v_high, 0, 255)
        row = self._scale(frame, row, "Brightness threshold", self.brightness, 0, 255)
        row = self._scale(frame, row, "Grayscale threshold", self.gray_threshold, 0, 255)
        row = self._scale(frame, row, "Adaptive block size", self.adaptive_block, 3, 101)
        row = self._entry(frame, row, "Adaptive C", self.adaptive_c)
        ttk.Separator(frame).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(frame, text="Preprocessing", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", padx=4); row += 1
        row = self._scale(frame, row, "Gaussian blur kernel", self.blur, 0, 31)
        row = self._scale(frame, row, "Open kernel", self.open_kernel, 0, 31)
        row = self._scale(frame, row, "Close kernel", self.close_kernel, 0, 31)
        row = self._scale(frame, row, "Erode iterations", self.erode, 0, 10)
        row = self._scale(frame, row, "Dilate iterations", self.dilate, 0, 10)
        ttk.Checkbutton(frame, text="Invert mask", variable=self.invert).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        ttk.Separator(frame).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8); row += 1
        ttk.Label(frame, text="Blob selection", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", padx=4); row += 1
        row = self._entry(frame, row, "Minimum area [px]", self.min_area)
        row = self._entry(frame, row, "Maximum area fraction", self.max_area_fraction)
        row = self._entry(frame, row, "Minimum circularity", self.circularity)
        ttk.Checkbutton(frame, text="Prefer brightest candidate", variable=self.prefer_bright).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=4); row += 1
        ttk.Button(frame, text="Reset from saved JSON", command=self._load_vision).grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=(12, 4))
        frame.columnconfigure(1, weight=1)

    def _scale(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, lower: int, upper: int) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        holder = ttk.Frame(parent)
        holder.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
        tk.Scale(holder, variable=variable, from_=lower, to=upper, orient="horizontal", showvalue=False, resolution=1, length=205).pack(side="left", fill="x", expand=True)
        ttk.Entry(holder, textvariable=variable, width=6).pack(side="left", padx=(4, 0))
        return row + 1

    def _entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=3)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        return row + 1

    def _build_previews(self, parent: ttk.Frame) -> None:
        for column in range(3):
            parent.columnconfigure(column, weight=1)
        for row in range(2):
            parent.rowconfigure(row, weight=1)
        titles = ("RGB frame", "HSV visualization", "Raw mask", "Processed mask", "Grayscale", "Detection overlay")
        self.panels: dict[str, ImagePanel] = {}
        for index, title in enumerate(titles):
            panel = ImagePanel(parent, title)
            panel.grid(row=index // 3, column=index % 3, sticky="nsew", padx=3, pady=3)
            self.panels[title] = panel

    def _load_vision(self) -> None:
        vision = self.cfg["vision"]
        self.mode.set(vision["threshold_mode"])
        self.h_low.set(vision["hsv_lower"][0]); self.s_low.set(vision["hsv_lower"][1]); self.v_low.set(vision["hsv_lower"][2])
        self.h_high.set(vision["hsv_upper"][0]); self.s_high.set(vision["hsv_upper"][1]); self.v_high.set(vision["hsv_upper"][2])
        self.brightness.set(vision["brightness_threshold"]); self.gray_threshold.set(vision["grayscale_threshold"])
        self.adaptive_block.set(vision["adaptive_block_size"]); self.adaptive_c.set(vision["adaptive_c"])
        self.blur.set(vision["gaussian_blur_kernel"]); self.open_kernel.set(vision["morph_open_kernel"]); self.close_kernel.set(vision["morph_close_kernel"])
        self.erode.set(vision["erode_iterations"]); self.dilate.set(vision["dilate_iterations"]); self.invert.set(vision["invert_mask"])
        self.min_area.set(vision["min_area_px"]); self.max_area_fraction.set(vision["max_area_fraction"]); self.circularity.set(vision["min_circularity"]); self.prefer_bright.set(vision["prefer_brightest_blob"])

    def _apply_vision(self) -> None:
        vision = self.cfg["vision"]
        vision.update(
            {
                "threshold_mode": self.mode.get(),
                "hsv_lower": [int(self.h_low.get()), int(self.s_low.get()), int(self.v_low.get())],
                "hsv_upper": [int(self.h_high.get()), int(self.s_high.get()), int(self.v_high.get())],
                "brightness_threshold": int(self.brightness.get()),
                "grayscale_threshold": int(self.gray_threshold.get()),
                "adaptive_block_size": int(self.adaptive_block.get()),
                "adaptive_c": float(self.adaptive_c.get()),
                "gaussian_blur_kernel": int(self.blur.get()),
                "morphology_kernel": int(self.open_kernel.get()),
                "morph_open_kernel": int(self.open_kernel.get()),
                "morph_close_kernel": int(self.close_kernel.get()),
                "erode_iterations": int(self.erode.get()),
                "dilate_iterations": int(self.dilate.get()),
                "invert_mask": bool(self.invert.get()),
                "min_area_px": float(self.min_area.get()),
                "max_area_fraction": float(self.max_area_fraction.get()),
                "min_circularity": float(self.circularity.get()),
                "prefer_brightest_blob": bool(self.prefer_bright.get()),
            }
        )

    def open_config(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, initialdir=str(PROJECT_ROOT / "config"), filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        self.config_path = Path(path).resolve()
        self.cfg = load_config(self.config_path)
        self.path_var.set(str(self.config_path))
        self._load_vision()
        self.status.set("Configuration loaded")

    def save_current(self) -> bool:
        try:
            self._apply_vision()
            validate_config(self.cfg)
            save_config(self.cfg, self.config_path)
            self.status.set(f"Saved {self.config_path.name}")
            return True
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc), parent=self.root)
            return False

    def save_as(self) -> None:
        path = filedialog.asksaveasfilename(parent=self.root, initialdir=str(PROJECT_ROOT / "config"), defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.config_path = Path(path).resolve()
        self.path_var.set(str(self.config_path))
        self.save_current()

    def open_image(self) -> None:
        path = filedialog.askopenfilename(parent=self.root, filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")])
        if path:
            self.offline_frame_path = Path(path)
            self.status.set(f"Offline image: {self.offline_frame_path.name}")
            self._process_path(self.offline_frame_path)

    def start_or_update_camera(self) -> None:
        if not self.save_current():
            return
        revision = write_command(DETECTOR_COMMAND, "reload", self.config_path)
        if self.worker is None or self.worker.poll() is not None:
            env = os.environ.copy()
            env["CRAZYFLIE_BRKGA_CONFIG"] = str(self.config_path)
            bat = PROJECT_ROOT / "run_detector_capture.bat"
            command = ["cmd", "/c", str(bat)] if os.name == "nt" else [sys.executable, str(PROJECT_ROOT / "detector_capture_isaac.py")]
            self.worker = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env)
            self.status.set(f"Isaac detector camera started, revision {revision}")
        else:
            self.status.set(f"Isaac detector scene updated, revision {revision}")
        self.offline_frame_path = None

    def stop_camera(self) -> None:
        write_command(DETECTOR_COMMAND, "stop", self.config_path)
        self.status.set("Stop requested")

    def _tick(self) -> None:
        try:
            if self.offline_frame_path is None and DETECTOR_FRAME.exists():
                mtime = DETECTOR_FRAME.stat().st_mtime
                if mtime > self.last_frame_mtime:
                    self.last_frame_mtime = mtime
                    self._process_path(DETECTOR_FRAME)
            status = read_json(DETECTOR_STATUS, {}) or {}
            if status.get("state") == "failed":
                self.status.set("Isaac detector worker failed; check terminal output")
        except Exception as exc:
            self.status.set(f"Preview error: {exc}")
        self.root.after(100, self._tick)

    def _process_path(self, path: Path) -> None:
        self.current_frame_path = path
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._apply_vision()
        started = time.perf_counter()
        detector = LuminousBlobDetector.from_config(self.cfg)
        views = detector.debug_views(rgb)
        duration_ms = (time.perf_counter() - started) * 1000.0
        self.panels["RGB frame"].set_rgb(views.rgb)
        self.panels["HSV visualization"].set_rgb(views.hsv_visualization)
        self.panels["Raw mask"].set_gray(views.raw_mask)
        self.panels["Processed mask"].set_gray(views.processed_mask)
        self.panels["Grayscale"].set_gray(views.grayscale)
        self.panels["Detection overlay"].set_rgb(views.overlay)
        obs = views.observation
        centroid = "none" if obs.centroid_x is None else f"({obs.centroid_x:.1f}, {obs.centroid_y:.1f})"
        self.metrics.set(f"visible={obs.visible} | centroid={centroid} | error={obs.euclidean_norm:.4f} | area={obs.area_px:.1f}px | {duration_ms:.2f} ms")

    def close(self) -> None:
        try:
            if self.worker is not None and self.worker.poll() is None:
                write_command(DETECTOR_COMMAND, "stop", self.config_path)
        finally:
            self.root.destroy()


def run_detector_gui(config_path: str | Path | None = None) -> None:
    root = tk.Tk()
    LandmarkDetectorConfigurator(root, config_path=config_path)
    root.mainloop()
