# last_frame_previewer.py
#
# Lightweight live previewer for camera_onboard/last_frame.png and
# camera_isometric/last_frame.png produced by scenario/crazyflie_room_scene.py.
#
# Run this in a second terminal while Isaac Sim is running:
#   python last_frame_previewer.py
#
# No argparse. The GUI exposes all paths and refresh settings.

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

try:
    from PIL import Image, ImageTk
except Exception:  # Pillow is optional. Tk PhotoImage can still load PNGs.
    Image = None
    ImageTk = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_NAME = "crazyflie_room_basic"
DEFAULT_ONBOARD = PROJECT_ROOT / "outputs" / DEFAULT_SCENE_NAME / "camera_onboard" / "last_frame.png"
DEFAULT_ISOMETRIC = PROJECT_ROOT / "outputs" / DEFAULT_SCENE_NAME / "camera_isometric" / "last_frame.png"


class LastFramePreviewer:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Crazyflie last_frame.png previewer")
        self.root.geometry("1120x680")
        self.root.minsize(820, 520)

        self.onboard_path = tk.StringVar(value=str(DEFAULT_ONBOARD))
        self.isometric_path = tk.StringVar(value=str(DEFAULT_ISOMETRIC))
        self.refresh_ms = tk.IntVar(value=100)
        self.running = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Waiting for last_frame.png files...")

        self._onboard_photo: tk.PhotoImage | None = None
        self._isometric_photo: tk.PhotoImage | None = None
        self._last_mtime_ns: dict[str, int] = {}

        self._build_ui()
        self._schedule_refresh()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)

        controls = ttk.LabelFrame(main, text="Inputs", padding=8)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Onboard last_frame.png").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.onboard_path).grid(row=0, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(controls, text="Browse", command=lambda: self._browse(self.onboard_path)).grid(row=0, column=2, pady=3)

        ttk.Label(controls, text="Isometric last_frame.png").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.isometric_path).grid(row=1, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(controls, text="Browse", command=lambda: self._browse(self.isometric_path)).grid(row=1, column=2, pady=3)

        ttk.Label(controls, text="Refresh [ms]").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Spinbox(controls, from_=30, to=2000, increment=10, textvariable=self.refresh_ms, width=10).grid(row=2, column=1, sticky="w", padx=8, pady=3)
        ttk.Checkbutton(controls, text="Live refresh", variable=self.running).grid(row=2, column=2, sticky="w", pady=3)

        images = ttk.Frame(main)
        images.grid(row=1, column=0, sticky="nsew")
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)
        images.rowconfigure(1, weight=1)

        ttk.Label(images, text="Onboard", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(images, text="Isometric", font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky="w")

        self.onboard_label = ttk.Label(images, text="No image yet", anchor="center")
        self.onboard_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        self.isometric_label = ttk.Label(images, text="No image yet", anchor="center")
        self.isometric_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))

        ttk.Label(main, textvariable=self.status, foreground="gray").grid(row=2, column=0, sticky="ew", pady=(8, 0))

    def _browse(self, variable: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select last_frame.png",
            initialdir=str(PROJECT_ROOT / "outputs"),
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if path:
            variable.set(path)
            self._last_mtime_ns.pop(path, None)
            self._refresh_once(force=True)

    def _schedule_refresh(self) -> None:
        if self.running.get():
            self._refresh_once(force=False)
        delay = max(int(self.refresh_ms.get()), 30)
        self.root.after(delay, self._schedule_refresh)

    def _refresh_once(self, force: bool) -> None:
        onboard_ok = self._load_into_label(Path(self.onboard_path.get()), self.onboard_label, "onboard", force)
        isometric_ok = self._load_into_label(Path(self.isometric_path.get()), self.isometric_label, "isometric", force)
        now = time.strftime("%H:%M:%S")
        self.status.set(
            f"{now} | onboard={'ok' if onboard_ok else 'missing'} | "
            f"isometric={'ok' if isometric_ok else 'missing'} | refresh={max(int(self.refresh_ms.get()), 30)} ms"
        )

    def _load_into_label(self, path: Path, label: ttk.Label, slot: str, force: bool) -> bool:
        if not path.exists() or not path.is_file():
            label.configure(text=f"Missing:\n{path}", image="")
            return False

        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError:
            return False

        key = str(path.resolve())
        if not force and self._last_mtime_ns.get(key) == mtime_ns:
            return True
        self._last_mtime_ns[key] = mtime_ns

        width = max(label.winfo_width(), 320)
        height = max(label.winfo_height(), 240)

        try:
            if Image is not None and ImageTk is not None:
                image = Image.open(path)
                image.thumbnail((width, height))
                photo = ImageTk.PhotoImage(image)
            else:
                # Tk PhotoImage supports PNG on normal modern Tk builds but does not resize well.
                photo = tk.PhotoImage(file=str(path))
        except Exception as exc:
            label.configure(text=f"Could not load:\n{path}\n{exc}", image="")
            return False

        if slot == "onboard":
            self._onboard_photo = photo
            label.configure(image=self._onboard_photo, text="")
        else:
            self._isometric_photo = photo
            label.configure(image=self._isometric_photo, text="")
        return True


if __name__ == "__main__":
    LastFramePreviewer().run()
