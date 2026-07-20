from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, ttk
from typing import Callable, Iterable


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_wheel)
        self.canvas.bind("<Leave>", self._unbind_wheel)

    def _on_content_configure(self, _event: tk.Event) -> None:
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)

    def _bind_wheel(self, _event: tk.Event) -> None:
        self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        self.canvas.bind_all("<Button-4>", self._on_linux_wheel)
        self.canvas.bind_all("<Button-5>", self._on_linux_wheel)

    def _unbind_wheel(self, _event: tk.Event) -> None:
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_linux_wheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(-1 if event.num == 4 else 1, "units")


class ColorEntry(ttk.Frame):
    def __init__(
        self,
        parent: tk.Widget,
        variable: tk.StringVar,
        *,
        width: int = 12,
        changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.variable = variable
        self.changed = changed
        self.entry = tk.Entry(self, textvariable=variable, width=width, relief="solid", borderwidth=1)
        self.entry.pack(side="left")
        ttk.Button(self, text="Pick", command=self.pick).pack(side="left", padx=(6, 0))
        self.variable.trace_add("write", self._on_value_changed)
        self.entry.bind("<FocusOut>", lambda _event: self.refresh())
        self.refresh()

    def pick(self) -> None:
        initial = self.variable.get() or "#ffffff"
        _rgb, value = colorchooser.askcolor(color=initial, parent=self)
        if value:
            self.variable.set(value.lower())
            if self.changed:
                self.changed()

    def _on_value_changed(self, *_args) -> None:
        self.refresh()
        if self.changed:
            self.changed()

    def refresh(self) -> None:
        value = self.variable.get().strip()
        try:
            r, g, b = hex_to_rgb255(value)
        except ValueError:
            self.entry.configure(background="white", foreground="black")
            return
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        foreground = "black" if luminance >= 145 else "white"
        self.entry.configure(background=value, foreground=foreground, insertbackground=foreground)


def hex_to_rgb255(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError("Color must use #RRGGBB.")
    return tuple(int(text[index : index + 2], 16) for index in (0, 2, 4))


def rgb255_to_hex(values: Iterable[float | int]) -> str:
    items = [max(0, min(255, int(round(float(value))))) for value in values]
    if len(items) != 3:
        raise ValueError("RGB requires three values.")
    return "#{:02x}{:02x}{:02x}".format(*items)


def rgb01_to_hex(values: Iterable[float | int]) -> str:
    return rgb255_to_hex(float(value) * 255.0 for value in values)


def hex_to_rgb01(value: str) -> list[float]:
    return [component / 255.0 for component in hex_to_rgb255(value)]


def parse_number_list(text: str, length: int, cast=float) -> list:
    cleaned = text.replace(";", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != length:
        raise ValueError(f"Expected {length} comma-separated values, got {len(parts)}.")
    return [cast(part) for part in parts]


def format_number_list(values: Iterable[float | int]) -> str:
    output: list[str] = []
    for value in values:
        number = float(value)
        output.append(str(int(number)) if number.is_integer() else f"{number:.6g}")
    return ", ".join(output)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def room_to_canvas(
    x: float,
    y: float,
    room_x: float,
    room_y: float,
    width: int,
    height: int,
    margin: int = 24,
) -> tuple[float, float, float]:
    usable_width = max(1.0, width - 2 * margin)
    usable_height = max(1.0, height - 2 * margin)
    scale = min(usable_width / max(room_x, 1e-6), usable_height / max(room_y, 1e-6))
    cx = width / 2.0 + x * scale
    cy = height / 2.0 - y * scale
    return cx, cy, scale


def canvas_to_room(
    cx: float,
    cy: float,
    room_x: float,
    room_y: float,
    width: int,
    height: int,
    margin: int = 24,
) -> tuple[float, float]:
    _, _, scale = room_to_canvas(0.0, 0.0, room_x, room_y, width, height, margin)
    x = (cx - width / 2.0) / scale
    y = -(cy - height / 2.0) / scale
    return (
        max(-room_x / 2.0, min(room_x / 2.0, x)),
        max(-room_y / 2.0, min(room_y / 2.0, y)),
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
