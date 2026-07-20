from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


RUNTIME_ROOT = PROJECT_ROOT / "outputs" / "runtime"
SCENARIO_PREVIEW_DIR = RUNTIME_ROOT / "scenario_preview"
DETECTOR_DIR = RUNTIME_ROOT / "landmark_detector"
SCENARIO_COMMAND = SCENARIO_PREVIEW_DIR / "command.json"
SCENARIO_STATUS = SCENARIO_PREVIEW_DIR / "status.json"
DETECTOR_COMMAND = DETECTOR_DIR / "command.json"
DETECTOR_STATUS = DETECTOR_DIR / "status.json"
DETECTOR_FRAME = DETECTOR_DIR / "last_frame.png"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def next_revision(path: Path) -> int:
    current = read_json(path, {}) or {}
    return max(int(current.get("revision", 0)) + 1, int(time.time_ns()))


def write_command(path: Path, action: str, config_path: str | Path) -> int:
    revision = next_revision(path)
    write_json(
        path,
        {
            "revision": revision,
            "action": action,
            "config_path": str(Path(config_path).resolve()),
            "created_at_unix": time.time(),
        },
    )
    return revision
