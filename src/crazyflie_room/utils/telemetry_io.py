# utils/telemetry_io.py
#
# Runtime telemetry writers shared by scenario runners.
#
# This module is pure Python. It does not import Isaac Sim.

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TelemetryWriter:
    def __init__(
        self,
        output_root: Path,
        scene_name: str,
        enabled: bool = True,
        write_latest_state_json: bool = True,
        write_state_history_csv: bool = True,
    ) -> None:
        self.enabled = enabled
        self.write_latest_state_json = write_latest_state_json
        self.write_state_history_csv = write_state_history_csv
        self.scene_name = scene_name
        self.output_dir = output_root / scene_name / "telemetry"
        self.latest_state_path = self.output_dir / "state_latest.json"
        self.history_path = self.output_dir / "state_history.csv"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self._csv_file = None
        self._csv_writer = None
        self._fieldnames: list[str] | None = None

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(metadata)
        payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        with self.metadata_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def write_state(self, state: dict[str, Any], write_csv_row: bool) -> None:
        if not self.enabled:
            return

        if self.write_latest_state_json:
            with self.latest_state_path.open("w", encoding="utf-8") as file:
                json.dump(state, file, indent=2)

        if self.write_state_history_csv and write_csv_row:
            flat_state = flatten_dict(state)
            if self._csv_writer is None:
                self._fieldnames = list(flat_state.keys())
                self._csv_file = self.history_path.open("w", newline="", encoding="utf-8")
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
                self._csv_writer.writeheader()

            # Preserve the initial field set. Unknown later fields are ignored instead of corrupting CSV shape.
            assert self._fieldnames is not None
            self._csv_writer.writerow({key: flat_state.get(key, "") for key in self._fieldnames})
            self._csv_file.flush()

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                indexed_key = f"{full_key}.{index}"
                if isinstance(item, dict):
                    flat.update(flatten_dict(item, prefix=indexed_key))
                else:
                    flat[indexed_key] = item
        else:
            flat[full_key] = value
    return flat
