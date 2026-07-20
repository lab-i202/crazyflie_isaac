from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import ExperimentDatabase


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def safe_folder_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "experiment"


class LatestRunMirror:
    """Maintains a stable, DBeaver-friendly mirror of the latest experiment.

    The timestamped experiment folder remains authoritative. The stable mirror keeps:
      - a consistent SQLite backup at ``run.sqlite``;
      - resolved configuration and run metadata;
      - CSV exports;
      - checkpoints;
      - a source manifest pointing to the authoritative timestamped folder.

    Raw integrity images are not duplicated. Artifact paths stored in SQLite remain
    absolute and continue to point to the authoritative experiment folder.
    """

    def __init__(
        self,
        output_root: Path,
        source_dir: Path,
        project_name: str,
        settings: dict[str, Any] | None = None,
    ) -> None:
        cfg = settings or {}
        self.enabled = bool(cfg.get("enabled", True))
        requested_name = str(cfg.get("folder_name", "")).strip()
        folder_name = requested_name or f"last_{safe_folder_component(project_name)}"
        self.mirror_dir = output_root / folder_name
        self.source_dir = source_dir
        self.database_sync_interval_s = max(0.25, float(cfg.get("database_sync_interval_s", 2.0)))
        self.copy_csv = bool(cfg.get("copy_csv", True))
        self.copy_checkpoints = bool(cfg.get("copy_checkpoints", True))
        self._last_database_sync_monotonic = 0.0

        if self.enabled:
            self._prepare_directory()

    @property
    def database_path(self) -> Path:
        return self.mirror_dir / "run.sqlite"

    def _prepare_directory(self) -> None:
        self.mirror_dir.mkdir(parents=True, exist_ok=True)

        # Keep run.sqlite in place so an existing DBeaver connection can continue
        # pointing at the same path. Remove only derived files from the prior run.
        for directory_name in ("csv", "checkpoints"):
            path = self.mirror_dir / directory_name
            if path.exists():
                shutil.rmtree(path)

        for filename in (
            "resolved_config.json",
            "run_metadata.json",
            "source_experiment.json",
            "SOURCE_EXPERIMENT.txt",
        ):
            path = self.mirror_dir / filename
            if path.exists():
                path.unlink()

        self._write_source_manifest(status="initializing")

    def _write_source_manifest(self, status: str) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": 1,
            "status": status,
            "authoritative_experiment_dir": str(self.source_dir.resolve()),
            "mirror_dir": str(self.mirror_dir.resolve()),
            "updated_at": utc_now(),
            "raw_debug_artifacts_duplicated": False,
        }
        (self.mirror_dir / "source_experiment.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        (self.mirror_dir / "SOURCE_EXPERIMENT.txt").write_text(
            f"Authoritative experiment folder:\n{self.source_dir.resolve()}\n",
            encoding="utf-8",
        )

    def sync_database(self, database: ExperimentDatabase, force: bool = False) -> bool:
        if not self.enabled:
            return False

        now = time.monotonic()
        if not force and now - self._last_database_sync_monotonic < self.database_sync_interval_s:
            return False

        try:
            database.backup_to(self.database_path)
            self._last_database_sync_monotonic = now
            self._write_source_manifest(status="running")
            return True
        except Exception as exc:
            # This mirror is a convenience feature. A temporary DBeaver lock must
            # never abort training or corrupt the authoritative experiment.
            print(f"[latest mirror warning] SQLite backup skipped: {exc}", flush=True)
            return False

    def sync_files(self, status: str = "running") -> None:
        if not self.enabled:
            return

        for filename in ("resolved_config.json", "run_metadata.json"):
            source = self.source_dir / filename
            if source.exists():
                shutil.copy2(source, self.mirror_dir / filename)

        if self.copy_csv:
            self._copy_directory(self.source_dir / "csv", self.mirror_dir / "csv")
        if self.copy_checkpoints:
            self._copy_directory(self.source_dir / "checkpoints", self.mirror_dir / "checkpoints")

        self._write_source_manifest(status=status)

    def sync_all(
        self,
        database: ExperimentDatabase,
        status: str,
        force_database: bool = True,
    ) -> None:
        if not self.enabled:
            return
        self.sync_database(database, force=force_database)
        try:
            self.sync_files(status=status)
        except Exception as exc:
            print(f"[latest mirror warning] File mirror skipped: {exc}", flush=True)

    @staticmethod
    def _copy_directory(source: Path, destination: Path) -> None:
        if not source.exists():
            return
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)
