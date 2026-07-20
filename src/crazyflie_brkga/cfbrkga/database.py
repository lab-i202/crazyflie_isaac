from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import threading
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .types import EvaluationResult, Individual


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def encode_array(array: np.ndarray) -> tuple[bytes, str, str, int, int]:
    value = np.asarray(array)
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    raw = buffer.getvalue()
    compressed = zlib.compress(raw, level=6)
    return compressed, str(value.dtype), json.dumps(list(value.shape)), len(raw), len(compressed)


def decode_array(blob: bytes) -> np.ndarray:
    raw = zlib.decompress(blob)
    return np.load(io.BytesIO(raw), allow_pickle=False)


SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_info (
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    config_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    host TEXT,
    process_id INTEGER,
    current_generation INTEGER DEFAULT 0,
    note TEXT
);

CREATE TABLE IF NOT EXISTS arrays (
    array_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    dtype TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    compression TEXT NOT NULL,
    bytes_raw INTEGER NOT NULL,
    bytes_stored INTEGER NOT NULL,
    data_blob BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    best_fitness REAL,
    mean_fitness REAL,
    median_fitness REAL,
    worst_fitness REAL,
    std_fitness REAL,
    success_rate REAL,
    collision_rate REAL,
    UNIQUE(run_id, generation_index)
);

CREATE TABLE IF NOT EXISTS individuals (
    individual_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    index_in_generation INTEGER NOT NULL,
    origin TEXT NOT NULL,
    elite_parent_id INTEGER,
    non_elite_parent_id INTEGER,
    chromosome_array_id INTEGER NOT NULL REFERENCES arrays(array_id),
    weights_array_id INTEGER NOT NULL REFERENCES arrays(array_id),
    fitness REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, individual_id),
    UNIQUE(run_id, generation_index, index_in_generation)
);

CREATE TABLE IF NOT EXISTS environment_slots (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    env_index INTEGER NOT NULL,
    current_individual_id INTEGER,
    current_evaluation_id INTEGER,
    state TEXT NOT NULL,
    state_since TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL,
    last_camera_frame INTEGER,
    last_simulation_step INTEGER,
    message TEXT,
    PRIMARY KEY(run_id, env_index)
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id) ON DELETE CASCADE,
    individual_id INTEGER NOT NULL,
    env_index INTEGER NOT NULL,
    episode_seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    reward REAL,
    success INTEGER,
    reason TEXT,
    steps INTEGER,
    simulated_time_s REAL,
    wall_time_s REAL,
    collision_count INTEGER,
    visible_steps INTEGER,
    invalid_camera_frames INTEGER,
    mean_euclid_norm REAL,
    min_front_tof_m REAL,
    final_front_tof_m REAL,
    final_euclid_norm REAL,
    debug_dir TEXT,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS agent_state_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluation_id INTEGER,
    individual_id INTEGER,
    env_index INTEGER NOT NULL,
    previous_state TEXT,
    new_state TEXT NOT NULL,
    reason TEXT,
    simulation_step INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_steps (
    step_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    simulation_time_s REAL NOT NULL,
    camera_frame_index INTEGER,
    action INTEGER NOT NULL,
    reward REAL NOT NULL,
    front_tof_m REAL NOT NULL,
    euclid_norm REAL NOT NULL,
    visible INTEGER NOT NULL,
    blob_x REAL,
    blob_y REAL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    yaw_rad REAL NOT NULL,
    collision INTEGER NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER,
    individual_id INTEGER,
    evaluation_id INTEGER,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER,
    individual_id INTEGER,
    env_index INTEGER,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_individuals_generation ON individuals(run_id, generation_index);
CREATE INDEX IF NOT EXISTS idx_evaluations_generation ON evaluations(run_id, generation_id);
CREATE INDEX IF NOT EXISTS idx_events_run_env ON agent_state_events(run_id, env_index, created_at);
"""


class ExperimentDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=60.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=60000")
        self._conn.executescript(SCHEMA)
        row = self._conn.execute("SELECT COUNT(*) AS n FROM schema_info").fetchone()
        if row["n"] == 0:
            self._conn.execute("INSERT INTO schema_info(schema_version, created_at) VALUES (?, ?)", (1, utc_now()))
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()

    def create_run(self, cfg: dict, output_dir: Path, backend: str) -> int:
        import os
        import socket

        clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO runs(name, status, mode, backend, started_at, config_json, config_path,
                   output_dir, host, process_id, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    cfg["project"]["name"], "running", cfg["data"]["execution_mode"], backend,
                    utc_now(), json.dumps(clean_cfg), cfg.get("_config_path", ""), str(output_dir),
                    socket.gethostname(), os.getpid(), cfg["project"].get("note", ""),
                ),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE runs SET status=?, finished_at=? WHERE run_id=?", (status, utc_now(), run_id))

    def start_generation(self, run_id: int, generation_index: int) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO generations(run_id, generation_index, status, started_at) VALUES (?, ?, ?, ?)",
                (run_id, generation_index, "running", utc_now()),
            )
            self._conn.execute("UPDATE runs SET current_generation=? WHERE run_id=?", (generation_index, run_id))
            return int(cursor.lastrowid)

    def finish_generation(self, generation_id: int, metrics: dict[str, float]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE generations SET status='finished', finished_at=?, best_fitness=?, mean_fitness=?,
                   median_fitness=?, worst_fitness=?, std_fitness=?, success_rate=?, collision_rate=?
                   WHERE generation_id=?""",
                (
                    utc_now(), metrics["best"], metrics["mean"], metrics["median"], metrics["worst"],
                    metrics["std"], metrics["success_rate"], metrics["collision_rate"], generation_id,
                ),
            )

    def put_array(self, array: np.ndarray, kind: str) -> int:
        blob, dtype, shape, raw_n, stored_n = encode_array(array)
        sha = hashlib.sha256(blob).hexdigest()
        with self._lock, self._conn:
            row = self._conn.execute("SELECT array_id FROM arrays WHERE sha256=?", (sha,)).fetchone()
            if row is not None:
                return int(row["array_id"])
            cursor = self._conn.execute(
                """INSERT INTO arrays(sha256, kind, dtype, shape_json, compression, bytes_raw, bytes_stored,
                   data_blob, created_at) VALUES (?, ?, ?, ?, 'zlib+npy', ?, ?, ?, ?)""",
                (sha, kind, dtype, shape, raw_n, stored_n, sqlite3.Binary(blob), utc_now()),
            )
            return int(cursor.lastrowid)

    def insert_population(self, run_id: int, generation_id: int, population: Iterable[Individual]) -> None:
        with self._lock:
            for item in population:
                chromosome_id = self.put_array(item.random_keys, "random_keys")
                weights_id = self.put_array(item.decoded_weights, "decoded_weights")
                with self._conn:
                    self._conn.execute(
                        """INSERT INTO individuals(individual_id, run_id, generation_id, generation_index,
                           index_in_generation, origin, elite_parent_id, non_elite_parent_id,
                           chromosome_array_id, weights_array_id, fitness, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item.individual_id, run_id, generation_id, item.generation, item.index_in_generation,
                            item.origin, item.elite_parent_id, item.non_elite_parent_id, chromosome_id,
                            weights_id, item.fitness, "waiting", utc_now(),
                        ),
                    )

    def set_individual_status(self, run_id: int, individual_id: int, status: str, fitness: float | None = None) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE individuals SET status=?, fitness=COALESCE(?, fitness) WHERE run_id=? AND individual_id=?",
                (status, fitness, run_id, individual_id),
            )

    def ensure_environment_slots(self, run_id: int, count: int) -> None:
        with self._lock, self._conn:
            for env_index in range(count):
                self._conn.execute(
                    """INSERT OR IGNORE INTO environment_slots(run_id, env_index, state, state_since,
                       last_heartbeat) VALUES (?, ?, 'waiting', ?, ?)""",
                    (run_id, env_index, utc_now(), utc_now()),
                )

    def update_slot(
        self, run_id: int, env_index: int, state: str, individual_id: int | None = None,
        evaluation_id: int | None = None, simulation_step: int | None = None,
        camera_frame: int | None = None, message: str = "",
    ) -> None:
        with self._lock, self._conn:
            previous = self._conn.execute(
                "SELECT state FROM environment_slots WHERE run_id=? AND env_index=?", (run_id, env_index)
            ).fetchone()
            previous_state = previous["state"] if previous else None
            self._conn.execute(
                """UPDATE environment_slots SET current_individual_id=?, current_evaluation_id=?, state=?,
                   state_since=?, last_heartbeat=?, last_camera_frame=COALESCE(?, last_camera_frame),
                   last_simulation_step=COALESCE(?, last_simulation_step), message=?
                   WHERE run_id=? AND env_index=?""",
                (
                    individual_id, evaluation_id, state, utc_now(), utc_now(), camera_frame,
                    simulation_step, message, run_id, env_index,
                ),
            )
            if previous_state != state:
                self._conn.execute(
                    """INSERT INTO agent_state_events(run_id, evaluation_id, individual_id, env_index,
                       previous_state, new_state, reason, simulation_step, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, evaluation_id, individual_id, env_index, previous_state, state,
                        message, simulation_step, utc_now(),
                    ),
                )

    def start_evaluation(
        self, run_id: int, generation_id: int, individual_id: int, env_index: int, episode_seed: int
    ) -> int:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """INSERT INTO evaluations(run_id, generation_id, individual_id, env_index, episode_seed,
                   status, started_at) VALUES (?, ?, ?, ?, ?, 'running', ?)""",
                (run_id, generation_id, individual_id, env_index, episode_seed, utc_now()),
            )
            evaluation_id = int(cursor.lastrowid)
        self.update_slot(run_id, env_index, "running", individual_id, evaluation_id, 0, 0)
        self.set_individual_status(run_id, individual_id, "running")
        return evaluation_id

    def finish_evaluation(self, run_id: int, evaluation_id: int, result: EvaluationResult) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """UPDATE evaluations SET status=?, finished_at=?, reward=?, success=?, reason=?, steps=?,
                   simulated_time_s=?, wall_time_s=?, collision_count=?, visible_steps=?,
                   invalid_camera_frames=?, mean_euclid_norm=?, min_front_tof_m=?, final_front_tof_m=?,
                   final_euclid_norm=?, debug_dir=?, extra_json=? WHERE evaluation_id=?""",
                (
                    result.status, utc_now(), result.reward, int(result.success), result.reason, result.steps,
                    result.simulated_time_s, result.wall_time_s, result.collision_count, result.visible_steps,
                    result.invalid_camera_frames, result.mean_euclid_norm, result.min_front_tof_m,
                    result.final_front_tof_m, result.final_euclid_norm, result.debug_dir,
                    json.dumps(result.extra), evaluation_id,
                ),
            )
        self.update_slot(run_id, result.env_index, "finished", result.individual_id, evaluation_id, result.steps)

    def insert_step(self, run_id: int, evaluation_id: int, row: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO episode_steps(run_id, evaluation_id, step_index, simulation_time_s,
                   camera_frame_index, action, reward, front_tof_m, euclid_norm, visible, blob_x, blob_y,
                   x_m, y_m, z_m, yaw_rad, collision, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, evaluation_id, row["step_index"], row["simulation_time_s"],
                    row.get("camera_frame_index"), row["action"], row["reward"], row["front_tof_m"],
                    row["euclid_norm"], int(row["visible"]), row.get("blob_x"), row.get("blob_y"),
                    row["x_m"], row["y_m"], row["z_m"], row["yaw_rad"], int(row["collision"]),
                    row.get("reason", ""),
                ),
            )

    def insert_artifact(
        self, run_id: int, kind: str, path: str | Path, generation_index: int | None = None,
        individual_id: int | None = None, evaluation_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO artifacts(run_id, generation_index, individual_id, evaluation_id, kind,
                   path, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id, generation_index, individual_id, evaluation_id, kind, str(path),
                    json.dumps(metadata or {}), utc_now(),
                ),
            )

    def insert_error(
        self, run_id: int | None, component: str, message: str, traceback_text: str = "",
        generation_index: int | None = None, individual_id: int | None = None, env_index: int | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """INSERT INTO errors(run_id, generation_index, individual_id, env_index, component,
                   message, traceback, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, generation_index, individual_id, env_index, component, message, traceback_text, utc_now()),
            )

    def add_checkpoint(self, run_id: int, generation_index: int, path: Path) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO checkpoints(run_id, generation_index, path, sha256, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, generation_index, str(path), digest, utc_now()),
            )

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, parameters).fetchall()
