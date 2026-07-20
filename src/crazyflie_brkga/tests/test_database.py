from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from cfbrkga.brkga import BrkgaEngine
from cfbrkga.config import load_config
from cfbrkga.database import ExperimentDatabase, decode_array


def test_database_roundtrip_and_live_backup(tmp_path: Path):
    cfg = load_config()
    cfg["brkga"]["population_size"] = 3
    db = ExperimentDatabase(tmp_path / "run.sqlite")
    run_id = db.create_run(cfg, tmp_path, "test")
    generation_id = db.start_generation(run_id, 0)
    engine = BrkgaEngine.from_config(cfg)
    population = engine.initial_population()
    db.insert_population(run_id, generation_id, population)

    rows = db.query("SELECT data_blob FROM arrays ORDER BY array_id LIMIT 1")
    restored = decode_array(rows[0]["data_blob"])
    assert restored.shape == (17541,)

    db.ensure_environment_slots(run_id, 1)
    db.update_slot(run_id, 0, "running", population[0].individual_id, 1, 0, 0)
    event = db.query(
        "SELECT previous_state, new_state, current_state FROM agent_state_events ORDER BY event_id DESC LIMIT 1"
    )[0]
    assert event["previous_state"] == "waiting"
    assert event["new_state"] == "running"
    assert event["current_state"] == "running"

    mirror_path = tmp_path / "last_test" / "run.sqlite"
    db.backup_to(mirror_path)
    mirror = sqlite3.connect(mirror_path)
    try:
        assert mirror.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert mirror.execute("SELECT current_state FROM agent_state_events").fetchone()[0] == "running"
        mirror.commit()

        # Keep a reader connected, as DBeaver normally would, and refresh the mirror.
        db.update_slot(run_id, 0, "finished", population[0].individual_id, 1, 1, 1)
        db.backup_to(mirror_path)
        assert mirror.execute(
            "SELECT current_state FROM agent_state_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()[0] == "finished"
    finally:
        mirror.close()

    db.finish_run(run_id, "finished")
    db.close()


def test_schema_migrates_old_agent_state_events(tmp_path: Path):
    path = tmp_path / "old.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE schema_info(schema_version INTEGER NOT NULL, created_at TEXT NOT NULL);
        INSERT INTO schema_info VALUES (1, 'old');
        CREATE TABLE agent_state_events(
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            evaluation_id INTEGER,
            individual_id INTEGER,
            env_index INTEGER NOT NULL,
            previous_state TEXT,
            new_state TEXT NOT NULL,
            reason TEXT,
            simulation_step INTEGER,
            created_at TEXT NOT NULL
        );
        INSERT INTO agent_state_events(
            run_id, env_index, previous_state, new_state, reason, simulation_step, created_at
        ) VALUES (1, 0, 'waiting', 'running', '', 0, 'old');
        """
    )
    connection.commit()
    connection.close()

    db = ExperimentDatabase(path)
    try:
        row = db.query("SELECT new_state, current_state FROM agent_state_events")[0]
        assert row["new_state"] == "running"
        assert row["current_state"] == "running"
        schema_version = db.query("SELECT schema_version FROM schema_info")[0]["schema_version"]
        assert schema_version == 2
    finally:
        db.close()
