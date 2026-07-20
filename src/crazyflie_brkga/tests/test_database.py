from __future__ import annotations

from pathlib import Path

import numpy as np

from cfbrkga.brkga import BrkgaEngine
from cfbrkga.config import load_config
from cfbrkga.database import ExperimentDatabase, decode_array


def test_database_roundtrip(tmp_path: Path):
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
    db.finish_run(run_id, "finished")
    db.close()
