from __future__ import annotations

import json
from pathlib import Path

from cfbrkga.config import load_config
from cfbrkga.mock_evaluator import MockBatchEvaluator
from cfbrkga.training_loop import TrainingCoordinator


def test_mock_training_smoke(tmp_path: Path):
    cfg = load_config()
    cfg["_project_root"] = str(tmp_path)
    cfg["project"]["name"] = "mirror_smoke"
    cfg["data"]["output_root"] = "outputs"
    cfg["data"]["latest_mirror"]["database_sync_interval_s"] = 0.25
    cfg["brkga"]["population_size"] = 4
    cfg["brkga"]["generations"] = 2
    cfg["brkga"]["episode_seeds"] = [1]
    cfg["environment"]["num_parallel_envs"] = 2
    cfg["mission"]["max_steps"] = 4
    cfg["mission"]["lost_landmark_max_steps"] = 4

    evaluator = MockBatchEvaluator(cfg, tmp_path)
    coordinator = TrainingCoordinator(cfg, evaluator)
    output = coordinator.run()

    assert (output / "run.sqlite").exists()
    assert (output / "csv" / "generations.csv").exists()

    latest = tmp_path / "outputs" / "last_mirror_smoke"
    assert (latest / "run.sqlite").exists()
    assert (latest / "csv" / "generations.csv").exists()
    assert (latest / "checkpoints" / "generation_0001.npz").exists()

    manifest = json.loads((latest / "source_experiment.json").read_text(encoding="utf-8"))
    assert Path(manifest["authoritative_experiment_dir"]) == output.resolve()
    assert manifest["status"] == "finished"
