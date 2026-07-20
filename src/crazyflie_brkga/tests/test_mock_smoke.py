from __future__ import annotations

from pathlib import Path

from cfbrkga.config import load_config
from cfbrkga.mock_evaluator import MockBatchEvaluator
from cfbrkga.training_loop import TrainingCoordinator


def test_mock_training_smoke(tmp_path: Path):
    cfg = load_config()
    cfg["_project_root"] = str(tmp_path)
    cfg["data"]["output_root"] = "outputs"
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
