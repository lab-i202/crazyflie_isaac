from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cfbrkga.brkga import BrkgaEngine
from cfbrkga.config import load_config
from cfbrkga.image_processing import LuminousBlobDetector
from cfbrkga.model import BatchedQNetwork, GENOME_LENGTH, decode_random_keys
from cfbrkga.reward import BaselineReward, MissionState


def test_genome_length_and_network():
    assert GENOME_LENGTH == 17541
    keys = np.full((2, GENOME_LENGTH), 0.5, dtype=np.float32)
    weights = decode_random_keys(keys, -2.0, 10.0)
    model = BatchedQNetwork(weights, "cpu")
    actions = model.actions(np.array([[4.0, 1.0], [1.0, 0.1]], dtype=np.float32))
    assert actions.shape == (2,)
    assert np.all((actions >= 0) & (actions < 5))


def test_brkga_evolution():
    cfg = load_config()
    cfg["brkga"]["population_size"] = 10
    engine = BrkgaEngine.from_config(cfg)
    population = engine.initial_population()
    for index, individual in enumerate(population):
        individual.fitness = float(index)
    next_population = engine.evolve(population, 1)
    assert len(next_population) == 10
    assert next_population[0].origin == "elite_copy"
    assert all(item.generation == 1 for item in next_population)


def test_luminous_detector():
    cfg = load_config()
    detector = LuminousBlobDetector.from_config(cfg)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(image, (210, 120), 14, (255, 225, 20), -1)
    result = detector.detect(image)
    assert result.visible
    assert abs(result.centroid_x - 210) < 2
    assert result.euclidean_norm > 0


def test_reward_success():
    cfg = load_config()
    reward = BaselineReward(cfg)
    outcome = reward.step(MissionState(), 0, 0.4, 0.01, True, False)
    assert outcome.success
    assert outcome.terminated
    assert outcome.reason == "success"


def test_runtime_version_major_parser():
    from cfbrkga.runtime_compat import _major

    assert _major("1.26.4") == 1
    assert _major("2.0.0") == 2
    assert _major("v3.1") == 3
