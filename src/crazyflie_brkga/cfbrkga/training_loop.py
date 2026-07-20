from __future__ import annotations

import math
import statistics
import traceback
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .brkga import BrkgaEngine
from .checkpoint import save_checkpoint
from .config import resolve_project_path
from .csv_export import export_csv_files
from .database import ExperimentDatabase
from .model import GENOME_LENGTH, parameter_manifest
from .types import EvaluationResult, Individual


class TrainingCoordinator:
    def __init__(self, cfg: dict[str, Any], evaluator: Any) -> None:
        self.cfg = cfg
        self.evaluator = evaluator
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_root = resolve_project_path(cfg, cfg["data"]["output_root"])
        self.output_dir = output_root / f"{timestamp}_{cfg['project']['name']}"
        self.output_dir.mkdir(parents=True, exist_ok=False)
        clean_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
        (self.output_dir / "resolved_config.json").write_text(
            json.dumps(clean_cfg, indent=2) + "\n", encoding="utf-8"
        )
        run_metadata = {
            "schema_version": 1,
            "policy_layers": list(cfg["policy"]["layers"]),
            "genome_length": GENOME_LENGTH,
            "parameter_layout": parameter_manifest(),
            "decoder": cfg["policy"]["decoder"],
            "action_mapping": {
                "0": "forward",
                "1": "yaw_left",
                "2": "yaw_right",
                "3": "up",
                "4": "down",
            },
            "observation_mapping": ["front_tof_m", "euclid_norm"],
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
        )
        self.csv_dir = self.output_dir / "csv"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.db = ExperimentDatabase(self.output_dir / "run.sqlite")
        self.engine = BrkgaEngine.from_config(cfg)
        self.run_id = self.db.create_run(cfg, self.output_dir, evaluator.backend_name)
        self.db.ensure_environment_slots(self.run_id, int(cfg["environment"]["num_parallel_envs"]))
        self.evaluator.output_dir = self.output_dir

    def run(self) -> Path:
        population = self.engine.initial_population()
        run_status = "failed"
        try:
            for generation_index in range(int(self.cfg["brkga"]["generations"])):
                generation_id = self.db.start_generation(self.run_id, generation_index)
                self.db.insert_population(self.run_id, generation_id, population)
                results = self._evaluate_generation(generation_id, generation_index, population)
                fitness_by_id = self._aggregate_fitness(results)
                for item in population:
                    item.fitness = fitness_by_id[item.individual_id]
                    self.db.set_individual_status(self.run_id, item.individual_id, "finished", item.fitness)
                metrics = self._generation_metrics(population, results)
                self.db.finish_generation(generation_id, metrics)
                export_csv_files(self.db, self.run_id, self.csv_dir)
                checkpoint = save_checkpoint(
                    self.checkpoint_dir / f"generation_{generation_index:04d}.npz",
                    generation_index,
                    population,
                    self.engine.get_rng_state(),
                )
                self.db.add_checkpoint(self.run_id, generation_index, checkpoint)
                print(
                    f"[generation {generation_index:04d}] best={metrics['best']:.6f} "
                    f"mean={metrics['mean']:.6f} success={metrics['success_rate']:.3f}"
                )
                if generation_index + 1 < int(self.cfg["brkga"]["generations"]):
                    population = self.engine.evolve(population, generation_index + 1)
            run_status = "finished"
            return self.output_dir
        except Exception as exc:
            self.db.insert_error(self.run_id, "training_coordinator", str(exc), traceback.format_exc())
            raise
        finally:
            self.db.finish_run(self.run_id, run_status)
            export_csv_files(self.db, self.run_id, self.csv_dir)
            self.db.close()

    def _evaluate_generation(
        self, generation_id: int, generation_index: int, population: list[Individual]
    ) -> list[EvaluationResult]:
        env_count = int(self.cfg["environment"]["num_parallel_envs"])
        seeds = [int(value) for value in self.cfg["brkga"]["episode_seeds"]]
        all_results: list[EvaluationResult] = []
        for seed in seeds:
            for start in range(0, len(population), env_count):
                batch = population[start:start + env_count]
                evaluation_ids: list[int] = []
                for env_index, item in enumerate(batch):
                    evaluation_ids.append(
                        self.db.start_evaluation(self.run_id, generation_id, item.individual_id, env_index, seed)
                    )
                evaluation_context = {
                    evaluation_id: (env_index, item.individual_id)
                    for env_index, (evaluation_id, item) in enumerate(zip(evaluation_ids, batch))
                }
                heartbeat_steps = max(1, int(self.cfg["data"].get("heartbeat_steps", 10)))

                def step_callback(evaluation_id: int, row: dict) -> None:
                    if evaluation_id not in evaluation_context:
                        raise KeyError(f"Unknown evaluation_id from evaluator: {evaluation_id}")
                    env_index, individual_id = evaluation_context[evaluation_id]
                    if int(row["step_index"]) % heartbeat_steps == 0:
                        self.db.update_slot(
                            self.run_id,
                            env_index,
                            "running",
                            individual_id,
                            evaluation_id,
                            int(row["step_index"]),
                            int(row.get("camera_frame_index", row["step_index"])),
                        )
                    if (
                        self.cfg["data"]["execution_mode"] == "integrity"
                        and self.cfg["data"]["integrity"]["store_steps_in_database"]
                    ):
                        self.db.insert_step(self.run_id, evaluation_id, row)

                try:
                    results = self.evaluator.evaluate_batch(
                        batch, seed, evaluation_ids, generation_index, step_callback=step_callback
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(
                            f"Evaluator returned {len(results)} results for a batch of {len(batch)} individuals."
                        )
                except Exception as exc:
                    self.db.insert_error(self.run_id, "evaluator", str(exc), traceback.format_exc(), generation_index)
                    results = [
                        EvaluationResult(
                            individual_id=item.individual_id, generation=item.generation, env_index=env_index,
                            episode_seed=seed, reward=float(self.cfg["mission"]["reward"]["worker_error_fitness"]),
                            status="failed", reason="simulation_error", success=False, steps=0,
                            simulated_time_s=0.0, wall_time_s=0.0, collision_count=0, visible_steps=0,
                            invalid_camera_frames=0, mean_euclid_norm=1.0,
                            min_front_tof_m=float(self.cfg["mission"]["front_tof_max_m"]),
                            final_front_tof_m=float(self.cfg["mission"]["front_tof_max_m"]),
                            final_euclid_norm=1.0, extra={"error": str(exc)},
                        )
                        for env_index, item in enumerate(batch)
                    ]
                for evaluation_id, result in zip(evaluation_ids, results):
                    self.db.finish_evaluation(self.run_id, evaluation_id, result)
                    if result.debug_dir:
                        self.db.insert_artifact(
                            self.run_id, "integrity_episode", result.debug_dir, generation_index,
                            result.individual_id, evaluation_id,
                        )
                all_results.extend(results)
        return all_results

    @staticmethod
    def _aggregate_fitness(results: list[EvaluationResult]) -> dict[int, float]:
        grouped: dict[int, list[float]] = {}
        for result in results:
            grouped.setdefault(result.individual_id, []).append(result.reward)
        return {individual_id: float(np.mean(values)) for individual_id, values in grouped.items()}

    @staticmethod
    def _generation_metrics(population: list[Individual], results: list[EvaluationResult]) -> dict[str, float]:
        values = [float(item.fitness) for item in population if item.fitness is not None]
        if not values:
            raise RuntimeError("Generation has no fitness values.")
        return {
            "best": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "worst": min(values),
            "std": float(np.std(values)),
            "success_rate": statistics.fmean([float(item.success) for item in results]) if results else 0.0,
            "collision_rate": statistics.fmean([float(item.collision_count > 0) for item in results]) if results else 0.0,
        }
