from __future__ import annotations

import csv
import os
from pathlib import Path

from .database import ExperimentDatabase


EXPORTS = {
    "generations.csv": """SELECT generation_index, status, started_at, finished_at, best_fitness,
        mean_fitness, median_fitness, worst_fitness, std_fitness, success_rate, collision_rate
        FROM generations WHERE run_id=? ORDER BY generation_index""",
    "population_latest.csv": """SELECT i.generation_index, i.index_in_generation, i.individual_id,
        i.origin, i.elite_parent_id, i.non_elite_parent_id, i.fitness, i.status
        FROM individuals i WHERE i.run_id=? AND i.generation_index=(SELECT MAX(generation_index)
        FROM individuals WHERE run_id=?) ORDER BY i.fitness DESC""",
    "evaluations.csv": """SELECT e.evaluation_id, g.generation_index, e.individual_id, e.env_index,
        e.episode_seed, e.status, e.reward, e.success, e.reason, e.steps, e.simulated_time_s,
        e.wall_time_s, e.collision_count, e.visible_steps, e.invalid_camera_frames,
        e.mean_euclid_norm, e.min_front_tof_m, e.final_front_tof_m, e.final_euclid_norm,
        e.debug_dir, e.extra_json
        FROM evaluations e JOIN generations g ON g.generation_id=e.generation_id
        WHERE e.run_id=? ORDER BY g.generation_index, e.individual_id, e.episode_seed""",
    "best_individual_history.csv": """SELECT i.generation_index, i.individual_id,
        i.index_in_generation, i.origin, i.fitness, i.chromosome_array_id, i.weights_array_id
        FROM individuals i JOIN (SELECT generation_index, MAX(fitness) AS best_fitness
        FROM individuals WHERE run_id=? GROUP BY generation_index) best
        ON best.generation_index=i.generation_index AND best.best_fitness=i.fitness
        WHERE i.run_id=? GROUP BY i.generation_index ORDER BY i.generation_index""",
    "agent_state_events.csv": """SELECT event_id, evaluation_id, individual_id, env_index,
        previous_state, new_state, current_state, reason, simulation_step, created_at
        FROM agent_state_events WHERE run_id=? ORDER BY event_id""",
    "errors.csv": """SELECT error_id, generation_index, individual_id, env_index, component, message,
        traceback, created_at FROM errors WHERE run_id=? ORDER BY error_id""",
}


def export_csv_files(db: ExperimentDatabase, run_id: int, output_dir: str | Path) -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, sql in EXPORTS.items():
        params = (run_id, run_id) if filename in {"population_latest.csv", "best_individual_history.csv"} else (run_id,)
        rows = db.query(sql, params)
        path = destination / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            if rows:
                writer = csv.writer(handle)
                writer.writerow(rows[0].keys())
                writer.writerows([tuple(row) for row in rows])
            else:
                handle.write("")
        os.replace(temporary, path)
        written.append(path)
    return written
