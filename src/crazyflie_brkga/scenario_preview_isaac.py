from __future__ import annotations

import copy
import gc
import time
import traceback
from pathlib import Path

from cfbrkga.config import PROJECT_ROOT, load_config
from cfbrkga.runtime_files import SCENARIO_COMMAND, SCENARIO_STATUS, read_json, write_json


def prepare_config(path: str | Path) -> dict:
    cfg = load_config(path)
    cfg = copy.deepcopy(cfg)
    cfg["app"]["headless"] = False
    cfg["environment"]["num_parallel_envs"] = 1
    cfg["environment"]["crazyflie"]["spawn_randomization_m"] = [0.0, 0.0, 0.0]
    cfg["scenario_preview"]["enabled"] = True
    cfg["data"]["execution_mode"] = "integrity"
    cfg["data"]["integrity"]["save_raw_data"] = False
    cfg["data"]["integrity"]["save_frames"] = False
    return cfg


def main() -> int:
    initial = read_json(SCENARIO_COMMAND, {}) or {}
    config_path = initial.get("config_path") or str(load_config()["_config_path"])
    last_revision = int(initial.get("revision", 0))
    cfg = prepare_config(config_path)

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": False,
            "width": int(cfg["app"]["width"]),
            "height": int(cfg["app"]["height"]),
            "renderer": str(cfg["app"]["renderer"]),
        }
    )
    evaluator = None
    try:
        from cfbrkga.isaac_backend import IsaacBatchEvaluator

        evaluator = IsaacBatchEvaluator(app, cfg, Path(PROJECT_ROOT))
        evaluator.prepare_preview()
        write_json(SCENARIO_STATUS, {"state": "running", "config_path": str(config_path), "revision": last_revision})
        print("Scenario preview is running. Use Update scenario in the GUI to rebuild it.", flush=True)
        next_check = 0.0
        while app.is_running():
            app.update()
            now = time.monotonic()
            if now < next_check:
                continue
            next_check = now + float(cfg["scenario_preview"].get("refresh_interval_s", 0.25))
            command = read_json(SCENARIO_COMMAND, {}) or {}
            revision = int(command.get("revision", 0))
            if revision <= last_revision:
                continue
            last_revision = revision
            action = str(command.get("action", "reload"))
            if action == "stop":
                break
            config_path = command.get("config_path") or config_path
            print(f"Reloading scenario from {config_path}", flush=True)
            if evaluator is not None:
                evaluator.close()
                evaluator = None
            gc.collect()
            cfg = prepare_config(config_path)
            evaluator = IsaacBatchEvaluator(app, cfg, Path(PROJECT_ROOT))
            evaluator.prepare_preview()
            write_json(SCENARIO_STATUS, {"state": "running", "config_path": str(config_path), "revision": last_revision})
        write_json(SCENARIO_STATUS, {"state": "finished", "config_path": str(config_path), "revision": last_revision})
        return 0
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        write_json(SCENARIO_STATUS, {"state": "failed", "config_path": str(config_path), "revision": last_revision, "error": error})
        return 1
    finally:
        if evaluator is not None:
            evaluator.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
