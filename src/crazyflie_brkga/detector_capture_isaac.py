from __future__ import annotations

import copy
import gc
import time
import traceback
from pathlib import Path

import cv2

from cfbrkga.config import PROJECT_ROOT, load_config
from cfbrkga.runtime_files import DETECTOR_COMMAND, DETECTOR_FRAME, DETECTOR_STATUS, read_json, write_json


def prepare_config(path: str | Path) -> dict:
    cfg = copy.deepcopy(load_config(path))
    cfg["app"]["headless"] = False
    cfg["environment"]["num_parallel_envs"] = 1
    cfg["environment"]["crazyflie"]["spawn_randomization_m"] = [0.0, 0.0, 0.0]
    cfg["scenario_preview"]["enabled"] = True
    cfg["data"]["execution_mode"] = "integrity"
    cfg["data"]["integrity"]["save_raw_data"] = False
    cfg["data"]["integrity"]["save_frames"] = False
    return cfg


def atomic_write_rgb_png(path: Path, rgb) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(temporary), bgr):
        raise RuntimeError(f"Could not write detector frame: {temporary}")
    temporary.replace(path)


def main() -> int:
    initial = read_json(DETECTOR_COMMAND, {}) or {}
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
    frame_index = 0
    try:
        from cfbrkga.isaac_backend import IsaacBatchEvaluator

        evaluator = IsaacBatchEvaluator(app, cfg, Path(PROJECT_ROOT))
        evaluator.prepare_preview()
        write_json(DETECTOR_STATUS, {"state": "running", "config_path": str(config_path), "revision": last_revision, "frame_index": frame_index})
        next_capture = 0.0
        next_command_check = 0.0
        while app.is_running():
            app.update()
            now = time.monotonic()
            if now >= next_command_check:
                next_command_check = now + 0.2
                command = read_json(DETECTOR_COMMAND, {}) or {}
                revision = int(command.get("revision", 0))
                if revision > last_revision:
                    last_revision = revision
                    action = str(command.get("action", "reload"))
                    if action == "stop":
                        break
                    config_path = command.get("config_path") or config_path
                    if evaluator is not None:
                        evaluator.close()
                        evaluator = None
                    gc.collect()
                    cfg = prepare_config(config_path)
                    evaluator = IsaacBatchEvaluator(app, cfg, Path(PROJECT_ROOT))
                    evaluator.prepare_preview()
            if now < next_capture:
                continue
            next_capture = now + 0.10
            if evaluator is None:
                continue
            frame = evaluator.capture_preview_frame(0, updates=1)
            if frame is None:
                continue
            atomic_write_rgb_png(DETECTOR_FRAME, frame)
            frame_index += 1
            write_json(DETECTOR_STATUS, {"state": "running", "config_path": str(config_path), "revision": last_revision, "frame_index": frame_index, "updated_at_unix": time.time()})
        write_json(DETECTOR_STATUS, {"state": "finished", "config_path": str(config_path), "revision": last_revision, "frame_index": frame_index})
        return 0
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        write_json(DETECTOR_STATUS, {"state": "failed", "config_path": str(config_path), "revision": last_revision, "frame_index": frame_index, "error": error})
        return 1
    finally:
        if evaluator is not None:
            evaluator.close()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
