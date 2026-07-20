from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path

from cfbrkga.config import PROJECT_ROOT, load_config
from cfbrkga.runtime_compat import format_runtime_versions, validate_isaac_runtime


ERROR_LOG_PATH = Path(PROJECT_ROOT) / "outputs" / "isaac_startup_error.txt"


def write_error_log(error_text: str) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write("=" * 100)
        handle.write("\n")
        handle.write(f"Timestamp: {timestamp}\n")
        handle.write(error_text)
        if not error_text.endswith("\n"):
            handle.write("\n")


def main() -> int:
    config = load_config()
    app_config = config["app"]

    print(f"Configuration: {config['_config_path']}", flush=True)
    print("Checking Isaac Python environment...", flush=True)

    try:
        versions = validate_isaac_runtime(require_cuda=True)
    except Exception:
        error_text = traceback.format_exc()
        print("\nISAAC BRKGA FAILED BEFORE SIMULATION START\n", flush=True)
        print(error_text, flush=True)
        write_error_log(error_text)
        print(f"Full traceback saved to: {ERROR_LOG_PATH}", flush=True)
        return 1

    print(format_runtime_versions(versions), flush=True)
    print("Starting Isaac SimulationApp...", flush=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(app_config["headless"]),
            "width": int(app_config["width"]),
            "height": int(app_config["height"]),
            "renderer": str(app_config.get("renderer", "RayTracedLighting")),
        }
    )

    evaluator = None
    exit_code = 0

    try:
        print("Importing Isaac backend...", flush=True)
        from cfbrkga.isaac_backend import IsaacBatchEvaluator
        from cfbrkga.training_loop import TrainingCoordinator

        print("Constructing IsaacBatchEvaluator...", flush=True)
        evaluator = IsaacBatchEvaluator(
            simulation_app=simulation_app,
            cfg=config,
            project_root=Path(PROJECT_ROOT),
        )

        print("IsaacBatchEvaluator constructed successfully.", flush=True)
        print("Constructing TrainingCoordinator...", flush=True)
        coordinator = TrainingCoordinator(config, evaluator)

        print("Starting BRKGA integrity/training run...", flush=True)
        output_dir = coordinator.run()
        print(f"Isaac BRKGA training finished: {output_dir}", flush=True)

    except BaseException:
        exit_code = 1
        error_text = traceback.format_exc()
        print("\nISAAC BRKGA FAILED\n", flush=True)
        print(error_text, flush=True)
        try:
            write_error_log(error_text)
            print(f"Full traceback saved to: {ERROR_LOG_PATH}", flush=True)
        except Exception:
            print("Failed to write the additional error log:", flush=True)
            traceback.print_exc()

    finally:
        print("Cleaning up Isaac resources...", flush=True)
        if evaluator is not None:
            try:
                evaluator.close()
            except Exception:
                print("Evaluator cleanup failed:", flush=True)
                traceback.print_exc()

        try:
            simulation_app.close()
        except Exception:
            print("SimulationApp cleanup failed:", flush=True)
            traceback.print_exc()
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
