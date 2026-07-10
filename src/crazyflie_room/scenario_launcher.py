# scenario_launcher.py
#
# New entry point for modular scenario selection.
#
# Run from src/kite-isaac with Isaac's Python, for example:
#   C:\isaac-lab\IsaacLab\isaaclab.bat -p scenario_launcher.py
#
# This file intentionally imports Isaac Sim only after the Tkinter launcher closes.

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from gui.scenario_launcher_gui import run_scenario_launcher_gui
from scenario.scenario_registry import get_scenario_spec, import_symbol


PROJECT_ROOT = Path(__file__).parent.resolve()

DEFAULT_ISAAC_SIM_CONFIG: dict[str, Any] = {
    "headless": False,
    "renderer": "RealTimePathTracing",
    "width": 1280,
    "height": 720,
    "anti_aliasing": 3,
    "sync_loads": True,
}


def build_simulation_app_config(profile: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_ISAAC_SIM_CONFIG)
    app_config = profile.get("app", {})
    if isinstance(app_config, dict):
        config.update(app_config)
    return config


def call_validator(validator: Any, profile: dict[str, Any]) -> None:
    signature = inspect.signature(validator)
    if "project_root" in signature.parameters:
        validator(profile, project_root=PROJECT_ROOT)
    else:
        validator(profile)


def call_runner(
    runner: Any,
    simulation_app: Any,
    profile: dict[str, Any],
    profile_path: Path,
) -> None:
    signature = inspect.signature(runner)
    kwargs: dict[str, Any] = {
        "simulation_app": simulation_app,
        "scene_config": profile,
    }
    if "project_root" in signature.parameters:
        kwargs["project_root"] = PROJECT_ROOT
    if "profile_path" in signature.parameters:
        kwargs["profile_path"] = profile_path
    runner(**kwargs)


def main() -> int:
    gui_result = run_scenario_launcher_gui(project_root=PROJECT_ROOT)

    if gui_result is None:
        print("Cancelled. Isaac Sim was not started.")
        return 0

    if not gui_result.get("run_requested", False):
        print("No run requested. Isaac Sim was not started.")
        return 0

    scenario_id = str(gui_result["scenario_id"])
    profile = gui_result["profile"]
    profile_path = Path(gui_result["profile_path"]).resolve()

    spec = get_scenario_spec(scenario_id)
    validator = import_symbol(spec.validator_symbol)
    call_validator(validator, profile)

    simulation_config = build_simulation_app_config(profile)

    print("=" * 100)
    print("Starting Isaac Sim from scenario launcher")
    print("=" * 100)
    print(f"Scenario: {spec.label}")
    print(f"Scenario ID: {scenario_id}")
    print(f"Profile path: {profile_path}")
    print(f"Scene name: {profile.get('scene_name', '<unnamed>')}")
    print(f"SimulationApp config: {simulation_config}")
    print("=" * 100)

    # Isaac Sim must be imported only after the normal-Python GUI closes.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(simulation_config)

    try:
        runner = import_symbol(spec.runner_symbol)
        call_runner(
            runner=runner,
            simulation_app=simulation_app,
            profile=profile,
            profile_path=profile_path,
        )
    finally:
        simulation_app.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
