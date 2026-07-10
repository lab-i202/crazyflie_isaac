# scenario/scenario_registry.py
#
# Scenario registry for kite-isaac.
#
# This module is intentionally lightweight and does not import Isaac Sim.
# It only stores scenario metadata and lazy import paths.

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str
    description: str
    default_profile_relative_path: str
    validator_symbol: str
    runner_symbol: str
    editable_in_launcher: bool
    tabs: tuple[str, ...]

    def default_profile_path(self, project_root: Path) -> Path:
        return project_root / self.default_profile_relative_path


SCENARIOS: dict[str, ScenarioSpec] = {
    "crazyflie_room_basic": ScenarioSpec(
        scenario_id="crazyflie_room_basic",
        label="Crazyflie / room scenario",
        description=(
            "One Crazyflie inside a configurable room, with obstacles, "
            "one luminous landmark, onboard camera, isometric camera, "
            "and telemetry output."
        ),
        default_profile_relative_path="profiles/crazyflie/crazyflie_room_basic.json",
        validator_symbol="utils.crazyflie_profile_io:validate_crazyflie_room_profile",
        runner_symbol="scenario.crazyflie_room_scene:run_crazyflie_room_scene",
        editable_in_launcher=True,
        tabs=("Room", "Obstacles", "Landmark", "Crazyflie", "Cameras", "Runtime"),
    ),
    "tethered_glider_basic": ScenarioSpec(
        scenario_id="tethered_glider_basic",
        label="AWES / Bixler tethered glider - legacy",
        description=(
            "Existing tethered-glider prototype. Kept registered so the new "
            "launcher can later host it, but the current rich editor remains "
            "the old glider-specific GUI."
        ),
        default_profile_relative_path="profiles/tethered_glider_basic.json",
        validator_symbol="utils.profile_io:validate_tethered_glider_profile",
        runner_symbol="scenario.tethered_glider_scene:run_tethered_glider_scene",
        editable_in_launcher=False,
        tabs=("Legacy profile",),
    ),
}


def list_scenarios() -> list[ScenarioSpec]:
    return list(SCENARIOS.values())


def get_scenario_spec(scenario_id: str) -> ScenarioSpec:
    try:
        return SCENARIOS[scenario_id]
    except KeyError as exc:
        valid = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario_id '{scenario_id}'. Valid values: {valid}") from exc


def import_symbol(symbol_path: str) -> Callable[..., Any]:
    if ":" not in symbol_path:
        raise ValueError(
            f"Invalid symbol path '{symbol_path}'. Expected format 'module.submodule:symbol'."
        )

    module_name, symbol_name = symbol_path.split(":", 1)
    module = import_module(module_name)

    try:
        symbol = getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(f"Could not import symbol '{symbol_name}' from '{module_name}'.") from exc

    if not callable(symbol):
        raise TypeError(f"Imported symbol is not callable: {symbol_path}")

    return symbol
