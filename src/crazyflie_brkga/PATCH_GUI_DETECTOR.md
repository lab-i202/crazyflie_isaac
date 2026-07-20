# GUI and landmark-detector patch

This patch adds the unified scenario/training GUI and the separate landmark detector configurator.

## Changed shared core

- `cfbrkga/config.py`
- `cfbrkga/image_processing.py`
- `cfbrkga/isaac_backend.py`
- `cfbrkga/geometry.py`
- `cfbrkga/gui.py`

## New modules

- `cfbrkga/gui_widgets.py`
- `cfbrkga/scenario_builder.py`
- `cfbrkga/process_manager.py`
- `cfbrkga/runtime_files.py`
- `cfbrkga/detector_gui.py`

## New launchers

- `scenario_preview_isaac.py`
- `detector_capture_isaac.py`
- `landmark_detector_configurator.py`
- `run_scenario_preview.bat`
- `run_detector_capture.bat`
- `run_detector_configurator.bat`

## Updated configuration and documentation

- `config/default.json`
- `config/isaac_integrity.json`
- `config/smoke.json`
- `GUI_AND_DETECTOR.md`
- `README.md`
- `VALIDATION.md`
- `PROJECT_TREE.txt`

Extract the patch over the existing `crazyflie_brkga` folder and allow file replacement. Then run:

```powershell
python -m compileall -q .
pytest -q
.\run_gui.bat
```
