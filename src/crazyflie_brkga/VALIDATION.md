# Validation performed in the build environment

Date: 2026-07-17

## Passed

- `python -m compileall -q .`
- `pytest -q`
- 6 unit/integration tests passed.
- Fixed network genome length verified as 17,541.
- Random-key decoder and batched network inference tested.
- BRKGA population evolution tested.
- Luminous RGB blob detection tested.
- Baseline reward success behavior tested.
- SQLite NumPy BLOB round trip tested.
- End-to-end mock training completed with:
  - 4 individuals;
  - 1 generation;
  - 2 evaluation slots;
  - 20 steps per evaluation;
  - camera frames and detection overlays;
  - 80 SQLite step records;
  - 4 evaluation records;
  - 4 integrity artifacts;
  - 1 checkpoint;
  - 0 database error records.
- CSV exports, resolved config, run metadata, and checkpoint files were verified.

## Not executable in this environment

Isaac Sim, Isaac Lab, Omniverse Replicator, and an NVIDIA GPU are not installed in
the build environment. The Isaac backend was syntax-checked but not runtime-tested.
Use `run_isaac_integrity.bat` before normal training.

## Isaac 5.1 Replicator compatibility fix

The camera startup error `Unable to write from unknown dtype, kind=f, size=0` is an
ABI mismatch caused by NumPy 2.x in an Isaac Sim 5.1 environment. The project now:

- pins `numpy==1.26.4`;
- pins `opencv-python==4.11.0.86` so OpenCV does not upgrade NumPy to 2.x;
- validates the runtime before SimulationApp starts;
- includes `repair_isaac_environment.bat`;
- produces a direct actionable error if RGB annotator attachment still fails.
