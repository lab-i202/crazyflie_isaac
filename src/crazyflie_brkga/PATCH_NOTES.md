# Camera startup fix

## Root cause

The uploaded traceback fails inside `annotator.attach()` with:

```text
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

The project's old `requirements-core.txt` allowed any NumPy version greater than
1.26. That permitted pip to install NumPy 2.x. Isaac Sim 5.1 SyntheticData camera
bindings are compiled for NumPy 1.x and fail when Replicator builds the RGB camera
graph under NumPy 2.x.

## Apply

Replace the existing `crazyflie_isaac_brkga` folder with this folder, or copy these
changed files:

- `cfbrkga/runtime_compat.py`
- `cfbrkga/isaac_backend.py`
- `train_isaac.py`
- `verify_isaac_environment.py`
- `requirements-core.txt`
- `repair_isaac_environment.bat`
- `run_isaac_integrity.bat`
- `run_training.bat`

## Run once

```powershell
repair_isaac_environment.bat
```

Expected versions:

```text
NumPy: 1.26.4
OpenCV: 4.11.0
```

## Retest

```powershell
run_isaac_integrity.bat
```

The expected startup sequence now includes:

```text
Checking Isaac Python environment...
Python=... | NumPy=1.26.4 | OpenCV=4.11.0 | ... | CUDA=True
Starting Isaac SimulationApp...
Constructing IsaacBatchEvaluator...
Isaac stage ready: 1 independent rooms, asset=...
IsaacBatchEvaluator constructed successfully.
Starting BRKGA integrity/training run...
[generation 0000] ...
Isaac BRKGA training finished: ...
```
