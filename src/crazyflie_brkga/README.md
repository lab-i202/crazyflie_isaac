# Crazyflie Isaac BRKGA

This folder contains the first end-to-end training core for the luminous-landmark
mission. It is designed to be copied into the repository as:

```text
<repository>/crazyflie_isaac_brkga/
```

The project keeps the executable behavior established by the previous Gazebo and
BRKGA repositories:

```text
Observation: [front_tof_m, euclid_norm]
Network:     2 -> 128 -> 128 -> 5
Actions:     forward, yaw left, yaw right, up, down
Genome:      17,541 random keys
Decoder:     linear [0,1] -> [-2,10] by default
Training:    direct BRKGA evolution of all weights and biases
```

There is no replay buffer, target network, Adam optimizer, or backpropagation in
this project. The network structure is retained, but BRKGA directly evolves its
parameters.

## What is included

- One Isaac Sim process with several independent room instances.
- One Crazyflie, luminous landmark, onboard RGB camera, and sensor set per room.
- Mandatory in-memory camera processing at every control iteration.
- HSV/morphology/contour luminous-blob detector.
- Euclidean image-center error compatible with the previous executable baseline.
- BRKGA population, elites, biased crossover, mutants, and generation checkpoints.
- SQLite database in WAL mode as the authoritative experiment record.
- Compressed chromosomes and decoded weights stored as database BLOBs.
- CSV exports regenerated from SQLite after every generation.
- Agent/environment-slot state and heartbeat tracking.
- Integrity mode for frames, camera overlays, step CSV, and optional SQLite steps.
- Normal training mode without frame archives.
- Streamlit dashboard for generations, individuals, weights, evaluations, slots,
  errors, and integrity artifacts.
- GUI for environment, camera, detector, BRKGA, reward, data, obstacles, and custom
  USD asset configuration.
- Custom USD asset discovery under `assets/`.
- Software-only mock evaluator that exercises the complete non-Isaac pipeline.

## Important limitation

The complete non-Isaac path is tested in this package. Isaac Sim is not installed in
the build environment used to create it, so the Isaac backend could not be executed
here. Do not start a long experiment before completing the supplied one-room Isaac
integrity test.

The first action adapter is kinematic, matching the current Crazyflie Isaac skeleton.
It moves the parent Crazyflie rig and preserves the original five action meanings.
The camera, BRKGA, persistence, dashboard, and reward modules are isolated from the
action adapter so a physical motor/thrust controller can replace it later.

This is research software. It is not validated for autonomous delivery of medical
supplies or any patient-facing deployment. Simulation success is not evidence of
real-flight safety.

## Files

```text
crazyflie_isaac_brkga/
├── main.py                       # GUI
├── train_mock.py                 # software-only end-to-end test
├── train_isaac.py                # Isaac Sim training entry point
├── config/
│   ├── default.json
│   ├── smoke.json
│   └── isaac_integrity.json
├── cfbrkga/
│   ├── brkga.py
│   ├── model.py
│   ├── image_processing.py
│   ├── reward.py
│   ├── geometry.py
│   ├── isaac_backend.py
│   ├── mock_evaluator.py
│   ├── database.py
│   ├── csv_export.py
│   ├── checkpoint.py
│   ├── debug_recorder.py
│   ├── training_loop.py
│   └── gui.py
├── dashboard/app.py
├── database_schema.sql
├── tests/
└── assets/
```

## 1. Install and run tests

Use the normal Python environment for the software-only tests:

```powershell
cd crazyflie_isaac_brkga
python -m pip install -r requirements-core.txt
pytest -q
```

Expected result:

```text
6 passed
```

## 2. Run the software-only integrity test

```powershell
run_mock_test.bat
```

or:

```powershell
set CRAZYFLIE_BRKGA_CONFIG=%CD%\config\smoke.json
python train_mock.py
```

This test generates synthetic onboard RGB frames, runs the real luminous-blob
detector, executes the fixed network, evolves one BRKGA generation, writes SQLite,
creates checkpoints, saves integrity images, and exports CSV.

## 3. Open the GUI

```powershell
run_gui.bat
```

The GUI does not use `argparse`. It saves JSON configuration files and launches the
mock test, Isaac training, dashboard, or output folder.

### Custom assets

Place `.usd`, `.usda`, or `.usdc` files anywhere under:

```text
assets/
```

Open the **Environment** tab, press **Rescan assets/**, choose an asset, and press
**Add selected**. Each custom asset entry contains:

```json
{
  "name": "room_model",
  "enabled": true,
  "usd_path": "assets/rooms/room_model.usd",
  "position_m": [0.0, 0.0, 0.0],
  "rotation_deg": [0.0, 0.0, 0.0],
  "scale": [1.0, 1.0, 1.0],
  "collision_aabb_size_m": null
}
```

Imported USD geometry is referenced into every replicated room. When
`collision_aabb_size_m` is provided, the deterministic fallback range/collision
model also includes that asset. The PhysX raycast backend detects authored collision
geometry directly.

## 4. First Isaac execution

Edit the Isaac Lab path in `run_isaac_integrity.bat` only when necessary. The current
path is:

```text
C:\isaac-lab\IsaacLab\isaaclab.bat
```

Then run:

```powershell
run_isaac_integrity.bat
```

The supplied integrity config uses:

```text
1 room
3 individuals
1 generation
20 control steps
camera and step retention enabled
```

Verify all of the following before increasing the workload:

1. The Crazyflie USD is found and visible.
2. The parent rig visibly moves when actions change.
3. The onboard camera sees the luminous landmark.
4. `last_frame.png` is a valid RGB frame.
5. `last_detection.png` shows the selected blob and image center.
6. `run.sqlite` contains three completed evaluations.
7. `errors` is empty.
8. The evaluation `extra_json` reports the requested range backend and fallback count.
9. The Isaac process closes cleanly.

If the built-in Crazyflie asset is unavailable, place it here:

```text
assets/robots/crazyflie/cf2x.usd
```

or set `environment.crazyflie.usd_path` in the GUI/config.

## 5. Normal training

After the integrity test passes:

```powershell
run_training.bat
```

The config path is passed through the `CRAZYFLIE_BRKGA_CONFIG` environment
variable. When the variable is absent, `config/default.json` is used.

Population size and parallel environment count are independent. For example, a
population of 100 with 8 rooms is evaluated in sequential batches of at most 8
individuals while each batch runs in parallel inside one Isaac process.

## Camera processing

Every room owns an onboard camera. The Isaac backend attaches an RGB Replicator
annotator to each camera and reads arrays directly in memory. No PNG is used as a
network input.

The detector performs:

```text
RGB -> HSV mask -> morphology -> contour filtering -> centroid -> euclid_norm
```

The policy receives exactly:

```text
[front_tof_m, euclid_norm]
```

Integrity mode can save images. Normal training does not create frame archives.
`last_frame.png` is updated on every processed integrity frame; `frame_stride`
controls only numbered `rgb_*.png` files.

## Range and collision data

`environment.sensors.range_backend` supports:

```text
physx_raycast
analytic
```

`physx_raycast` is the default Isaac setting. It performs six scene queries from the
Crazyflie pose. If the installed Isaac version changes the scene-query API or the
query fails, the evaluator falls back to the deterministic room/AABB model and
records `range_fallback_count` in the evaluation's `extra_json`.

The mock test uses the analytic backend.

## Database

Each experiment creates one database:

```text
outputs/experiments/<timestamp>_<project>/run.sqlite
```

SQLite WAL mode allows DBeaver and the Streamlit dashboard to read the database
while training writes to it. Environment slots do not write independently; the
training coordinator serializes database writes.

Main tables:

- `runs`
- `generations`
- `individuals`
- `arrays`
- `environment_slots`
- `evaluations`
- `agent_state_events`
- `episode_steps`
- `artifacts`
- `errors`
- `checkpoints`

`arrays` stores zlib-compressed NumPy BLOBs and SHA-256 identifiers. The dashboard
can inspect statistics and download an individual's chromosome and decoded weights
as `.npz`.

## CSV exports

CSV is derived from SQLite and is not a second source of truth:

```text
csv/generations.csv
csv/population_latest.csv
csv/best_individual_history.csv
csv/evaluations.csv
csv/agent_state_events.csv
csv/errors.csv
```

Exports are written through a temporary file and replaced atomically.

## Dashboard

Install dashboard requirements:

```powershell
python -m pip install -r requirements-dashboard.txt
run_dashboard.bat
```

The dashboard provides:

- fitness history;
- success and collision rates;
- population ranking and lineage;
- chromosome/weight statistics and `.npz` download;
- evaluation summaries and integrity step plots;
- environment-slot states and heartbeats;
- errors;
- RGB/detection integrity images;
- resolved configuration.

## Output layout

```text
outputs/experiments/<run>/
├── run.sqlite
├── resolved_config.json
├── run_metadata.json
├── checkpoints/
├── csv/
└── debug/                     # integrity mode only
```

## Current limits

- Isaac runtime still requires local one-room validation.
- Kinematic control is not a physical flight controller.
- Camera processing currently loops over camera arrays. After runtime stability is
  verified, this can be replaced by tiled camera acquisition and batched GPU blob
  processing without changing the database or BRKGA interfaces.
- Arbitrary USD geometry needs authored collision meshes for PhysX raycasts. The
  fallback model requires `collision_aabb_size_m`.
- Checkpoints are written, but resume controls are not exposed in the first GUI.

## Isaac Sim 5.1 camera dependency repair

The Isaac Sim 5.1 SyntheticData camera bindings fail when NumPy 2.x is installed.
The exact failure is:

```text
TypeError: Unable to write from unknown dtype, kind=f, size=0
```

This project now pins NumPy and OpenCV to a compatible pair. On the affected machine,
run this once:

```powershell
repair_isaac_environment.bat
```

Then run:

```powershell
run_isaac_integrity.bat
```

The integrity launcher now checks NumPy, OpenCV, PyTorch, CUDA availability, and the
CUDA device before opening Isaac Sim. It will refuse to start if NumPy 2.x is present.
