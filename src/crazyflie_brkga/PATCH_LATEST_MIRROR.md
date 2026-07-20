# Stable latest-run mirror and state-event schema update

## Decision

The feature is implemented in the shared training core, not in an integrity-only script.
The integrity configuration is the validation path. After it passes, the same behavior is
used by normal Isaac training and mock runs.

## Stable database path

Every run still has its authoritative timestamped folder:

```text
outputs/experiments/20260720_160139_884266_crazyflie_isaac_integrity/
```

A stable mirror is maintained at:

```text
outputs/experiments/last_crazyflie_isaac_integrity/
```

The stable folder contains:

```text
run.sqlite
resolved_config.json
run_metadata.json
source_experiment.json
SOURCE_EXPERIMENT.txt
csv/
checkpoints/
```

The SQLite file is refreshed with SQLite's backup API, which produces a consistent copy
even though the authoritative database uses WAL mode. This allows a permanent DBeaver
connection to the stable path.

Raw integrity images are intentionally not duplicated. Their absolute paths remain stored
in SQLite and the dashboard can still open them from the authoritative timestamped folder.

## Configuration

```json
"latest_mirror": {
  "enabled": true,
  "folder_name": "",
  "database_sync_interval_s": 2.0,
  "copy_csv": true,
  "copy_checkpoints": true
}
```

An empty `folder_name` produces `last_<project.name>`.

## Database schema

`agent_state_events` now contains:

```text
previous_state
new_state
current_state
```

For each transition, `current_state` is the state after the transition and therefore equals
`new_state`. The redundant column is deliberate: it makes the event table easier to inspect
in DBeaver. The actual live state remains in `environment_slots.state`.

Existing databases are migrated automatically to schema version 2. Existing rows receive
`current_state = new_state`.

## Dashboard

The dashboard sidebar now provides an experiment database dropdown. Stable `last_*` mirrors
are listed first and labeled `CURRENT`. A manual database path remains available.

## Validation

Run:

```powershell
python -m compileall -q .
pytest -q
```

Then run the Isaac integrity configuration and confirm that both the timestamped folder and
the stable `last_<project>` folder are updated.
