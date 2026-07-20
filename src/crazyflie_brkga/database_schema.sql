PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_info (
    schema_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    backend TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    config_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    host TEXT,
    process_id INTEGER,
    current_generation INTEGER DEFAULT 0,
    note TEXT
);

CREATE TABLE IF NOT EXISTS arrays (
    array_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    dtype TEXT NOT NULL,
    shape_json TEXT NOT NULL,
    compression TEXT NOT NULL,
    bytes_raw INTEGER NOT NULL,
    bytes_stored INTEGER NOT NULL,
    data_blob BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    best_fitness REAL,
    mean_fitness REAL,
    median_fitness REAL,
    worst_fitness REAL,
    std_fitness REAL,
    success_rate REAL,
    collision_rate REAL,
    UNIQUE(run_id, generation_index)
);

CREATE TABLE IF NOT EXISTS individuals (
    individual_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    index_in_generation INTEGER NOT NULL,
    origin TEXT NOT NULL,
    elite_parent_id INTEGER,
    non_elite_parent_id INTEGER,
    chromosome_array_id INTEGER NOT NULL REFERENCES arrays(array_id),
    weights_array_id INTEGER NOT NULL REFERENCES arrays(array_id),
    fitness REAL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(run_id, individual_id),
    UNIQUE(run_id, generation_index, index_in_generation)
);

CREATE TABLE IF NOT EXISTS environment_slots (
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    env_index INTEGER NOT NULL,
    current_individual_id INTEGER,
    current_evaluation_id INTEGER,
    state TEXT NOT NULL,
    state_since TEXT NOT NULL,
    last_heartbeat TEXT NOT NULL,
    last_camera_frame INTEGER,
    last_simulation_step INTEGER,
    message TEXT,
    PRIMARY KEY(run_id, env_index)
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id) ON DELETE CASCADE,
    individual_id INTEGER NOT NULL,
    env_index INTEGER NOT NULL,
    episode_seed INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    reward REAL,
    success INTEGER,
    reason TEXT,
    steps INTEGER,
    simulated_time_s REAL,
    wall_time_s REAL,
    collision_count INTEGER,
    visible_steps INTEGER,
    invalid_camera_frames INTEGER,
    mean_euclid_norm REAL,
    min_front_tof_m REAL,
    final_front_tof_m REAL,
    final_euclid_norm REAL,
    debug_dir TEXT,
    extra_json TEXT
);

CREATE TABLE IF NOT EXISTS agent_state_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluation_id INTEGER,
    individual_id INTEGER,
    env_index INTEGER NOT NULL,
    previous_state TEXT,
    new_state TEXT NOT NULL,
    reason TEXT,
    simulation_step INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_steps (
    step_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    simulation_time_s REAL NOT NULL,
    camera_frame_index INTEGER,
    action INTEGER NOT NULL,
    reward REAL NOT NULL,
    front_tof_m REAL NOT NULL,
    euclid_norm REAL NOT NULL,
    visible INTEGER NOT NULL,
    blob_x REAL,
    blob_y REAL,
    x_m REAL NOT NULL,
    y_m REAL NOT NULL,
    z_m REAL NOT NULL,
    yaw_rad REAL NOT NULL,
    collision INTEGER NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER,
    individual_id INTEGER,
    evaluation_id INTEGER,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER,
    individual_id INTEGER,
    env_index INTEGER,
    component TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    generation_index INTEGER NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_individuals_generation ON individuals(run_id, generation_index);
CREATE INDEX IF NOT EXISTS idx_evaluations_generation ON evaluations(run_id, generation_id);
CREATE INDEX IF NOT EXISTS idx_events_run_env ON agent_state_events(run_id, env_index, created_at);
