from __future__ import annotations

import io
import json
import os
import sqlite3
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Crazyflie BRKGA Dashboard", layout="wide")
st.title("Crazyflie Isaac BRKGA")

project_root = Path(__file__).resolve().parents[1]
default_db = os.environ.get("CRAZYFLIE_BRKGA_DB", "")
if not default_db:
    candidates = sorted((project_root / "outputs" / "experiments").glob("*/run.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
    default_db = str(candidates[0]) if candidates else ""
db_text = st.sidebar.text_input("SQLite database", value=default_db)
refresh = st.sidebar.button("Refresh")
if not db_text or not Path(db_text).exists():
    st.warning("Select an existing run.sqlite file.")
    st.stop()

connection = sqlite3.connect(f"file:{Path(db_text).resolve()}?mode=ro", uri=True)
connection.row_factory = sqlite3.Row


def frame(sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, connection, params=params)

runs = frame("SELECT * FROM runs ORDER BY run_id DESC")
if runs.empty:
    st.error("Database has no runs.")
    st.stop()
run_id = int(st.sidebar.selectbox("Run", runs["run_id"].tolist()))
run = runs[runs["run_id"] == run_id].iloc[0]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", run["status"])
col2.metric("Backend", run["backend"])
col3.metric("Current generation", int(run["current_generation"]))
col4.metric("Mode", run["mode"])
st.caption(f"Database: {Path(db_text).resolve()}")

tabs = st.tabs(["Overview", "Population", "Evaluations", "Environment slots", "Problems", "Artifacts", "Configuration"])

with tabs[0]:
    generations = frame("SELECT * FROM generations WHERE run_id=? ORDER BY generation_index", (run_id,))
    if not generations.empty:
        st.subheader("Fitness history")
        chart = generations.set_index("generation_index")[["best_fitness", "mean_fitness", "median_fitness", "worst_fitness"]]
        st.line_chart(chart)
        left, right = st.columns(2)
        left.dataframe(generations, use_container_width=True, hide_index=True)
        rates = generations.set_index("generation_index")[["success_rate", "collision_rate"]]
        right.line_chart(rates)
    else:
        st.info("No completed generation yet.")

with tabs[1]:
    generation_values = frame("SELECT DISTINCT generation_index FROM individuals WHERE run_id=? ORDER BY generation_index DESC", (run_id,))
    if not generation_values.empty:
        selected_generation = int(st.selectbox("Generation", generation_values["generation_index"].tolist()))
        population = frame(
            """SELECT individual_id, index_in_generation, origin, elite_parent_id, non_elite_parent_id,
               fitness, status, chromosome_array_id, weights_array_id FROM individuals
               WHERE run_id=? AND generation_index=? ORDER BY fitness DESC""",
            (run_id, selected_generation),
        )
        st.dataframe(population, use_container_width=True, hide_index=True)
        if not population.empty:
            st.bar_chart(population.set_index("individual_id")[["fitness"]])
            individual_id = int(st.selectbox("Inspect individual", population["individual_id"].tolist()))
            selected_row = population[population["individual_id"] == individual_id].iloc[0]
            array_rows = frame(
                """SELECT array_id, kind, dtype, shape_json, bytes_raw, bytes_stored, data_blob
                   FROM arrays WHERE array_id IN (?, ?) ORDER BY kind""",
                (int(selected_row["chromosome_array_id"]), int(selected_row["weights_array_id"])),
            )
            decoded_arrays = {}
            summaries = []
            for _, array_row in array_rows.iterrows():
                raw = zlib.decompress(bytes(array_row["data_blob"]))
                value = np.load(io.BytesIO(raw), allow_pickle=False)
                decoded_arrays[str(array_row["kind"])] = value
                summaries.append({
                    "kind": array_row["kind"],
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "minimum": float(np.min(value)),
                    "maximum": float(np.max(value)),
                    "mean": float(np.mean(value)),
                    "std": float(np.std(value)),
                    "bytes_raw": int(array_row["bytes_raw"]),
                    "bytes_stored": int(array_row["bytes_stored"]),
                })
            st.dataframe(pd.DataFrame(summaries), use_container_width=True, hide_index=True)
            if decoded_arrays:
                buffer = io.BytesIO()
                np.savez_compressed(buffer, **decoded_arrays)
                st.download_button(
                    "Download chromosome and decoded weights (.npz)",
                    buffer.getvalue(),
                    file_name=f"individual_{individual_id:08d}.npz",
                    mime="application/octet-stream",
                )

with tabs[2]:
    evaluations = frame(
        """SELECT e.*, g.generation_index FROM evaluations e JOIN generations g
           ON g.generation_id=e.generation_id WHERE e.run_id=? ORDER BY e.evaluation_id DESC""",
        (run_id,),
    )
    st.dataframe(evaluations, use_container_width=True, hide_index=True)
    if not evaluations.empty:
        evaluation_id = int(st.selectbox("Evaluation detail", evaluations["evaluation_id"].tolist()))
        steps = frame("SELECT * FROM episode_steps WHERE evaluation_id=? ORDER BY step_index", (evaluation_id,))
        if not steps.empty:
            c1, c2 = st.columns(2)
            c1.line_chart(steps.set_index("step_index")[["front_tof_m", "euclid_norm"]])
            c2.line_chart(steps.set_index("step_index")[["reward"]])
            st.dataframe(steps, use_container_width=True, hide_index=True)
        else:
            st.caption("Per-step records exist only when integrity mode enables SQLite step storage.")

with tabs[3]:
    slots = frame("SELECT * FROM environment_slots WHERE run_id=? ORDER BY env_index", (run_id,))
    st.dataframe(slots, use_container_width=True, hide_index=True)
    events = frame("SELECT * FROM agent_state_events WHERE run_id=? ORDER BY event_id DESC LIMIT 500", (run_id,))
    st.subheader("Recent state transitions")
    st.dataframe(events, use_container_width=True, hide_index=True)

with tabs[4]:
    errors = frame("SELECT * FROM errors WHERE run_id=? ORDER BY error_id DESC", (run_id,))
    if errors.empty:
        st.success("No recorded errors.")
    else:
        st.dataframe(errors, use_container_width=True, hide_index=True)

with tabs[5]:
    artifacts = frame("SELECT * FROM artifacts WHERE run_id=? ORDER BY artifact_id DESC", (run_id,))
    st.dataframe(artifacts, use_container_width=True, hide_index=True)
    image_paths = []
    for path in artifacts.get("path", []):
        candidate = Path(path) / "camera" / "last_frame.png"
        if candidate.exists():
            image_paths.append(candidate)
    if image_paths:
        selected = Path(st.selectbox("Integrity camera image", [str(path) for path in image_paths]))
        left, right = st.columns(2)
        left.image(str(selected), caption="Last RGB frame", use_container_width=True)
        detection = selected.with_name("last_detection.png")
        if detection.exists():
            right.image(str(detection), caption="Last blob detection", use_container_width=True)

with tabs[6]:
    config = json.loads(run["config_json"])
    st.json(config)

connection.close()
