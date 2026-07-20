from __future__ import annotations

import os
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class ProcessState:
    name: str
    process: subprocess.Popen[str]
    command: list[str]


class ProcessManager:
    def __init__(self, project_root: Path, on_output: Callable[[str], None], on_state: Callable[[str], None]) -> None:
        self.project_root = project_root
        self.on_output = on_output
        self.on_state = on_state
        self.processes: dict[str, ProcessState] = {}
        self.output_queue: queue.Queue[str] = queue.Queue()

    def is_running(self, name: str) -> bool:
        state = self.processes.get(name)
        return state is not None and state.process.poll() is None

    def start(self, name: str, command: list[str], env: dict[str, str] | None = None) -> int:
        if self.is_running(name):
            raise RuntimeError(f"{name} is already running.")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=creationflags,
        )
        self.processes[name] = ProcessState(name=name, process=process, command=command)
        self.on_state(f"{name}: running (PID {process.pid})")
        thread = threading.Thread(target=self._read_output, args=(name, process), daemon=True)
        thread.start()
        return process.pid

    def _read_output(self, name: str, process: subprocess.Popen[str]) -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    self.output_queue.put(f"[{name}] {line.rstrip()}\n")
        finally:
            exit_code = process.wait()
            self.output_queue.put(f"[{name}] process finished with exit code {exit_code}.\n")
            self.output_queue.put(f"__STATE__{name}:{exit_code}")

    def drain(self) -> None:
        while True:
            try:
                message = self.output_queue.get_nowait()
            except queue.Empty:
                break
            if message.startswith("__STATE__"):
                payload = message[len("__STATE__") :]
                name, code = payload.rsplit(":", 1)
                self.on_state(f"{name}: finished (exit {code})")
                continue
            self.on_output(message)

    def terminate(self, name: str) -> None:
        state = self.processes.get(name)
        if state is None or state.process.poll() is not None:
            return
        try:
            state.process.terminate()
        except Exception:
            state.process.kill()

    def terminate_all(self) -> None:
        for name in list(self.processes):
            self.terminate(name)
