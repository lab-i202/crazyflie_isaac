from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeVersions:
    python: str
    numpy: str
    opencv: str
    torch: str
    cuda_available: bool
    cuda_device: str


def _major(version: str) -> int:
    text = str(version).strip()
    first = text.split(".", 1)[0]
    digits = "".join(character for character in first if character.isdigit())
    if not digits:
        return 0
    return int(digits)


def collect_runtime_versions() -> RuntimeVersions:
    import platform

    import cv2
    import numpy as np
    import torch

    cuda_available = bool(torch.cuda.is_available())
    cuda_device = ""
    if cuda_available:
        try:
            cuda_device = str(torch.cuda.get_device_name(0))
        except Exception:
            cuda_device = "CUDA device available"

    return RuntimeVersions(
        python=platform.python_version(),
        numpy=str(np.__version__),
        opencv=str(cv2.__version__),
        torch=str(torch.__version__),
        cuda_available=cuda_available,
        cuda_device=cuda_device,
    )


def validate_isaac_runtime(require_cuda: bool = True) -> RuntimeVersions:
    """Validate dependencies that must remain ABI-compatible with Isaac Sim 5.1.

    Isaac Sim 5.1 synthetic-data and camera bindings are built against NumPy 1.x.
    Installing an unconstrained modern package can upgrade NumPy to 2.x. When that
    happens, Replicator camera attachment fails with:

        TypeError: Unable to write from unknown dtype, kind=f, size=0

    The check runs before SimulationApp is started so the user gets an immediate,
    useful error instead of a black window followed by shutdown.
    """

    versions = collect_runtime_versions()

    errors: list[str] = []
    if _major(versions.numpy) >= 2:
        errors.append(
            "NumPy 2.x is installed. Isaac Sim 5.1 camera/Replicator bindings require NumPy 1.x."
        )
    if require_cuda and not versions.cuda_available:
        errors.append("PyTorch cannot access CUDA in the Isaac Lab environment.")

    if errors:
        details = "\n".join(f"- {item}" for item in errors)
        raise RuntimeError(
            "Isaac runtime compatibility check failed:\n"
            f"{details}\n\n"
            "Detected versions:\n"
            f"  Python: {versions.python}\n"
            f"  NumPy: {versions.numpy}\n"
            f"  OpenCV: {versions.opencv}\n"
            f"  PyTorch: {versions.torch}\n"
            f"  CUDA available: {versions.cuda_available}\n"
            f"  CUDA device: {versions.cuda_device or 'none'}\n\n"
            "Run repair_isaac_environment.bat once, then rerun the integrity test."
        )

    return versions


def format_runtime_versions(versions: RuntimeVersions) -> str:
    return (
        f"Python={versions.python} | NumPy={versions.numpy} | OpenCV={versions.opencv} | "
        f"PyTorch={versions.torch} | CUDA={versions.cuda_available} | "
        f"Device={versions.cuda_device or 'none'}"
    )
