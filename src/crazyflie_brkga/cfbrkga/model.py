from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BASELINE_LAYERS = (2, 128, 128, 5)


@dataclass(frozen=True, slots=True)
class ParameterSlice:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


def build_parameter_layout(layers: Iterable[int] = BASELINE_LAYERS) -> list[ParameterSlice]:
    values = tuple(int(v) for v in layers)
    if values != BASELINE_LAYERS:
        raise ValueError(f"Baseline layout is fixed at {BASELINE_LAYERS}, got {values}")
    layout: list[ParameterSlice] = []
    offset = 0
    for layer_index, (fan_in, fan_out) in enumerate(zip(values[:-1], values[1:]), start=1):
        w_size = fan_out * fan_in
        layout.append(ParameterSlice(f"fc{layer_index}.weight", (fan_out, fan_in), offset, offset + w_size))
        offset += w_size
        layout.append(ParameterSlice(f"fc{layer_index}.bias", (fan_out,), offset, offset + fan_out))
        offset += fan_out
    return layout


PARAMETER_LAYOUT = build_parameter_layout()
GENOME_LENGTH = PARAMETER_LAYOUT[-1].stop


def decode_random_keys(random_keys: np.ndarray, lower_bound: float = -2.0, upper_bound: float = 10.0) -> np.ndarray:
    keys = np.asarray(random_keys, dtype=np.float32)
    if keys.shape[-1] != GENOME_LENGTH:
        raise ValueError(f"Expected genome length {GENOME_LENGTH}, got {keys.shape[-1]}")
    if np.any(keys < 0.0) or np.any(keys > 1.0):
        raise ValueError("Random keys must be inside [0, 1].")
    return (float(lower_bound) + (float(upper_bound) - float(lower_bound)) * keys).astype(np.float32, copy=False)


def parameter_manifest() -> list[dict[str, object]]:
    return [
        {"name": item.name, "shape": list(item.shape), "start": item.start, "stop": item.stop, "size": item.size}
        for item in PARAMETER_LAYOUT
    ]


class BatchedQNetwork:
    """Runs the fixed 2->128->128->5 network for a batch of genomes.

    One row in ``weights`` belongs to one evaluated individual. The implementation
    does not instantiate hundreds of ``nn.Module`` objects and therefore keeps the
    evaluator simple and GPU-friendly.
    """

    def __init__(self, weights: np.ndarray, device: str = "cuda") -> None:
        import torch

        matrix = np.asarray(weights, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.ndim != 2 or matrix.shape[1] != GENOME_LENGTH:
            raise ValueError(f"weights must have shape [batch, {GENOME_LENGTH}]")
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = matrix.shape[0]
        tensor = torch.as_tensor(matrix, dtype=torch.float32, device=self.device)
        self._params: dict[str, object] = {}
        for item in PARAMETER_LAYOUT:
            self._params[item.name] = tensor[:, item.start:item.stop].reshape((self.batch_size,) + item.shape)

    def q_values(self, observations: np.ndarray):
        import torch

        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if tuple(obs.shape) != (self.batch_size, 2):
            raise ValueError(f"Expected observations shape {(self.batch_size, 2)}, got {tuple(obs.shape)}")
        w1 = self._params["fc1.weight"]
        b1 = self._params["fc1.bias"]
        w2 = self._params["fc2.weight"]
        b2 = self._params["fc2.bias"]
        w3 = self._params["fc3.weight"]
        b3 = self._params["fc3.bias"]
        h1 = torch.relu(torch.bmm(w1, obs.unsqueeze(-1)).squeeze(-1) + b1)
        h2 = torch.relu(torch.bmm(w2, h1.unsqueeze(-1)).squeeze(-1) + b2)
        return torch.bmm(w3, h2.unsqueeze(-1)).squeeze(-1) + b3

    def actions(self, observations: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            actions = torch.argmax(self.q_values(observations), dim=1)
        return actions.detach().cpu().numpy().astype(np.int64, copy=False)
