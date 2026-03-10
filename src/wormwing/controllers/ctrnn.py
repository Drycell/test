from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wormwing.connectome.types import ConnectomeData, HybridGenome, StructuralGenome


@dataclass
class ControllerState:
    x: np.ndarray
    t: float


class ConnectomeCTRNN:
    def __init__(
        self,
        connectome: ConnectomeData,
        dt: float = 0.02,
        tau_init: float = 1.0,
        bias_init: float = 0.0,
        motor_bias: tuple[float, float] = (0.0, 0.0),
    ):
        self.connectome = connectome
        self.dt = dt
        self.base_chem = connectome.chemical_weights.copy()
        self.base_gap = connectome.gap_weights.copy()
        self.w_chem = self.base_chem.copy()
        self.w_gap = self.base_gap.copy()
        self.tau = np.full(self.w_chem.shape[0], tau_init, dtype=float)
        self.bias = np.full(self.w_chem.shape[0], bias_init, dtype=float)
        self.motor_bias = np.asarray(motor_bias, dtype=float)
        self.state = ControllerState(x=np.zeros(self.w_chem.shape[0], dtype=float), t=0.0)
        self._last_edited_edges: list[tuple[int, int]] = []

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.state = ControllerState(x=rng.normal(0.0, 0.01, self.w_chem.shape[0]), t=0.0)

    def _sensor_vector(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=float)
        if obs.shape[0] == 6:
            return obs
        # obs indices: [0:up,1:roll,2:pitch,3:droll,4:dpitch,5:vx,6:vy,7:vz,8:height,...]
        return np.array([obs[1], obs[2], obs[3], obs[4], obs[8], obs[7]], dtype=float)

    def _inject_observation(self, observation: np.ndarray) -> None:
        sensor_values = self._sensor_vector(observation)
        for i, idx in enumerate(self.connectome.virtual_sensor_indices):
            self.state.x[idx] = float(sensor_values[i]) if i < len(sensor_values) else 0.0

    def step(self, observation: np.ndarray) -> np.ndarray:
        self._inject_observation(observation)
        x = np.nan_to_num(self.state.x, nan=0.0, posinf=10.0, neginf=-10.0)
        tanh_x = np.tanh(x)
        gap_term = np.sum(self.w_gap * (x[None, :] - x[:, None]), axis=1)
        x_dot = (-x + self.w_chem @ tanh_x + gap_term + self.bias) / np.maximum(self.tau, 1e-3)
        self.state.x = np.nan_to_num(x + self.dt * x_dot, nan=0.0, posinf=10.0, neginf=-10.0)
        self._inject_observation(observation)
        self.state.t += self.dt
        out = np.zeros(2, dtype=float)
        for i, idx in enumerate(self.connectome.virtual_motor_indices[:2]):
            out[i] = np.tanh(self.state.x[idx])
        out = out + self.motor_bias
        return np.clip(out, -1.0, 1.0)

    def serialize_state(self) -> dict:
        return {"x": self.state.x.copy(), "t": float(self.state.t)}

    def load_state(self, state: dict) -> None:
        self.state = ControllerState(x=np.array(state["x"], dtype=float), t=float(state["t"]))

    def apply_structural_genome(self, genome: StructuralGenome) -> None:
        self.w_chem = self.base_chem.copy()
        self.w_gap = self.base_gap.copy()
        self._last_edited_edges = []
        for edit in genome.edits[: genome.max_edits]:
            if edit.src == edit.dst:
                continue
            if edit.op == "add_chem":
                self.w_chem[edit.src, edit.dst] = float(edit.value if edit.value is not None else 0.5)
                self._last_edited_edges.append((edit.src, edit.dst))
            elif edit.op == "del_chem":
                self.w_chem[edit.src, edit.dst] = 0.0
            elif edit.op == "flip_sign":
                self.w_chem[edit.src, edit.dst] *= -1.0
                self._last_edited_edges.append((edit.src, edit.dst))
            elif edit.op == "add_gap":
                v = float(edit.value if edit.value is not None else 0.1)
                self.w_gap[edit.src, edit.dst] = v
                self.w_gap[edit.dst, edit.src] = v
            elif edit.op == "del_gap":
                self.w_gap[edit.src, edit.dst] = 0.0
                self.w_gap[edit.dst, edit.src] = 0.0
            elif edit.op == "retarget_chem" and edit.aux_src is not None and edit.aux_dst is not None:
                self.w_chem[edit.src, edit.dst] = 0.0
                self.w_chem[edit.aux_src, edit.aux_dst] = float(edit.value if edit.value is not None else 0.5)
                self._last_edited_edges.append((edit.aux_src, edit.aux_dst))

    def apply_hybrid_genome(self, genome: HybridGenome) -> None:
        self.apply_structural_genome(genome.structural)
        if genome.bias_delta is not None:
            self.bias += np.clip(genome.bias_delta, -2.0, 2.0)
        if genome.tau_log_scale_delta is not None:
            self.tau *= np.exp(np.clip(genome.tau_log_scale_delta, -5.0, 5.0))
            self.tau = np.clip(self.tau, 0.01, 100.0)
        if genome.edited_edge_scale_delta is not None and self._last_edited_edges:
            scales = np.exp(np.clip(genome.edited_edge_scale_delta, -5.0, 5.0))
            for i, (s, d) in enumerate(self._last_edited_edges[: len(scales)]):
                self.w_chem[s, d] *= float(scales[i])
