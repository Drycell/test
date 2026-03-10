from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wormwing.connectome.types import ConnectomeData, HybridGenome, StructuralGenome


@dataclass
class ControllerState:
    x: np.ndarray
    t: float


class ConnectomeCTRNN:
    def __init__(self, connectome: ConnectomeData, dt: float = 0.02, tau_init: float = 1.0, bias_init: float = 0.0):
        self.connectome = connectome
        self.dt = dt
        self.base_chem = connectome.chemical_weights.copy()
        self.base_gap = connectome.gap_weights.copy()
        self.w_chem = self.base_chem.copy()
        self.w_gap = self.base_gap.copy()
        self.tau = np.full(self.w_chem.shape[0], tau_init, dtype=float)
        self.bias = np.full(self.w_chem.shape[0], bias_init, dtype=float)
        self.state = ControllerState(x=np.zeros(self.w_chem.shape[0], dtype=float), t=0.0)

    def reset(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.state = ControllerState(x=rng.normal(0.0, 0.01, self.w_chem.shape[0]), t=0.0)

    def _inject_observation(self, obs: np.ndarray) -> None:
        # obs indices: [0:up,1:roll,2:pitch,3:droll,4:dpitch,5:vx,6:vy,7:vz,8:height,...]
        sensor_values = np.array([obs[1], obs[2], obs[3], obs[4], obs[8], obs[7]], dtype=float)
        for i, idx in enumerate(self.connectome.virtual_sensor_indices):
            self.state.x[idx] = float(sensor_values[i]) if i < len(sensor_values) else 0.0

    def step(self, observation: np.ndarray) -> np.ndarray:
        self._inject_observation(observation)
        x = self.state.x
        tanh_x = np.tanh(x)
        gap_term = np.sum(self.w_gap * (x[None, :] - x[:, None]), axis=1)
        x_dot = (-x + self.w_chem @ tanh_x + gap_term + self.bias) / np.maximum(self.tau, 1e-3)
        self.state.x = x + self.dt * x_dot
        # Virtual sensory nodes are clamped to observation channels each control step.
        # Re-apply after integration so recurrent dynamics cannot drift these nodes.
        self._inject_observation(observation)
        self.state.t += self.dt
        out = np.zeros(2, dtype=float)
        for i, idx in enumerate(self.connectome.virtual_motor_indices[:2]):
            out[i] = np.tanh(self.state.x[idx])
        return np.clip(out, -1.0, 1.0)

    def serialize_state(self) -> dict:
        return {"x": self.state.x.copy(), "t": float(self.state.t)}

    def load_state(self, state: dict) -> None:
        self.state = ControllerState(x=np.array(state["x"], dtype=float), t=float(state["t"]))

    def apply_structural_genome(self, genome: StructuralGenome) -> None:
        self.w_chem = self.base_chem.copy()
        self.w_gap = self.base_gap.copy()
        for edit in genome.edits[: genome.max_edits]:
            if edit.src == edit.dst:
                continue
            if edit.op == "add_chem":
                self.w_chem[edit.src, edit.dst] = float(edit.value if edit.value is not None else 0.5)
            elif edit.op == "del_chem":
                self.w_chem[edit.src, edit.dst] = 0.0
            elif edit.op == "flip_sign":
                self.w_chem[edit.src, edit.dst] *= -1.0
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

    def apply_hybrid_genome(self, genome: HybridGenome) -> None:
        self.apply_structural_genome(genome.structural)
        if genome.bias_delta is not None:
            self.bias += genome.bias_delta
        if genome.tau_log_scale_delta is not None:
            self.tau *= np.exp(genome.tau_log_scale_delta)
