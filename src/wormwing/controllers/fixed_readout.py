from __future__ import annotations

import numpy as np

from wormwing.controllers.ctrnn import ConnectomeCTRNN


class FixedReadoutController(ConnectomeCTRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout_w = np.eye(2, dtype=float)
        self.readout_b = np.zeros(2, dtype=float)

    @property
    def param_dim(self) -> int:
        return 6

    def set_params(self, params: np.ndarray) -> None:
        p = np.asarray(params, dtype=float)
        self.readout_w = p[:4].reshape(2, 2)
        self.readout_b = p[4:6]

    def step(self, observation: np.ndarray) -> np.ndarray:
        core = super().step(observation)
        return np.clip(self.readout_w @ core + self.readout_b, -1.0, 1.0)
