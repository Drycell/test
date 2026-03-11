from __future__ import annotations

import numpy as np


class PDBaselineController:
    def __init__(self, kp_roll: float = 0.8, kd_roll: float = 0.2, kp_pitch: float = 0.5, kd_pitch: float = 0.1):
        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch

    def act(self, obs: np.ndarray) -> np.ndarray:
        roll, pitch = float(obs[1]), float(obs[2])
        droll, dpitch = float(obs[3]), float(obs[4])
        balance = -(self.kp_roll * roll + self.kd_roll * droll)
        pitch_ctrl = -(self.kp_pitch * pitch + self.kd_pitch * dpitch)
        left = balance + pitch_ctrl
        right = -balance + pitch_ctrl
        return np.clip(np.array([left, right], dtype=float), -1.0, 1.0)
