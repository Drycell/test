from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


@dataclass
class EnvConfig:
    episode_seconds: float = 8.0
    physics_dt: float = 0.005
    control_dt: float = 0.02
    start_height: float = 1.5
    start_forward_speed: float = 1.0
    crash_height: float = 0.08
    max_tilt_deg: float = 75.0


def build_xml() -> str:
    return """
<mujoco model='winged_worm'>
  <option timestep='0.005' gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='ground' type='plane' size='5 5 0.1' rgba='0.2 0.3 0.2 1'/>
    <body name='torso' pos='0 0 1.5'>
      <freejoint/>
      <geom type='capsule' fromto='-0.2 0 0 0.2 0 0' size='0.05' density='500'/>
      <body name='left_wing' pos='0 0.08 0'>
        <joint name='left_hinge' type='hinge' axis='1 0 0' range='-90 90'/>
        <geom type='capsule' fromto='0 0 0 0 0.3 0' size='0.02' density='100'/>
      </body>
      <body name='right_wing' pos='0 -0.08 0'>
        <joint name='right_hinge' type='hinge' axis='1 0 0' range='-90 90'/>
        <geom type='capsule' fromto='0 0 0 0 -0.3 0' size='0.02' density='100'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint='left_hinge' gear='5'/>
    <motor joint='right_hinge' gear='5'/>
  </actuator>
</mujoco>
"""


class WingedWorm3DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.config.control_dt / self.config.physics_dt))
        self.max_steps = int(self.config.episode_seconds / self.config.control_dt)
        self.step_count = 0

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos
        qvel = self.data.qvel
        quat = qpos[3:7]
        roll, pitch = self._quat_to_euler(quat)
        obs = np.array([
            math.cos(roll) * math.cos(pitch),
            roll,
            pitch,
            qvel[3],
            qvel[4],
            qvel[0],
            qvel[1],
            qvel[2],
            qpos[2],
            qpos[7],
            qpos[8],
            qvel[6],
            qvel[7],
        ], dtype=np.float32)
        return obs

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> tuple[float, float]:
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, pitch

    def _reward(self, action: np.ndarray, done: bool) -> tuple[float, dict[str, float]]:
        obs = self._get_obs()
        alive = 1.0
        forward = 0.1 * obs[5]
        height = 0.2 * obs[8]
        tilt = -0.3 * (abs(obs[1]) + abs(obs[2]))
        angvel = -0.05 * (abs(obs[3]) + abs(obs[4]))
        effort = -0.01 * float(np.sum(action ** 2))
        terminal = -5.0 if done and self.step_count < self.max_steps else 0.0
        total = alive + forward + height + tilt + angvel + effort + terminal
        parts = {
            "alive_reward": alive,
            "forward_component": forward,
            "height_component": height,
            "tilt_penalty": tilt,
            "angvel_penalty": angvel,
            "control_effort_penalty": effort,
            "terminal_penalty": terminal,
        }
        return total, parts

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[2] = self.config.start_height
        self.data.qvel[0] = self.config.start_forward_speed
        self.data.qpos[7] = rng.uniform(-0.05, 0.05)
        self.data.qpos[8] = rng.uniform(-0.05, 0.05)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.data.ctrl[:] = action
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        roll, pitch = obs[1], obs[2]
        crash = obs[8] < self.config.crash_height
        tilted = abs(roll) > math.radians(self.config.max_tilt_deg) or abs(pitch) > math.radians(self.config.max_tilt_deg)
        nan_state = not np.isfinite(obs).all()
        timeout = self.step_count >= self.max_steps
        terminated = crash or tilted or nan_state
        truncated = timeout and not terminated
        reward, parts = self._reward(action, terminated)
        info = {"reward_components": parts}
        return obs, reward, terminated, truncated, info
