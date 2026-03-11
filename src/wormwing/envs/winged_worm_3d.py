from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from wormwing.aero.quasi_steady import body_drag, wing_lift_drag, wing_torque_damping


@dataclass
class EnvConfig:
    episode_seconds: float = 2.0
    physics_dt: float = 0.00005
    control_dt: float = 0.001
    start_height: float = 0.005
    start_forward_speed: float = 0.05
    crash_height: float = 0.0002
    max_tilt_deg: float = 75.0
    body_length_m: float = 0.001
    body_radius_m: float = 0.00004
    wing_span_m: float = 0.0005
    wing_chord_m: float = 0.0004
    air_density: float = 1.225
    wing_target_range_rad: float = 1.5
    beat_freq_hz: float = 400.0
    suppress_mujoco_warnings: bool = True
    energy_budget: float = 1.0
    basal_cost_per_step: float = 0.0001
    wing_cost_scale: float = 0.002
    stall_cost_per_step: float = 0.005


def build_xml(cfg: EnvConfig) -> str:
    half_body = cfg.body_length_m * 0.5
    wing_attach = cfg.body_radius_m * 1.8
    wing_half_span = cfg.wing_span_m * 0.5
    wing_half_chord = cfg.wing_chord_m * 0.5
    wing_half_thickness = max(1e-6, cfg.wing_chord_m * 0.0125)
    return f"""
<mujoco model='winged_worm'>
  <compiler inertiafromgeom='false' boundmass='1e-14' boundinertia='1e-20'/>
  <option timestep='{cfg.physics_dt}' gravity='0 0 -9.81'/>
  <worldbody>
    <geom name='ground' type='plane' size='0.05 0.05 0.001' rgba='0.2 0.3 0.2 1'/>
    <body name='torso' pos='0 0 {cfg.start_height}'>
      <freejoint/>
      <inertial pos='0 0 0' mass='5.0e-9' diaginertia='4e-15 4e-15 1e-15'/>
      <geom type='capsule' fromto='-{half_body} 0 0 {half_body} 0 0' size='{cfg.body_radius_m}' contype='0' conaffinity='0'/>
      <body name='left_wing' pos='0 {wing_attach} 0'>
        <joint name='left_hinge' type='hinge' axis='1 0 0' range='-90 90'/>
        <inertial pos='0 {wing_half_span} 0' mass='5.0e-11' diaginertia='1e-15 1e-15 1e-15'/>
        <geom name='left_wing_geom' type='box' pos='0 {wing_half_span} 0' size='{wing_half_chord} {wing_half_span} {wing_half_thickness}' contype='0' conaffinity='0'/>
      </body>
      <body name='right_wing' pos='0 -{wing_attach} 0'>
        <joint name='right_hinge' type='hinge' axis='1 0 0' range='-90 90'/>
        <inertial pos='0 -{wing_half_span} 0' mass='5.0e-11' diaginertia='1e-15 1e-15 1e-15'/>
        <geom name='right_wing_geom' type='box' pos='0 -{wing_half_span} 0' size='{wing_half_chord} {wing_half_span} {wing_half_thickness}' contype='0' conaffinity='0'/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint='left_hinge' kp='1e-10' ctrlrange='-{cfg.wing_target_range_rad} {cfg.wing_target_range_rad}'/>
    <position joint='right_hinge' kp='1e-10' ctrlrange='-{cfg.wing_target_range_rad} {cfg.wing_target_range_rad}'/>
  </actuator>
</mujoco>
"""


class WingedWorm3DEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        if self.config.suppress_mujoco_warnings:
            mujoco.set_mju_user_warning(lambda _msg: None)
        self.model = mujoco.MjModel.from_xml_string(build_xml(self.config))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.config.control_dt / self.config.physics_dt))
        self.max_steps = int(self.config.episode_seconds / self.config.control_dt)
        self.step_count = 0
        self.left_wing_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_wing")
        self.right_wing_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wing")
        self.torso_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.left_hinge_qvel_idx = 6
        self.right_hinge_qvel_idx = 7
        self.left_hinge_qpos_idx = 7
        self.right_hinge_qpos_idx = 8
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.energy = float(self.config.energy_budget)

    def _get_obs(self) -> np.ndarray:
        qpos, qvel = self.data.qpos, self.data.qvel
        roll, pitch = self._quat_to_euler(qpos[3:7])
        return np.array([
            math.cos(roll) * math.cos(pitch),
            roll,
            pitch,
            qvel[3],
            qvel[4],
            qvel[0],
            qvel[1],
            qvel[2],
            qpos[2],
            qpos[self.left_hinge_qpos_idx],
            qpos[self.right_hinge_qpos_idx],
            qvel[self.left_hinge_qvel_idx],
            qvel[self.right_hinge_qvel_idx],
        ], dtype=np.float32)

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> tuple[float, float]:
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        return roll, pitch

    def _apply_aero(self) -> None:
        vx = float(self.data.qvel[0])
        lhv = float(self.data.qvel[self.left_hinge_qvel_idx])
        rhv = float(self.data.qvel[self.right_hinge_qvel_idx])
        lha = float(self.data.qpos[self.left_hinge_qpos_idx])
        rha = float(self.data.qpos[self.right_hinge_qpos_idx])
        llift, ldrag = wing_lift_drag(
            lhv,
            vx,
            self.config.wing_span_m,
            self.config.wing_chord_m,
            self.config.air_density,
            hinge_angle=lha,
            beat_freq_hz=self.config.beat_freq_hz,
        )
        rlift, rdrag = wing_lift_drag(
            rhv,
            vx,
            self.config.wing_span_m,
            self.config.wing_chord_m,
            self.config.air_density,
            hinge_angle=rha,
            beat_freq_hz=self.config.beat_freq_hz,
        )
        body_area = self.config.body_length_m * 2.0 * self.config.body_radius_m
        torso_drag = body_drag(vx, area_m2=body_area, air_density=self.config.air_density)

        self.data.xfrc_applied[self.left_wing_body] = np.array([ldrag, 0.0, llift, 0.0, 0.0, 0.0])
        self.data.xfrc_applied[self.right_wing_body] = np.array([rdrag, 0.0, rlift, 0.0, 0.0, 0.0])
        self.data.xfrc_applied[self.torso_body, 0] += torso_drag

        self.data.qfrc_applied[self.left_hinge_qvel_idx] += wing_torque_damping(lhv)
        self.data.qfrc_applied[self.right_hinge_qvel_idx] += wing_torque_damping(rhv)

    def virtual_sensor_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.array([obs[1], obs[2], obs[3], obs[4], obs[8], obs[7]], dtype=np.float32)

    def _reward(self, action: np.ndarray, done: bool, energy_component: float) -> tuple[float, dict[str, float]]:
        obs = self._get_obs()
        alive = 1.0
        forward = 2.0 * (obs[5] / max(1e-6, self.config.start_forward_speed))
        height = 2.0 * max(0.0, obs[8] / max(1e-6, self.config.start_height))
        tilt = -0.3 * (abs(obs[1]) + abs(obs[2]))
        angvel = -0.05 * (abs(obs[3]) + abs(obs[4]))
        effort = -0.01 * float(np.sum(action**2))
        terminal = -5.0 if done and self.step_count < self.max_steps else 0.0
        total = alive + forward + height + tilt + angvel + effort + energy_component + terminal
        return total, {
            "alive_reward": alive,
            "forward_component": forward,
            "height_component": height,
            "tilt_penalty": tilt,
            "angvel_penalty": angvel,
            "control_effort_penalty": effort,
            "energy_component": energy_component,
            "terminal_penalty": terminal,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        self.data.qpos[2] = self.config.start_height
        roll = rng.uniform(-0.12, 0.12)
        pitch = rng.uniform(-0.12, 0.12)
        q = np.array([
            math.cos(roll / 2) * math.cos(pitch / 2),
            math.sin(roll / 2) * math.cos(pitch / 2),
            math.cos(roll / 2) * math.sin(pitch / 2),
            0.0,
        ])
        self.data.qpos[3:7] = q
        self.data.qvel[0] = self.config.start_forward_speed + rng.uniform(-0.002, 0.002)
        self.data.qvel[3] = rng.uniform(-0.1, 0.1)
        self.data.qvel[4] = rng.uniform(-0.1, 0.1)
        self.data.qpos[self.left_hinge_qpos_idx] = rng.uniform(0.3, 0.5)
        self.data.qpos[self.right_hinge_qpos_idx] = rng.uniform(-0.5, -0.3)
        self.energy = float(self.config.energy_budget)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.data.ctrl[:] = action * self.config.wing_target_range_rad
        for _ in range(self.n_substeps):
            self._apply_aero()
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        roll, pitch = obs[1], obs[2]
        crash = obs[8] < self.config.crash_height
        tilted = abs(roll) > math.radians(self.config.max_tilt_deg) or abs(pitch) > math.radians(self.config.max_tilt_deg)
        nan_state = not np.isfinite(obs).all()
        wing_activity = float(np.sum(np.abs(action)))
        stall_cost = self.config.stall_cost_per_step if wing_activity < 0.1 else 0.0
        self.energy -= self.config.basal_cost_per_step + self.config.wing_cost_scale * wing_activity + stall_cost
        exhausted = self.energy <= 0.0
        timeout = self.step_count >= self.max_steps
        terminated = bool(crash or tilted or nan_state or exhausted)
        truncated = bool(timeout and not terminated)
        reward, parts = self._reward(action, terminated, energy_component=0.3 * max(0.0, self.energy))
        reason = "running"
        if crash:
            reason = "crash"
        elif tilted:
            reason = "tilted"
        elif nan_state:
            reason = "nan"
        elif exhausted:
            reason = "exhausted"
        elif timeout:
            reason = "timeout"
        return obs, reward, terminated, truncated, {"reward_components": parts, "termination_reason": reason}
