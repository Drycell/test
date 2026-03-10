from __future__ import annotations

import numpy as np

from wormwing.connectome.types import StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.controllers.pd_baseline import PDBaselineController
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv


def run_pd_baseline(env: WingedWorm3DEnv, seed: int) -> float:
    obs, _ = env.reset(seed=seed)
    controller = PDBaselineController()
    total = 0.0
    done = False
    while not done:
        action = controller.act(obs)
        obs, rew, term, trunc, _ = env.step(action)
        total += float(rew)
        done = bool(term or trunc)
    return total


def run_fixed_readout_baseline(controller: ConnectomeCTRNN, env: WingedWorm3DEnv, seed: int) -> float:
    controller.reset(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        core = controller.step(env.virtual_sensor_obs(obs))
        action = np.clip(np.array([0.8 * core[0] - 0.2 * core[1], 0.8 * core[1] - 0.2 * core[0]]), -1.0, 1.0)
        obs, rew, term, trunc, _ = env.step(action)
        total += float(rew)
        done = bool(term or trunc)
    return total


def run_structural_candidate(controller: ConnectomeCTRNN, env: WingedWorm3DEnv, genome: StructuralGenome, seed: int) -> float:
    controller.apply_structural_genome(genome)
    controller.reset(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        obs, rew, term, trunc, _ = env.step(controller.step(env.virtual_sensor_obs(obs)))
        total += float(rew)
        done = bool(term or trunc)
    return total
