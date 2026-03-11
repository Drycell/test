from __future__ import annotations

import numpy as np
from cmaes import CMA

from wormwing.controllers.fixed_readout import FixedReadoutController
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv


def _rollout(controller: FixedReadoutController, env: WingedWorm3DEnv, seed: int) -> float:
    controller.reset(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    while not done:
        action = controller.step(env.virtual_sensor_obs(obs))
        obs, rew, term, trunc, _ = env.step(action)
        total += float(rew)
        done = bool(term or trunc)
    return total


def run_fixed_readout_optimization(
    controller: FixedReadoutController,
    env: WingedWorm3DEnv,
    seeds: list[int],
    generations: int = 12,
    population_size: int = 16,
) -> dict[str, float]:
    optimizer = CMA(mean=np.zeros(controller.param_dim), sigma=0.5, population_size=population_size)
    best = -1e18
    for _ in range(generations):
        sols = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            controller.set_params(x)
            score = float(np.mean([_rollout(controller, env, s) for s in seeds]))
            sols.append((x, -score))
            best = max(best, score)
        optimizer.tell(sols)
    return {"fixed_readout_best_reward": float(best)}
