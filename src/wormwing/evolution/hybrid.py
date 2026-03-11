from __future__ import annotations

import numpy as np
from cmaes import CMA

from wormwing.connectome.types import HybridGenome, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv


def _evaluate_hybrid_candidate(
    controller: ConnectomeCTRNN,
    env: WingedWorm3DEnv,
    hybrid: HybridGenome,
    seeds: list[int],
    edit_penalty: float,
) -> float:
    controller.apply_hybrid_genome(hybrid)
    rewards: list[float] = []
    for s in seeds:
        controller.reset(s)
        obs, _ = env.reset(seed=s)
        total = 0.0
        done = False
        while not done:
            action = controller.step(env.virtual_sensor_obs(obs))
            obs, rew, term, trunc, _ = env.step(action)
            total += float(rew)
            done = bool(term or trunc)
        rewards.append(total)
    return float(np.mean(rewards) - edit_penalty * len(hybrid.structural.edits))


def run_structure_first_hybrid(
    controller: ConnectomeCTRNN,
    env: WingedWorm3DEnv,
    structural: StructuralGenome,
    seeds: list[int],
    local_steps: int = 8,
    sigma: float = 0.2,
    edit_penalty: float = 0.05,
) -> dict[str, float]:
    n = controller.w_chem.shape[0]
    dim = 2 * n
    opt = CMA(mean=np.zeros(dim, dtype=float), sigma=sigma, population_size=8)
    best = -1e18
    for _ in range(local_steps):
        sols: list[tuple[np.ndarray, float]] = []
        for _ in range(opt.population_size):
            x = opt.ask()
            g = HybridGenome(structural=structural, bias_delta=x[:n], tau_log_scale_delta=x[n:], edited_edge_scale_delta=None)
            score = _evaluate_hybrid_candidate(controller, env, g, seeds, edit_penalty=edit_penalty)
            sols.append((x, -score))
            best = max(best, score)
        opt.tell(sols)
    return {"hybrid_best_reward": float(best)}
