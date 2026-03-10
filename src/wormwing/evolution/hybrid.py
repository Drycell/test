from __future__ import annotations

import numpy as np
from cmaes import CMA

from wormwing.connectome.types import HybridGenome, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv
from wormwing.evolution.structure_only import evaluate_genome


def run_structure_first_hybrid(
    controller: ConnectomeCTRNN,
    env: WingedWorm3DEnv,
    structural: StructuralGenome,
    seeds: list[int],
    local_steps: int = 8,
    sigma: float = 0.2,
) -> dict[str, float]:
    n = controller.w_chem.shape[0]
    dim = 2 * n
    opt = CMA(mean=np.zeros(dim, dtype=float), sigma=sigma, population_size=8)
    best = -1e18
    for _ in range(local_steps):
        sols = []
        for _ in range(opt.population_size):
            x = opt.ask()
            g = HybridGenome(
                structural=structural,
                bias_delta=x[:n],
                tau_log_scale_delta=x[n:],
                edited_edge_scale_delta=None,
            )
            controller.apply_hybrid_genome(g)
            score, _, _ = evaluate_genome(structural, controller, env, seeds, edit_penalty=0.05)
            sols.append((x, -score))
            best = max(best, score)
        opt.tell(sols)
    return {"hybrid_best_reward": float(best)}
