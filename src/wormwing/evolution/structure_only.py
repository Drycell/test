from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from wormwing.connectome.types import StructuralEdit, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv

EDGE_VALUES = [-1.0, -0.5, 0.5, 1.0]


def random_genome(n_nodes: int, max_edits: int, seed: int) -> StructuralGenome:
    rng = np.random.default_rng(seed)
    edits = []
    n = int(rng.integers(1, max_edits + 1))
    for _ in range(n):
        src, dst = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        if src == dst:
            continue
        edits.append(StructuralEdit(op="add_chem", src=src, dst=dst, value=float(rng.choice(EDGE_VALUES))))
    return StructuralGenome(edits=edits, max_edits=max_edits, seed=seed)


def _run_episode(genome: StructuralGenome, controller: ConnectomeCTRNN, env: WingedWorm3DEnv, seed: int) -> tuple[float, dict[str, Any], np.ndarray]:
    controller.apply_structural_genome(genome)
    controller.reset(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    forward_end = 0.0
    heights: list[float] = []
    trajectory: list[np.ndarray] = []
    while not done:
        trajectory.append(obs.astype(np.float32, copy=True))
        action = controller.step(obs)
        obs, rew, term, trunc, _ = env.step(action)
        total += float(rew)
        steps += 1
        forward_end = float(obs[5])
        heights.append(float(obs[8]))
        done = bool(term or trunc)

    info = {
        "episode_length": steps,
        "forward_distance": forward_end,
        "mean_height": float(np.mean(heights)) if heights else 0.0,
        "termination_reason": "done",
    }
    return float(total - 0.05 * len(genome.edits)), info, np.asarray(trajectory, dtype=np.float32)


def _apply_edits(base: np.ndarray, genome: StructuralGenome) -> np.ndarray:
    out = base.copy()
    for edit in genome.edits[: genome.max_edits]:
        if edit.src == edit.dst:
            continue
        if edit.op == "add_chem":
            out[edit.src, edit.dst] = float(edit.value if edit.value is not None else 0.5)
        elif edit.op == "del_chem":
            out[edit.src, edit.dst] = 0.0
        elif edit.op == "flip_sign":
            out[edit.src, edit.dst] *= -1.0
    return out


def _write_graph_export(path: Path, matrix: np.ndarray) -> None:
    lines = ["graph [", "  directed 1"]
    n = matrix.shape[0]
    for i in range(n):
        lines.extend(["  node [", f"    id {i}", f"    label \"{i}\"", "  ]"])
    src_idx, dst_idx = np.where(matrix != 0.0)
    for s, d in zip(src_idx.tolist(), dst_idx.tolist()):
        w = float(matrix[s, d])
        lines.extend(["  edge [", f"    source {s}", f"    target {d}", f"    weight {w}", "  ]"])
    lines.append("]")
    path.write_text("\n".join(lines))


def run_structure_only(
    controller: ConnectomeCTRNN,
    env: WingedWorm3DEnv,
    run_dir: Path,
    generations: int = 3,
    population_size: int = 8,
    max_edits: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    best: StructuralGenome | None = None
    best_score = -1e18
    best_info: dict[str, Any] = {}
    best_trajectory = np.zeros((0, 13), dtype=np.float32)
    rows = []

    for g in range(generations):
        gen_best = -1e18
        for _ in range(population_size):
            genome = random_genome(controller.w_chem.shape[0], max_edits=max_edits, seed=int(rng.integers(0, 1_000_000)))
            score, info, traj = _run_episode(genome, controller, env, seed=int(rng.integers(0, 10000)))
            gen_best = max(gen_best, score)
            if score > best_score:
                best, best_score = genome, score
                best_info = info
                best_trajectory = traj
        rows.append({"generation": g, "best_fitness": gen_best})

    assert best is not None
    metrics = {
        "best_fitness": float(best_score),
        "edit_count": float(len(best.edits)),
        "mean_episode_length": float(best_info.get("episode_length", 0.0)),
        "mean_height": float(best_info.get("mean_height", 0.0)),
        "mean_forward_distance": float(best_info.get("forward_distance", 0.0)),
        "graph_edit_distance_proxy": float(len(best.edits)),
    }

    (run_dir / "generation_summary.csv").write_text(
        "generation,best_fitness\n" + "\n".join(f"{r['generation']},{r['best_fitness']}" for r in rows)
    )
    (run_dir / "metrics.csv").write_text("key,value\n" + "\n".join(f"{k},{v}" for k, v in metrics.items()))
    (run_dir / "best_genome.json").write_text(json.dumps(asdict(best), indent=2))

    before = controller.base_chem
    after = _apply_edits(before, best)
    edit_summary = {
        "edge_count_before": int(np.sum(before != 0.0)),
        "edge_count_after": int(np.sum(after != 0.0)),
        "edit_count": len(best.edits),
    }
    (run_dir / "best_graph_before_after.json").write_text(json.dumps(edit_summary, indent=2))
    _write_graph_export(run_dir / "best_graph_before.gml", before)
    _write_graph_export(run_dir / "best_graph_after.gml", after)

    ckpt = {"best_genome": asdict(best), "metrics": metrics}
    with (run_dir / "best.ckpt").open("wb") as f:
        pickle.dump(ckpt, f)
    with (run_dir / "last.ckpt").open("wb") as f:
        pickle.dump(ckpt, f)

    np.savez(run_dir / "trajectory_best.npz", observations=best_trajectory)
    return metrics
