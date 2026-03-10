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


def _random_edit(rng: np.random.Generator, n_nodes: int) -> StructuralEdit:
    op = rng.choice(["add_chem", "del_chem", "flip_sign", "retarget_chem"])
    src, dst = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
    while src == dst:
        dst = int(rng.integers(0, n_nodes))
    if op == "retarget_chem":
        aux_src, aux_dst = int(rng.integers(0, n_nodes)), int(rng.integers(0, n_nodes))
        return StructuralEdit(op=op, src=src, dst=dst, aux_src=aux_src, aux_dst=aux_dst, value=float(rng.choice(EDGE_VALUES)))
    return StructuralEdit(op=op, src=src, dst=dst, value=float(rng.choice(EDGE_VALUES)))


def random_genome(n_nodes: int, max_edits: int, seed: int) -> StructuralGenome:
    rng = np.random.default_rng(seed)
    edits = [_random_edit(rng, n_nodes) for _ in range(int(rng.integers(1, max_edits + 1)))]
    return StructuralGenome(edits=edits, max_edits=max_edits, seed=seed)


def mutate_genome(parent: StructuralGenome, n_nodes: int, seed: int, mutation_cfg: dict[str, float]) -> StructuralGenome:
    rng = np.random.default_rng(seed)
    edits = list(parent.edits)
    p_append = float(mutation_cfg.get("p_append", 0.35))
    p_delete = float(mutation_cfg.get("p_delete", 0.15))
    p_replace = float(mutation_cfg.get("p_replace", 0.20))
    p_retarget = float(mutation_cfg.get("p_retarget", 0.15))
    p_rescale = float(mutation_cfg.get("p_rescale", 0.15))

    r = rng.random()
    if r < p_append and len(edits) < parent.max_edits:
        edits.append(_random_edit(rng, n_nodes))
    elif r < p_append + p_delete and edits:
        del edits[int(rng.integers(0, len(edits)))]
    elif r < p_append + p_delete + p_replace and edits:
        edits[int(rng.integers(0, len(edits)))] = _random_edit(rng, n_nodes)
    elif r < p_append + p_delete + p_replace + p_retarget and edits:
        i = int(rng.integers(0, len(edits)))
        e = edits[i]
        edits[i] = StructuralEdit(op=e.op, src=e.src, dst=e.dst, aux_src=int(rng.integers(0, n_nodes)), aux_dst=int(rng.integers(0, n_nodes)), value=e.value)
    elif edits:
        i = int(rng.integers(0, len(edits)))
        e = edits[i]
        edits[i] = StructuralEdit(op=e.op, src=e.src, dst=e.dst, aux_src=e.aux_src, aux_dst=e.aux_dst, value=float(rng.choice(EDGE_VALUES)))

    return StructuralGenome(edits=edits[: parent.max_edits], max_edits=parent.max_edits, seed=seed)


def _run_episode(genome: StructuralGenome, controller: ConnectomeCTRNN, env: WingedWorm3DEnv, seed: int) -> tuple[float, dict[str, Any], np.ndarray]:
    controller.apply_structural_genome(genome)
    controller.reset(seed)
    obs, _ = env.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    heights: list[float] = []
    trajectory: list[np.ndarray] = []
    while not done:
        trajectory.append(obs.astype(np.float32, copy=True))
        obs, rew, term, trunc, _ = env.step(controller.step(obs))
        total += float(rew)
        steps += 1
        heights.append(float(obs[8]))
        done = bool(term or trunc)
    return total, {"episode_length": steps, "mean_height": float(np.mean(heights)) if heights else 0.0, "forward_distance": float(obs[5])}, np.asarray(trajectory, dtype=np.float32)


def evaluate_genome(genome: StructuralGenome, controller: ConnectomeCTRNN, env: WingedWorm3DEnv, seeds: list[int], edit_penalty: float) -> tuple[float, dict[str, Any], np.ndarray]:
    scores, infos = [], []
    best_traj = np.zeros((0, 13), dtype=np.float32)
    for s in seeds:
        score, info, traj = _run_episode(genome, controller, env, s)
        scores.append(score)
        infos.append(info)
        if traj.shape[0] > best_traj.shape[0]:
            best_traj = traj
    mean_score = float(np.mean(scores) - edit_penalty * len(genome.edits))
    mean_info = {
        "episode_length": float(np.mean([i["episode_length"] for i in infos])),
        "mean_height": float(np.mean([i["mean_height"] for i in infos])),
        "forward_distance": float(np.mean([i["forward_distance"] for i in infos])),
    }
    return mean_score, mean_info, best_traj


def _apply_edits(base: np.ndarray, genome: StructuralGenome) -> np.ndarray:
    out = base.copy()
    for e in genome.edits[: genome.max_edits]:
        if e.src == e.dst:
            continue
        if e.op == "add_chem": out[e.src, e.dst] = float(e.value if e.value is not None else 0.5)
        elif e.op == "del_chem": out[e.src, e.dst] = 0.0
        elif e.op == "flip_sign": out[e.src, e.dst] *= -1.0
        elif e.op == "retarget_chem" and e.aux_src is not None and e.aux_dst is not None:
            out[e.src, e.dst] = 0.0; out[e.aux_src, e.aux_dst] = float(e.value if e.value is not None else 0.5)
    return out


def _write_graph_export(path: Path, matrix: np.ndarray) -> None:
    lines = ["graph [", "  directed 1"]
    for i in range(matrix.shape[0]): lines += ["  node [", f"    id {i}", f"    label \"{i}\"", "  ]"]
    src, dst = np.where(matrix != 0.0)
    for s, d in zip(src.tolist(), dst.tolist()): lines += ["  edge [", f"    source {s}", f"    target {d}", f"    weight {float(matrix[s,d])}", "  ]"]
    lines.append("]")
    path.write_text("\n".join(lines))


def run_structure_only(controller: ConnectomeCTRNN, env: WingedWorm3DEnv, run_dir: Path, generations: int = 3, population_size: int = 8, max_edits: int = 8, seed: int = 0, elite_count: int = 2, mutation_cfg: dict[str, float] | None = None, train_eval_seeds: list[int] | None = None, edit_penalty: float = 0.05) -> dict[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    mutation_cfg = mutation_cfg or {}
    eval_seeds = train_eval_seeds or [0, 1, 2, 3]
    rng = np.random.default_rng(seed)

    population = [random_genome(controller.w_chem.shape[0], max_edits, int(rng.integers(0, 1_000_000))) for _ in range(population_size)]
    best, best_score, best_info, best_traj = None, -1e18, {}, np.zeros((0, 13), dtype=np.float32)
    rows = []

    for g in range(generations):
        scored = []
        for genome in population:
            score, info, traj = evaluate_genome(genome, controller, env, eval_seeds, edit_penalty)
            scored.append((score, genome, info, traj))
            if score > best_score:
                best, best_score, best_info, best_traj = genome, score, info, traj
        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [x[1] for x in scored[: max(1, elite_count)]]
        rows.append({"generation": g, "best_fitness": float(scored[0][0])})

        offspring = []
        while len(offspring) + len(elites) < population_size:
            p = elites[int(rng.integers(0, len(elites)))]
            offspring.append(mutate_genome(p, controller.w_chem.shape[0], int(rng.integers(0, 1_000_000)), mutation_cfg))
        population = elites + offspring

    assert best is not None
    metrics = {
        "best_fitness": float(best_score),
        "edit_count": float(len(best.edits)),
        "mean_episode_length": float(best_info.get("episode_length", 0.0)),
        "mean_height": float(best_info.get("mean_height", 0.0)),
        "mean_forward_distance": float(best_info.get("forward_distance", 0.0)),
        "graph_edit_distance_proxy": float(len(best.edits)),
    }

    (run_dir / "generation_summary.csv").write_text("generation,best_fitness\n" + "\n".join(f"{r['generation']},{r['best_fitness']}" for r in rows))
    (run_dir / "metrics.csv").write_text("key,value\n" + "\n".join(f"{k},{v}" for k, v in metrics.items()))
    (run_dir / "best_genome.json").write_text(json.dumps(asdict(best), indent=2))

    before, after = controller.base_chem, _apply_edits(controller.base_chem, best)
    (run_dir / "best_graph_before_after.json").write_text(json.dumps({"edge_count_before": int(np.sum(before != 0.0)), "edge_count_after": int(np.sum(after != 0.0)), "edit_count": len(best.edits)}, indent=2))
    _write_graph_export(run_dir / "best_graph_before.gml", before)
    _write_graph_export(run_dir / "best_graph_after.gml", after)

    ckpt = {"best_genome": asdict(best), "metrics": metrics}
    with (run_dir / "best.ckpt").open("wb") as f: pickle.dump(ckpt, f)
    with (run_dir / "last.ckpt").open("wb") as f: pickle.dump(ckpt, f)
    np.savez(run_dir / "trajectory_best.npz", observations=best_traj)
    return metrics
