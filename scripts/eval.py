from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import yaml

from wormwing.connectome.loader import load_connectome
from wormwing.connectome.types import StructuralEdit, StructuralGenome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import EnvConfig, WingedWorm3DEnv
from wormwing.evolution.structure_only import evaluate_genome


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    conn = load_connectome("data/mock_connectome" if cfg.get("experiment", {}).get("connectome_mode", "mock") == "mock" else cfg.get("experiment", {}).get("real_data_dir", "data/real_connectome"))
    env = WingedWorm3DEnv(EnvConfig(**{k: v for k, v in cfg.get("env", {}).items() if k in EnvConfig.__annotations__}))
    controller = ConnectomeCTRNN(conn, dt=float(cfg.get("env", {}).get("control_dt", 0.02)))

    with (run_dir / "best.ckpt").open("rb") as f:
        ckpt = pickle.load(f)
    g = ckpt["best_genome"]
    genome = StructuralGenome(edits=[StructuralEdit(**e) for e in g["edits"]], max_edits=int(g["max_edits"]), seed=int(g["seed"]))
    seeds = list(cfg.get("evolution", {}).get("train_eval_seeds", [0, 1, 2, 3]))
    score, info, _ = evaluate_genome(genome, controller, env, seeds, float(cfg.get("evolution", {}).get("edit_penalty", 0.05)))
    summary = {
        "status": "ok",
        "success_rate": 1.0 if info.get("episode_length", 0.0) >= env.max_steps else 0.0,
        "mean_total_reward": float(score),
        "mean_episode_length": float(info.get("episode_length", 0.0)),
        "mean_forward_distance": float(info.get("forward_distance", 0.0)),
        "mean_height": float(info.get("mean_height", 0.0)),
        "edit_count": len(genome.edits),
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print("eval_done", summary)


if __name__ == "__main__":
    main()
