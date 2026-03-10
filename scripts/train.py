from __future__ import annotations

import argparse
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from wormwing.connectome.loader import load_connectome
from wormwing.connectome.types import RunManifest
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import EnvConfig, WingedWorm3DEnv
from wormwing.evolution.structure_only import run_structure_only
from wormwing.experiments.baselines import run_fixed_readout_baseline, run_pd_baseline


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _resolve_connectome_dir(cfg: dict) -> str:
    exp = cfg.get("experiment", {})
    return str(exp.get("real_data_dir", "data/real_connectome")) if exp.get("connectome_mode", "mock") == "real" else "data/mock_connectome"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    run_dir = Path("runs/latest")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    manifest = RunManifest(
        resolved_config=cfg,
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        platform=platform.platform(),
        seed_list=list(cfg.get("evolution", {}).get("train_eval_seeds", [])),
        connectome_mode=cfg.get("experiment", {}).get("connectome_mode", "mock"),
        experiment_name=cfg.get("experiment", {}).get("name", "exp001"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    (run_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, default=str))

    conn = load_connectome(_resolve_connectome_dir(cfg))
    controller = ConnectomeCTRNN(conn, dt=float(cfg.get("env", {}).get("control_dt", 0.02)), tau_init=float(cfg.get("controller", {}).get("tau_init", 1.0)), bias_init=float(cfg.get("controller", {}).get("bias_init", 0.0)))
    env_cfg = EnvConfig(**{k: v for k, v in cfg.get("env", {}).items() if k in EnvConfig.__annotations__})
    env = WingedWorm3DEnv(env_cfg)

    seeds = list(cfg.get("evolution", {}).get("train_eval_seeds", [0, 1, 2, 3]))
    pd_score = float(sum(run_pd_baseline(env, s) for s in seeds) / len(seeds))
    fixed_score = float(sum(run_fixed_readout_baseline(controller, env, s) for s in seeds) / len(seeds))
    (run_dir / "baseline_summary.json").write_text(json.dumps({"pd_mean_total_reward": pd_score, "fixed_readout_mean_total_reward": fixed_score}, indent=2))

    run_structure_only(
        controller, env, run_dir,
        generations=int(cfg.get("evolution", {}).get("generations", 4)),
        population_size=int(cfg.get("evolution", {}).get("population_size", 8)),
        max_edits=int(cfg.get("evolution", {}).get("max_edits", 8)),
        seed=int(cfg.get("seeds", {}).get("master_seed", 0)),
        elite_count=int(cfg.get("evolution", {}).get("elite_count", 2)),
        mutation_cfg=cfg.get("evolution", {}).get("mutation", {}),
        train_eval_seeds=seeds,
        edit_penalty=float(cfg.get("evolution", {}).get("edit_penalty", 0.05)),
    )
    print("train_done", run_dir)


if __name__ == "__main__":
    main()
