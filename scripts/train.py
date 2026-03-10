from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

from wormwing.connectome.loader import load_connectome
from wormwing.controllers.ctrnn import ConnectomeCTRNN
from wormwing.envs.winged_worm_3d import WingedWorm3DEnv
from wormwing.evolution.structure_only import run_structure_only


def _resolve_connectome_dir(cfg: dict) -> str:
    exp = cfg.get("experiment", {})
    if exp.get("connectome_mode", "mock") == "real":
        return str(exp.get("real_data_dir", "data/real_connectome"))
    return "data/mock_connectome"


def _write_manifest(run_dir: Path, cfg: dict) -> None:
    manifest = {
        "experiment_name": cfg.get("experiment", {}).get("name", "exp001"),
        "connectome_mode": cfg.get("experiment", {}).get("connectome_mode", "mock"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seed_list": cfg.get("evolution", {}).get("train_eval_seeds", []),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    run_dir = Path("runs/latest")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _write_manifest(run_dir, cfg)

    conn = load_connectome(_resolve_connectome_dir(cfg))
    controller = ConnectomeCTRNN(
        conn,
        dt=float(cfg.get("env", {}).get("control_dt", 0.02)),
        tau_init=float(cfg.get("controller", {}).get("tau_init", 1.0)),
        bias_init=float(cfg.get("controller", {}).get("bias_init", 0.0)),
    )
    env = WingedWorm3DEnv()

    run_structure_only(
        controller,
        env,
        run_dir,
        generations=int(cfg.get("evolution", {}).get("generations", 4)),
        population_size=int(cfg.get("evolution", {}).get("population_size", 8)),
        max_edits=int(cfg.get("evolution", {}).get("max_edits", 8)),
        seed=int(cfg.get("seeds", {}).get("master_seed", 0)),
    )
    print("train_done", run_dir)


if __name__ == "__main__":
    main()
