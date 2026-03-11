from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default="testOut")
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    run = Path(args.run_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in ["main.html", "style.css", "vis.js"]:
        (out / f).write_text((Path("viz") / f).read_text())

    traj = np.load(run / "trajectory_best.npz")["observations"]
    step = max(1, int(np.ceil(traj.shape[0] / max(1, args.max_frames))))
    sampled = traj[::step].tolist()

    generation_csv = run / "generation_summary.csv"
    fitness_curve: list[float] = []
    if generation_csv.exists():
        df = pd.read_csv(generation_csv)
        if "best_fitness" in df.columns:
            fitness_curve = df["best_fitness"].astype(float).tolist()

    summary = {
        "steps": int(traj.shape[0]),
        "obs_dim": int(traj.shape[1]) if traj.ndim > 1 else 0,
        "eval": json.loads((run / "eval_summary.json").read_text()) if (run / "eval_summary.json").exists() else {},
    }
    payload = {"summary": summary, "fitness_curve": fitness_curve, "trajectory": sampled}
    (out / "run_data.json").write_text(json.dumps(payload))
    print("export_viz_done", out)


if __name__ == "__main__":
    main()
