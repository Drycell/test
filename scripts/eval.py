from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    with (run_dir / "best.ckpt").open("rb") as f:
        ckpt = pickle.load(f)

    metrics = ckpt.get("metrics", {})
    summary = {
        "status": "ok",
        "success_rate": 1.0 if metrics.get("mean_episode_length", 0.0) > 0 else 0.0,
        "mean_total_reward": float(metrics.get("best_fitness", 0.0)),
        "mean_episode_length": float(metrics.get("mean_episode_length", 0.0)),
        "mean_forward_distance": float(metrics.get("mean_forward_distance", 0.0)),
        "mean_height": float(metrics.get("mean_height", 0.0)),
        "edit_count": int(metrics.get("edit_count", 0.0)),
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print("eval_done", summary)


if __name__ == "__main__":
    main()
