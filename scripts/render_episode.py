from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    arr = np.load(run_dir / "trajectory_best.npz")["observations"]
    out = run_dir / "trajectory_preview.txt"
    out.write_text(f"steps={arr.shape[0]} obs_dim={arr.shape[1] if arr.ndim>1 else 0}\n")
    print("render_episode_done", out)


if __name__ == "__main__":
    main()
