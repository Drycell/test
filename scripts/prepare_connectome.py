from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=False, default="data/mock_connectome")
    parser.add_argument("--dst", required=False, default="data/mock_connectome")
    args = parser.parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["neurons.csv", "chemical_synapses.csv", "gap_junctions.csv"]:
        (dst / name).write_text((src / name).read_text())
    print("prepare_connectome_done", dst)


if __name__ == "__main__":
    main()
