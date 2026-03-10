from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_run(run_dir: Path) -> dict:
    eval_summary = json.loads((run_dir / "eval_summary.json").read_text())
    best_genome = json.loads((run_dir / "best_genome.json").read_text())
    return {
        "eval": eval_summary,
        "edit_count": len(best_genome.get("edits", [])),
    }


def compare_runs(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for rd in run_dirs:
        s = summarize_run(rd)
        rows.append({"run_dir": str(rd), "mean_total_reward": s["eval"].get("mean_total_reward", 0.0), "edit_count": s["edit_count"]})
    return rows


def export_best_edit_table(run_dir: Path) -> Path:
    genome = json.loads((run_dir / "best_genome.json").read_text())
    out = run_dir / "edit_frequency.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "op", "src", "dst", "value"])
        for i, e in enumerate(genome.get("edits", [])):
            writer.writerow([i, e.get("op"), e.get("src"), e.get("dst"), e.get("value")])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = summarize_run(run_dir)
    edit_table = export_best_edit_table(run_dir)

    md = "\n".join(
        [
            "# Run Report",
            "",
            f"- status: {summary['eval'].get('status', 'n/a')}",
            f"- mean_total_reward: {summary['eval'].get('mean_total_reward', 'n/a')}",
            f"- mean_episode_length: {summary['eval'].get('mean_episode_length', 'n/a')}",
            f"- edit_count: {summary['edit_count']}",
            f"- edit_table: `{edit_table.name}`",
        ]
    )
    (run_dir / "report.md").write_text(md)
    print("report_done", run_dir / "report.md")


if __name__ == "__main__":
    main()
