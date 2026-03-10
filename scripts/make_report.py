from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_run(run_dir: Path) -> dict:
    eval_summary = json.loads((run_dir / "eval_summary.json").read_text())
    best_genome = json.loads((run_dir / "best_genome.json").read_text())
    baseline = json.loads((run_dir / "baseline_summary.json").read_text()) if (run_dir / "baseline_summary.json").exists() else {}
    return {"eval": eval_summary, "edit_count": len(best_genome.get("edits", [])), "baseline": baseline}


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


def _plot_learning_curve(run_dir: Path) -> None:
    df = pd.read_csv(run_dir / "generation_summary.csv")
    plt.figure(figsize=(6, 4))
    plt.plot(df["generation"], df["best_fitness"]) ; plt.xlabel("generation"); plt.ylabel("best_fitness"); plt.tight_layout()
    plt.savefig(run_dir / "fig_learning_curve.png"); plt.close()


def _plot_edit_hist(run_dir: Path) -> None:
    g = json.loads((run_dir / "best_genome.json").read_text())
    ops = [e.get("op", "unknown") for e in g.get("edits", [])]
    counts = {k: ops.count(k) for k in sorted(set(ops))} or {"none": 0}
    plt.figure(figsize=(6, 4))
    plt.bar(list(counts.keys()), list(counts.values())); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(run_dir / "fig_edit_histogram.png"); plt.close()


def _plot_baseline_compare(run_dir: Path, summary: dict) -> None:
    baseline = summary.get("baseline", {})
    labels = ["pd", "fixed", "evolved"]
    values = [baseline.get("pd_mean_total_reward", 0.0), baseline.get("fixed_readout_mean_total_reward", 0.0), summary["eval"].get("mean_total_reward", 0.0)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values); plt.tight_layout(); plt.savefig(run_dir / "fig_best_vs_baseline.png"); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    summary = summarize_run(run_dir)
    edit_table = export_best_edit_table(run_dir)
    _plot_learning_curve(run_dir)
    _plot_edit_hist(run_dir)
    _plot_baseline_compare(run_dir, summary)

    md = "\n".join([
        "# Run Report", "",
        f"- status: {summary['eval'].get('status', 'n/a')}",
        f"- mean_total_reward: {summary['eval'].get('mean_total_reward', 'n/a')}",
        f"- mean_episode_length: {summary['eval'].get('mean_episode_length', 'n/a')}",
        f"- edit_count: {summary['edit_count']}",
        f"- edit_table: `{edit_table.name}`",
        "- figures: `fig_learning_curve.png`, `fig_best_vs_baseline.png`, `fig_edit_histogram.png`",
    ])
    (run_dir / "report.md").write_text(md)
    print("report_done", run_dir / "report.md")


if __name__ == "__main__":
    main()
