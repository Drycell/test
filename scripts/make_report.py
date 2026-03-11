from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from wormwing.connectome.loader import load_connectome


def summarize_run(run_dir: Path) -> dict:
    eval_summary = json.loads((run_dir / "eval_summary.json").read_text())
    best_genome = json.loads((run_dir / "best_genome.json").read_text())
    baseline = json.loads((run_dir / "baseline_summary.json").read_text()) if (run_dir / "baseline_summary.json").exists() else {}
    graph_summary = json.loads((run_dir / "best_graph_before_after.json").read_text()) if (run_dir / "best_graph_before_after.json").exists() else {}
    node_counts: dict[int, int] = {}
    for e in best_genome.get("edits", []):
        for key in ("src", "dst"):
            idx = int(e.get(key, -1))
            if idx >= 0:
                node_counts[idx] = node_counts.get(idx, 0) + 1
    return {
        "eval": eval_summary,
        "edit_count": len(best_genome.get("edits", [])),
        "baseline": baseline,
        "node_edit_frequency": node_counts,
        "graph_summary": graph_summary,
        "best_genome": best_genome,
    }


def _load_connectome_for_run(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config_resolved.yaml").read_text())
    exp = cfg.get("experiment", {})
    connectome_dir = "data/mock_connectome" if exp.get("connectome_mode", "mock") == "mock" else exp.get("real_data_dir", "data/real_connectome")
    return load_connectome(connectome_dir, normalize=True)


def _node_role(index: int, conn) -> str:
    if index in set(conn.virtual_sensor_indices) or index in set(conn.sensor_node_indices):
        return "sensor"
    if index in set(conn.virtual_motor_indices) or index in set(conn.motor_node_indices):
        return "motor"
    return "interneuron"


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
    plt.plot(df["generation"], df["best_fitness"])
    plt.xlabel("generation")
    plt.ylabel("best_fitness")
    plt.tight_layout()
    plt.savefig(run_dir / "fig_learning_curve.png")
    plt.close()


def _plot_edit_hist(run_dir: Path) -> None:
    g = json.loads((run_dir / "best_genome.json").read_text())
    ops = [e.get("op", "unknown") for e in g.get("edits", [])]
    counts = {k: ops.count(k) for k in sorted(set(ops))} or {"none": 0}
    plt.figure(figsize=(6, 4))
    plt.bar(list(counts.keys()), list(counts.values()))
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(run_dir / "fig_edit_histogram.png")
    plt.close()


def _plot_baseline_compare(run_dir: Path, summary: dict) -> None:
    baseline = summary.get("baseline", {})
    labels = ["pd", "fixed", "evolved"]
    values = [baseline.get("pd_mean_total_reward", 0.0), baseline.get("fixed_readout_mean_total_reward", 0.0), summary["eval"].get("mean_total_reward", 0.0)]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.tight_layout()
    plt.savefig(run_dir / "fig_best_vs_baseline.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    summary = summarize_run(run_dir)
    conn = _load_connectome_for_run(run_dir)
    edit_table = export_best_edit_table(run_dir)
    _plot_learning_curve(run_dir)
    _plot_edit_hist(run_dir)
    _plot_baseline_compare(run_dir, summary)

    role_counts = {"sensor": 0, "interneuron": 0, "motor": 0}
    for idx, cnt in summary.get("node_edit_frequency", {}).items():
        role_counts[_node_role(int(idx), conn)] += int(cnt)

    baseline = summary.get("baseline", {})
    md = "\n".join(
        [
            "# Run Report",
            "",
            "## Core Metrics",
            f"- status: {summary['eval'].get('status', 'n/a')}",
            f"- success_rate: {summary['eval'].get('success_rate', 'n/a')}",
            f"- mean_total_reward: {summary['eval'].get('mean_total_reward', 'n/a')}",
            f"- mean_episode_length: {summary['eval'].get('mean_episode_length', 'n/a')}",
            f"- edit_count: {summary['edit_count']}",
            "",
            "## Baseline vs Evolved",
            "| method | mean_total_reward |",
            "|---|---:|",
            f"| pd | {baseline.get('pd_mean_total_reward', 'n/a')} |",
            f"| fixed_readout | {baseline.get('fixed_readout_mean_total_reward', 'n/a')} |",
            f"| evolved | {summary['eval'].get('mean_total_reward', 'n/a')} |",
            "",
            "## Perturbation/Seed Robustness",
            f"- termination_reasons: {summary['eval'].get('termination_reasons', [])}",
            "",
            "## Edit Concentration",
            f"- role_counts: {role_counts}",
            f"- top_edited_nodes: {sorted(summary.get('node_edit_frequency', {}).items(), key=lambda kv: kv[1], reverse=True)[:8]}",
            f"- graph_summary: {summary.get('graph_summary', {})}",
            f"- edit_table: `{edit_table.name}`",
            "- figures: `fig_learning_curve.png`, `fig_best_vs_baseline.png`, `fig_edit_histogram.png`",
        ]
    )
    (run_dir / "report.md").write_text(md)
    print("report_done", run_dir / "report.md")


if __name__ == "__main__":
    main()
