#!/usr/bin/env python
"""Aggregate ClinVar sweep results and report the best run."""
from __future__ import annotations

import json
from pathlib import Path

import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

SWEEP_ROOT = Path("checkpoints/Clinvar_trm-ACT-torch")
OUTPUT_FIG = Path("docs/figures/clinvar_sweep_heatmap.png")


def load_run_metrics(run_dir: Path) -> dict:
    metrics_file = run_dir / "ClinVarEvaluator_metrics.json"
    if metrics_file.exists():
        metrics = json.loads(metrics_file.read_text())
    else:
        metrics = {}
    cfg = yaml.safe_load((run_dir / "all_config.yaml").read_text())

    return {
        "run": run_dir.name,
        "roc_auc": metrics.get("ClinVar/roc_auc"),
        "accuracy": metrics.get("ClinVar/accuracy"),
        "hidden_size": cfg["arch"]["hidden_size"],
        "L_layers": cfg["arch"]["L_layers"],
        "L_cycles": cfg["arch"]["L_cycles"],
        "lr": cfg["lr"],
        "checkpoint": str(run_dir / "step_1560"),
        "config": str(run_dir / "all_config.yaml"),
    }


def main() -> None:
    runs = sorted([d for d in SWEEP_ROOT.glob("arch.L_cycles=*") if d.is_dir()])
    if not runs:
        print("No sweep results found under", SWEEP_ROOT)
        return

    records = [load_run_metrics(run) for run in runs if (run / "ClinVarEvaluator_metrics.json").exists()]
    if not records:
        print("No completed runs found (missing metrics).")
        return
    finished = [r for r in records if r["roc_auc"] is not None and r["accuracy"] is not None]
    if not finished:
        print("No runs reported ClinVar metrics yet.")
        return

    best = max(finished, key=lambda r: r["roc_auc"])
    print('Best run:')
    for k, v in best.items():
        print(f'  {k}: {v}')

    with open('sweep_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=finished[0].keys())
        writer.writeheader()
        writer.writerows(finished)
    print("Summary saved to sweep_summary.csv")

    df = pd.DataFrame(finished)
    try:
        plt.figure(figsize=(10, 6))
        pivot = df.pivot_table(index="hidden_size", columns=["L_layers", "L_cycles", "lr"], values="roc_auc")
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
        plt.title("ClinVar Sweep ROC AUC")
        plt.tight_layout()
        OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_FIG, dpi=200)
        plt.close()
        print(f"Saved heatmap to {OUTPUT_FIG}")
    except Exception as exc:  # pylint: disable=broad-except
        print("Unable to render heatmap:", exc)


if __name__ == "__main__":
    main()
