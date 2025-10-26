#!/usr/bin/env python
"""Create a heatmap of ROC AUC for ClinVar sweep results."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_OUTPUT = Path("docs/figures/clinvar_sweep_heatmap.png")

PALETTES = {
    "viridis": "viridis",
    "redblue": "RdBu_r",
    "reds": "Reds",
    "custom": ["#4575b4", "#74add1", "#fdae61", "#d73027"]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("sweep_summary.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--palette", type=str, default="RdBu_r", help="Matplotlib colormap name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    pivot = df.pivot_table(values="roc_auc", index="hidden_size", columns=["L_layers", "L_cycles", "lr"])

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap=args.palette)
    plt.title("ClinVar Sweep ROC AUC")
    plt.ylabel("hidden_size")
    plt.xlabel("(L_layers, L_cycles, lr)")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    plt.close()
    print(f"Saved sweep heatmap to {args.output}")


if __name__ == "__main__":
    main()
