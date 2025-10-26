#!/usr/bin/env python
"""Create a heatmap of ROC AUC for ClinVar sweep results."""
from __future__ import annotations

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="sweep_summary.csv", help="Sweep summary CSV produced by analyze_sweep.py")
    parser.add_argument("--output", type=str, default="docs/figures/clinvar_sweep_heatmap.png", help="Output image path")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    pivot = df.pivot_table(values="roc_auc", index="hidden_size", columns=["L_layers","L_cycles","lr"])
    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title("ClinVar Sweep ROC AUC")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved sweep heatmap to {args.output}")

if __name__ == "__main__":
    main()
