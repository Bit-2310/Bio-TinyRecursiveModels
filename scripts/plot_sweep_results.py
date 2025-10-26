#!/usr/bin/env python
"""Create a heatmap of ROC AUC for ClinVar sweep results."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DEFAULT_OUTPUT = Path("docs/figures/clinvar_sweep_heatmap.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("sweep_summary.csv"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--palette",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name (diverging palettes like 'coolwarm' or 'RdYlBu_r' work well).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    # Bring columns into a tidy string for readability on the axis
    df["combo"] = df.apply(
        lambda r: f"{int(r['L_layers'])}-{int(r['L_cycles'])}-{r['lr']:.4f}".rstrip("0").rstrip("."),
        axis=1,
    )
    pivot = (
        df.pivot_table(values="roc_auc", index="hidden_size", columns="combo")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap=args.palette,
        vmin=df["roc_auc"].min(),
        vmax=df["roc_auc"].max(),
        cbar_kws={"label": "ROC AUC"},
    )
    heatmap.set_ylabel("hidden_size")
    heatmap.set_xlabel("L_layers – L_cycles – lr")
    plt.title("ClinVar Sweep ROC AUC")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220)
    plt.close()
    print(f"Saved sweep heatmap to {args.output}")


if __name__ == "__main__":
    main()
