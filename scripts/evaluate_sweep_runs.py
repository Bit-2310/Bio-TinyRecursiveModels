#!/usr/bin/env python
"""Evaluate all Hydra sweep runs and save ClinVar metrics."""
from __future__ import annotations

import argparse
import subprocess
import yaml
from pathlib import Path
import re

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("checkpoints/Clinvar_trm-ACT-torch"),
        help="Root directory containing sweep run folders.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("outputs/clinvar_config_eval.yaml"),
        help="Base evaluation config to clone per run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for evaluation (cpu or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = yaml.safe_load(args.base_config.read_text())
    pattern = re.compile(r"arch.L_cycles=(\d+),arch.L_layers=(\d+),arch.hidden_size=(\d+),lr=([0-9.]+e?-?[0-9]*)")
    run_dirs = sorted(args.root.glob("arch.L_cycles=*"))

    for run_dir in run_dirs:
        match = pattern.fullmatch(run_dir.name)
        if not match:
            print(f"Skipping {run_dir}")
            continue

        L_cycles, L_layers, hidden_size, lr = match.groups()
        cfg = yaml.safe_load(args.base_config.read_text())
        cfg["arch"]["L_cycles"] = int(L_cycles)
        cfg["arch"]["L_layers"] = int(L_layers)
        cfg["arch"]["hidden_size"] = int(hidden_size)
        cfg["lr"] = float(lr)

        config_path = run_dir / "eval_config.yaml"
        config_path.write_text(yaml.safe_dump(cfg))

        checkpoints = sorted(run_dir.glob("step_*"))
        if not checkpoints:
            print(f"No checkpoints in {run_dir} (skipping)")
            continue

        checkpoint = checkpoints[-1]
        metrics_path = run_dir / "ClinVarEvaluator_metrics.json"
        if metrics_path.exists():
            print(f"Metrics already present for {run_dir.name}")
            continue

        cmd = [
            "python",
            "tools/evaluate_clinvar_checkpoint.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--output",
            str(metrics_path),
        ]
        print(f"Evaluating {run_dir.name} on {args.device}...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
