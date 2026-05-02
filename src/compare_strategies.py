from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .validate_profit import backtest_predictions


DEFAULT_CANDIDATES = [
    {
        "name": "base_unconstrained",
        "pred_csv": "outputs/val_predictions.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_base_unconstrained.csv",
    },
    {
        "name": "nwp_unconstrained",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_unconstrained.csv",
    },
    {
        "name": "nwp_c0_55",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 55,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_c0_55.csv",
    },
    {
        "name": "nwp_c0_55_d72_88",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 55,
        "discharge_start_min": 72,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_c0_55_d72_88.csv",
    },
    {
        "name": "nwp_threshold_500",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 500.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_unconstrained_t500.csv",
    },
    {
        "name": "nwp_threshold_1000",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 1000.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_unconstrained_t1000.csv",
    },
    {
        "name": "nwp_threshold_2000",
        "pred_csv": "outputs/val_predictions_nwp.csv",
        "threshold": 2000.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_unconstrained_t2000.csv",
    },
    {
        "name": "nwp_bias_unconstrained",
        "pred_csv": "outputs/val_predictions_nwp_bias.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_bias.csv",
    },
    {
        "name": "nwp_exact_bias_unconstrained",
        "pred_csv": "outputs/val_predictions_nwp_exact_bias.csv",
        "threshold": 0.0,
        "charge_start_min": 0,
        "charge_start_max": 80,
        "discharge_start_min": 8,
        "discharge_start_max": 88,
        "submission_csv": "outputs/output_nwp_exact_bias.csv",
    },
]


def parse_candidate(text: str) -> Dict[str, object]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) not in {7, 8}:
        raise ValueError(
            "--candidate format: name,pred_csv,threshold,charge_min,charge_max,discharge_min,discharge_max[,submission_csv]"
        )
    candidate = {
        "name": parts[0],
        "pred_csv": parts[1],
        "threshold": float(parts[2]),
        "charge_start_min": int(parts[3]),
        "charge_start_max": int(parts[4]),
        "discharge_start_min": int(parts[5]),
        "discharge_start_max": int(parts[6]),
    }
    if len(parts) == 8:
        candidate["submission_csv"] = parts[7]
    return candidate


def compare_candidates(candidates: List[Dict[str, object]], pred_col: str, true_col: str) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        pred_csv = str(candidate["pred_csv"])
        if not Path(pred_csv).exists():
            rows.append(
                {
                    "name": candidate["name"],
                    "pred_csv": pred_csv,
                    "available": False,
                    "error": "prediction file not found",
                }
            )
            continue

        df = pd.read_csv(pred_csv)
        try:
            summary, _ = backtest_predictions(
                df,
                threshold=float(candidate["threshold"]),
                pred_col=pred_col,
                true_col=true_col,
                charge_start_min=int(candidate["charge_start_min"]),
                charge_start_max=int(candidate["charge_start_max"]),
                discharge_start_min=int(candidate["discharge_start_min"]),
                discharge_start_max=int(candidate["discharge_start_max"]),
            )
            rows.append(
                {
                    "name": candidate["name"],
                    "pred_csv": pred_csv,
                    "available": True,
                    "error": "",
                    **candidate,
                    **summary,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": candidate["name"],
                    "pred_csv": pred_csv,
                    "available": True,
                    "error": str(exc),
                    **candidate,
                }
            )

    out = pd.DataFrame(rows)
    sort_cols = [col for col in ["available", "avg_profit", "capture_ratio"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation profit metrics for strategy candidates.")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--pred-col", default="pred_price")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--output", default="outputs/strategy_compare.csv")
    args = parser.parse_args()

    candidates = [parse_candidate(item) for item in args.candidate] if args.candidate else DEFAULT_CANDIDATES
    summary = compare_candidates(candidates, pred_col=args.pred_col, true_col=args.true_col)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved_strategy_compare={output_path}")


if __name__ == "__main__":
    main()
