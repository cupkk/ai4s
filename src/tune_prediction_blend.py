from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .storage_optimizer import generate_strategy, infer_price_column
from .validate_profit import backtest_predictions, parse_threshold_grid


def _parse_csv_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if len(values) < 2:
        raise ValueError("at least two CSV files are required")
    return values


def _parse_float_list(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("float list is empty")
    return values


def _blend_two_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    weight_second: float,
    pred_col_first: str,
    pred_col_second: str,
    output_col: str,
    true_col: str | None = None,
) -> pd.DataFrame:
    left = first[["times", pred_col_first] + ([true_col] if true_col and true_col in first.columns else [])].copy()
    right = second[["times", pred_col_second]].copy()
    left["times"] = pd.to_datetime(left["times"])
    right["times"] = pd.to_datetime(right["times"])
    merged = left.merge(right, on="times", how="inner", suffixes=("_first", "_second"))
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError(
            f"blend inputs are not aligned: first={len(left)}, second={len(right)}, merged={len(merged)}"
        )

    p1 = merged[f"{pred_col_first}_first"].to_numpy(dtype=float)
    p2 = merged[f"{pred_col_second}_second"].to_numpy(dtype=float)
    merged[output_col] = (1.0 - weight_second) * p1 + weight_second * p2
    keep_cols = ["times"]
    if true_col and true_col in merged.columns:
        keep_cols.append(true_col)
    keep_cols.append(output_col)
    return merged[keep_cols].copy()


def tune_two_model_blend(
    val_first: pd.DataFrame,
    val_second: pd.DataFrame,
    weights: Sequence[float],
    thresholds: Sequence[float],
    true_col: str,
    pred_col: str = "pred_price",
) -> pd.DataFrame:
    rows = []
    pred_first = infer_price_column(val_first, pred_col)
    pred_second = infer_price_column(val_second, pred_col)
    for weight in weights:
        blended = _blend_two_frames(
            val_first,
            val_second,
            weight,
            pred_first,
            pred_second,
            output_col="pred_price",
            true_col=true_col,
        )
        y_true = blended[true_col].to_numpy(dtype=float)
        y_pred = blended["pred_price"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        best_summary = None
        for threshold in thresholds:
            summary, _ = backtest_predictions(
                blended,
                threshold=threshold,
                pred_col="pred_price",
                true_col=true_col,
            )
            if best_summary is None or summary["avg_profit"] > best_summary["avg_profit"]:
                best_summary = summary
        assert best_summary is not None
        rows.append(
            {
                "weight_second": float(weight),
                "rmse": rmse,
                "mae": mae,
                "best_threshold": float(best_summary["threshold"]),
                "avg_profit": float(best_summary["avg_profit"]),
                "total_profit": float(best_summary["total_profit"]),
                "traded_days": int(best_summary["traded_days"]),
                "loss_days": int(best_summary["loss_days"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["avg_profit", "weight_second"], ascending=[False, False])


def emit_test_blend(
    test_first: pd.DataFrame,
    test_second: pd.DataFrame,
    weight_second: float,
    prediction_output: str,
    submission_output: str,
    meta_output: str,
    submission_threshold: float,
) -> None:
    pred_first = infer_price_column(test_first)
    pred_second = infer_price_column(test_second)
    blended = _blend_two_frames(
        test_first,
        test_second,
        weight_second,
        pred_first,
        pred_second,
            output_col="实时价格",
    )
    pred_path = Path(prediction_output)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    blended.to_csv(pred_path, index=False)

    submission, meta = generate_strategy(
        blended,
        threshold=submission_threshold,
        price_col=None,
    )
    sub_path = Path(submission_output)
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(sub_path, index=False)
    meta_path = Path(meta_output)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(meta_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune and emit a two-model prediction blend using storage-profit validation."
    )
    parser.add_argument("--val-first", required=True)
    parser.add_argument("--val-second", required=True)
    parser.add_argument("--test-first", required=True)
    parser.add_argument("--test-second", required=True)
    parser.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,0.9,1.0")
    parser.add_argument("--threshold-grid", default="0,1000,2000,5000,10000,20000")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--summary-output", default="outputs/prediction_blend_profit_tuning.csv")
    parser.add_argument("--emit-weights", default="0.5,0.75,0.9,1.0")
    parser.add_argument("--prediction-prefix", default="outputs/test_predictions_blend")
    parser.add_argument("--submission-prefix", default="outputs/output_blend")
    parser.add_argument("--meta-prefix", default="outputs/blend_meta")
    parser.add_argument("--submission-threshold", type=float, default=0.0)
    args = parser.parse_args()

    val_first = pd.read_csv(args.val_first)
    val_second = pd.read_csv(args.val_second)
    weights = _parse_float_list(args.weights)
    thresholds = parse_threshold_grid(args.threshold_grid)
    summary = tune_two_model_blend(
        val_first,
        val_second,
        weights=weights,
        thresholds=thresholds,
        true_col=args.true_col,
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"saved_summary={summary_path}")

    test_first = pd.read_csv(args.test_first)
    test_second = pd.read_csv(args.test_second)
    for weight in _parse_float_list(args.emit_weights):
        suffix = f"w{int(round(weight * 100)):03d}"
        emit_test_blend(
            test_first,
            test_second,
            weight_second=weight,
            prediction_output=f"{args.prediction_prefix}_{suffix}.csv",
            submission_output=f"{args.submission_prefix}_{suffix}.csv",
            meta_output=f"{args.meta_prefix}_{suffix}.csv",
            submission_threshold=args.submission_threshold,
        )
        print(f"emitted_weight={weight:.3f}, suffix={suffix}")


if __name__ == "__main__":
    main()
