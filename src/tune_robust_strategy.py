from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .make_robust_submission import generate_robust_strategy
from .storage_optimizer import evaluate_power, optimize_one_day
from .validate_profit import parse_threshold_grid


def _complete_days_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["times"] = pd.to_datetime(out["times"])
    date_key = out["times"].dt.date
    counts = out.groupby(date_key, sort=True).size()
    complete_dates = set(counts[counts == 96].index)
    if not complete_dates:
        raise ValueError("no complete 96-point days found")
    return out[date_key.isin(complete_dates)].copy()


def _parse_float_grid(text: str) -> list[float]:
    values: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("float grid is empty")
    return values


def _score_strategy(
    pred_df: pd.DataFrame,
    true_col: str,
    lambda_uncertainty: float,
    threshold: float,
    price_col: str,
    uncertainty_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
) -> dict:
    complete_pred_df = _complete_days_only(pred_df)
    strategy, meta = generate_robust_strategy(
        complete_pred_df,
        lambda_uncertainty=lambda_uncertainty,
        threshold=threshold,
        price_col=price_col or None,
        uncertainty_col=uncertainty_col,
        charge_start_min=charge_start_min,
        charge_start_max=charge_start_max,
        discharge_start_min=discharge_start_min,
        discharge_start_max=discharge_start_max,
    )
    joined = strategy.merge(complete_pred_df[["times", true_col]], on="times", how="left")
    if joined[true_col].isna().any():
        raise ValueError("robust strategy times could not be matched to true prices")

    day_rows = []
    joined["__date__"] = pd.to_datetime(joined["times"]).dt.date
    for date, group in joined.groupby("__date__", sort=True):
        profit = evaluate_power(group[true_col], group["power"])
        traded = bool((group["power"] != 0).any())
        oracle = optimize_one_day(
            group[true_col].to_numpy(dtype=float),
            threshold=0.0,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        oracle_profit = max(0.0, float(oracle.best_profit))
        day_rows.append(
            {
                "date": str(date),
                "profit": float(profit),
                "traded": traded,
                "oracle_profit": oracle_profit,
                "regret": oracle_profit - float(profit),
                "capture_ratio": float(profit) / oracle_profit if oracle_profit > 0 else np.nan,
            }
        )

    day_df = pd.DataFrame(day_rows)
    oracle_total = float(day_df["oracle_profit"].sum())
    total_profit = float(day_df["profit"].sum())
    return {
        "lambda_uncertainty": float(lambda_uncertainty),
        "threshold": float(threshold),
        "days": int(len(day_df)),
        "avg_profit": float(day_df["profit"].mean()),
        "total_profit": total_profit,
        "avg_oracle_profit": float(day_df["oracle_profit"].mean()),
        "oracle_total_profit": oracle_total,
        "capture_ratio": total_profit / oracle_total if oracle_total > 0 else np.nan,
        "avg_regret": float(day_df["regret"].mean()),
        "traded_days": int(day_df["traded"].sum()),
        "loss_days": int((day_df["profit"] < 0).sum()),
    }


def search_robust(
    pred_df: pd.DataFrame,
    true_col: str,
    lambdas: Iterable[float],
    thresholds: Iterable[float],
    price_col: str,
    uncertainty_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
) -> pd.DataFrame:
    rows = []
    for lambda_uncertainty in lambdas:
        for threshold in thresholds:
            rows.append(
                _score_strategy(
                    pred_df=pred_df,
                    true_col=true_col,
                    lambda_uncertainty=lambda_uncertainty,
                    threshold=threshold,
                    price_col=price_col,
                    uncertainty_col=uncertainty_col,
                    charge_start_min=charge_start_min,
                    charge_start_max=charge_start_max,
                    discharge_start_min=discharge_start_min,
                    discharge_start_max=discharge_start_max,
                )
            )
    return pd.DataFrame(rows).sort_values(
        ["avg_profit", "capture_ratio", "loss_days"],
        ascending=[False, False, True],
    )


def _candidate_name(row: pd.Series) -> str:
    lam_text = str(row["lambda_uncertainty"]).replace(".", "p")
    return f"lambda{lam_text}_t{int(row['threshold'])}"


def write_test_candidates(
    search_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    top_k: int,
    price_col: str,
    uncertainty_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
) -> pd.DataFrame:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in search_df.head(top_k).iterrows():
        name = _candidate_name(row)
        output_path = output_dir / f"{prefix}_{name}.csv"
        meta_path = output_dir / f"{prefix}_{name}_meta.csv"
        out, meta = generate_robust_strategy(
            test_df,
            lambda_uncertainty=float(row["lambda_uncertainty"]),
            threshold=float(row["threshold"]),
            price_col=price_col or None,
            uncertainty_col=uncertainty_col,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        out.to_csv(output_path, index=False)
        meta.to_csv(meta_path, index=False)
        rows.append(
            {
                "candidate": f"{prefix}_{name}",
                "submission_csv": str(output_path),
                "meta_csv": str(meta_path),
                "validation_avg_profit": float(row["avg_profit"]),
                "validation_capture_ratio": float(row["capture_ratio"]),
                "validation_loss_days": int(row["loss_days"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune robust storage strategy using seed prediction uncertainty.")
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--uncertainty-col", default="")
    parser.add_argument("--lambda-grid", default="0,0.25,0.5,0.75,1,1.5,2,3")
    parser.add_argument("--threshold-grid", default="0,100,200,500,1000")
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=55)
    parser.add_argument("--discharge-start-min", type=int, default=72)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--output", default="outputs/robust_strategy_search.csv")
    parser.add_argument("--test-price-csv", default="")
    parser.add_argument("--submission-prefix", default="output_nwp_robust")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-manifest", default="outputs/robust_strategy_candidates.csv")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    result = search_robust(
        pred_df=pred_df,
        true_col=args.true_col,
        lambdas=_parse_float_grid(args.lambda_grid),
        thresholds=parse_threshold_grid(args.threshold_grid),
        price_col=args.price_col,
        uncertainty_col=args.uncertainty_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.head(20).to_string(index=False))
    print(f"saved_robust_search={output_path}")

    if args.test_price_csv:
        manifest = write_test_candidates(
            result,
            pd.read_csv(args.test_price_csv),
            output_path.parent,
            args.submission_prefix,
            args.top_k,
            args.price_col,
            args.uncertainty_col,
            args.charge_start_min,
            args.charge_start_max,
            args.discharge_start_min,
            args.discharge_start_max,
        )
        manifest_path = Path(args.candidate_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(manifest.to_string(index=False))
        print(f"saved_robust_candidates={manifest_path}")


if __name__ == "__main__":
    main()
