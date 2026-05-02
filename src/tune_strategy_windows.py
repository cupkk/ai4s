from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .storage_optimizer import generate_strategy
from .validate_profit import backtest_predictions, parse_threshold_grid


def _parse_int_grid(text: str) -> list[int]:
    values: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError("integer grid is empty")
    return values


def _iter_candidates(
    charge_start_min_grid: Iterable[int],
    charge_start_max_grid: Iterable[int],
    discharge_start_min_grid: Iterable[int],
    discharge_start_max_grid: Iterable[int],
    threshold_grid: Iterable[float],
):
    for c_min in charge_start_min_grid:
        for c_max in charge_start_max_grid:
            if c_max < c_min or c_max > 80:
                continue
            for d_min in discharge_start_min_grid:
                if d_min < 8:
                    continue
                for d_max in discharge_start_max_grid:
                    if d_max < d_min or d_max > 88:
                        continue
                    # Need room for an 8-slot charge block before discharge.
                    if c_min + 8 > d_max:
                        continue
                    for threshold in threshold_grid:
                        yield {
                            "threshold": float(threshold),
                            "charge_start_min": int(c_min),
                            "charge_start_max": int(c_max),
                            "discharge_start_min": int(d_min),
                            "discharge_start_max": int(d_max),
                        }


def search_windows(
    pred_df: pd.DataFrame,
    pred_col: str,
    true_col: str,
    charge_start_min_grid: list[int],
    charge_start_max_grid: list[int],
    discharge_start_min_grid: list[int],
    discharge_start_max_grid: list[int],
    threshold_grid: list[float],
) -> pd.DataFrame:
    rows = []
    for candidate in _iter_candidates(
        charge_start_min_grid,
        charge_start_max_grid,
        discharge_start_min_grid,
        discharge_start_max_grid,
        threshold_grid,
    ):
        summary, _ = backtest_predictions(
            pred_df,
            pred_col=pred_col,
            true_col=true_col,
            **candidate,
        )
        rows.append({**candidate, **summary})

    if not rows:
        raise ValueError("window search produced no valid candidates")
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["avg_profit", "capture_ratio", "loss_days", "window_hit_2_rate"],
        ascending=[False, False, True, False],
    )


def _candidate_name(row: pd.Series) -> str:
    return (
        f"c{int(row['charge_start_min'])}_{int(row['charge_start_max'])}"
        f"_d{int(row['discharge_start_min'])}_{int(row['discharge_start_max'])}"
        f"_t{int(row['threshold'])}"
    )


def write_test_candidates(
    search_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    top_k: int,
    price_col: str,
) -> pd.DataFrame:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, row in search_df.head(top_k).iterrows():
        name = _candidate_name(row)
        output_path = output_dir / f"{prefix}_{name}.csv"
        meta_path = output_dir / f"{prefix}_{name}_meta.csv"
        out, meta = generate_strategy(
            test_df,
            threshold=float(row["threshold"]),
            price_col=price_col or None,
            charge_start_min=int(row["charge_start_min"]),
            charge_start_max=int(row["charge_start_max"]),
            discharge_start_min=int(row["discharge_start_min"]),
            discharge_start_max=int(row["discharge_start_max"]),
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
    parser = argparse.ArgumentParser(description="Search charge/discharge window constraints on validation predictions.")
    parser.add_argument("--pred-csv", required=True)
    parser.add_argument("--pred-col", default="pred_price")
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--threshold-grid", default="0,100,200,500,1000")
    parser.add_argument("--charge-start-min-grid", default="0")
    parser.add_argument("--charge-start-max-grid", default="35,40,45,50,55,60,65,70,75,80")
    parser.add_argument("--discharge-start-min-grid", default="56,60,64,68,72,76,80")
    parser.add_argument("--discharge-start-max-grid", default="80,84,88")
    parser.add_argument("--output", default="outputs/window_constraint_search.csv")
    parser.add_argument("--test-price-csv", default="")
    parser.add_argument("--test-price-col", default="")
    parser.add_argument("--submission-prefix", default="output_window")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-manifest", default="outputs/window_constraint_candidates.csv")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    result = search_windows(
        pred_df=pred_df,
        pred_col=args.pred_col,
        true_col=args.true_col,
        charge_start_min_grid=_parse_int_grid(args.charge_start_min_grid),
        charge_start_max_grid=_parse_int_grid(args.charge_start_max_grid),
        discharge_start_min_grid=_parse_int_grid(args.discharge_start_min_grid),
        discharge_start_max_grid=_parse_int_grid(args.discharge_start_max_grid),
        threshold_grid=parse_threshold_grid(args.threshold_grid),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.head(20).to_string(index=False))
    print(f"saved_window_search={output_path}")

    if args.test_price_csv:
        manifest = write_test_candidates(
            result,
            pd.read_csv(args.test_price_csv),
            output_path.parent,
            args.submission_prefix,
            args.top_k,
            args.test_price_col,
        )
        manifest_path = Path(args.candidate_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        print(manifest.to_string(index=False))
        print(f"saved_window_candidates={manifest_path}")


if __name__ == "__main__":
    main()
