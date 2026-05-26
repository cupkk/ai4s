from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .storage_optimizer import infer_price_column


DEFAULT_SCENARIO_PREFIX = "resid_scenario_"
DEFAULT_BASE_COL = "residual_base_price"
DEFAULT_SOURCE_MAP = "outputs/residual_scenario_source_map.csv"


@dataclass(frozen=True)
class ResidualLibrary:
    residuals: np.ndarray
    source_dates: list[str]
    prediction_cols: list[str]
    true_col: str


def detect_prediction_columns(
    df: pd.DataFrame,
    explicit: str = "",
    prefix: str = "pred_price_seed",
) -> list[str]:
    if explicit:
        columns = [item.strip() for item in explicit.split(",") if item.strip()]
    else:
        columns = [col for col in df.columns if col.startswith(prefix)]
        if not columns and "pred_price" in df.columns:
            columns = ["pred_price"]
    if not columns:
        raise ValueError("no prediction columns found; pass --prediction-cols")
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"prediction columns not found: {missing}")
    return columns


def build_daily_residual_library(
    validation_df: pd.DataFrame,
    true_col: str = "A",
    prediction_cols: Sequence[str] | None = None,
    day_length: int = 96,
) -> ResidualLibrary:
    if "times" not in validation_df.columns:
        raise ValueError("validation data missing required column: times")
    if true_col not in validation_df.columns:
        raise ValueError(f"validation data missing true column: {true_col}")
    cols = list(prediction_cols or detect_prediction_columns(validation_df))
    missing = [col for col in cols if col not in validation_df.columns]
    if missing:
        raise ValueError(f"validation data missing prediction columns: {missing}")

    df = validation_df.copy()
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    df["date"] = df["times"].dt.date.astype(str)

    residual_rows: list[np.ndarray] = []
    dates: list[str] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != int(day_length):
            continue
        base = group[cols].mean(axis=1).to_numpy(dtype=float)
        true = group[true_col].to_numpy(dtype=float)
        residual_rows.append(true - base)
        dates.append(str(date))

    if not residual_rows:
        raise ValueError("no complete daily residual vectors found")

    return ResidualLibrary(
        residuals=np.vstack(residual_rows),
        source_dates=dates,
        prediction_cols=cols,
        true_col=true_col,
    )


def generate_residual_scenarios(
    test_df: pd.DataFrame,
    library: ResidualLibrary,
    n_scenarios: int = 100,
    random_seed: int = 20260526,
    prediction_cols: Sequence[str] | None = None,
    price_col: str = "",
    scenario_prefix: str = DEFAULT_SCENARIO_PREFIX,
    base_col: str = DEFAULT_BASE_COL,
    day_length: int = 96,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "times" not in test_df.columns:
        raise ValueError("test data missing required column: times")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    cols = list(prediction_cols or library.prediction_cols)
    missing = [col for col in cols if col not in test_df.columns]
    if missing:
        raise ValueError(f"test data missing prediction columns: {missing}")
    if price_col and price_col not in test_df.columns:
        raise ValueError(f"test data missing price column: {price_col}")

    rng = np.random.default_rng(int(random_seed))
    df = test_df.copy()
    df["times"] = pd.to_datetime(df["times"])
    df = df.sort_values("times").reset_index(drop=True)
    df["date"] = df["times"].dt.date.astype(str)

    scenario_cols = [f"{scenario_prefix}{idx:03d}" for idx in range(int(n_scenarios))]
    output_days: list[pd.DataFrame] = []
    source_rows: list[dict[str, str | int]] = []

    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != int(day_length):
            raise ValueError(f"{date} must contain {day_length} rows, got {len(group)}")
        base = group[cols].mean(axis=1).to_numpy(dtype=float)
        sampled_idx = rng.integers(0, library.residuals.shape[0], size=int(n_scenarios))
        scenarios = base.reshape(1, -1) + library.residuals[sampled_idx]

        day_values: dict[str, np.ndarray] = {
            "times": group["times"].to_numpy(),
            base_col: base,
        }
        if price_col:
            day_values[price_col] = group[price_col].to_numpy(dtype=float)
        for col_idx, col in enumerate(scenario_cols):
            day_values[col] = scenarios[col_idx]
            source_rows.append(
                {
                    "date": str(date),
                    "scenario_col": col,
                    "scenario_index": int(col_idx),
                    "source_date": library.source_dates[int(sampled_idx[col_idx])],
                    "source_index": int(sampled_idx[col_idx]),
                }
            )
        day_out = pd.DataFrame(day_values)
        output_days.append(day_out)

    return pd.concat(output_days, ignore_index=True), pd.DataFrame(source_rows)


def scenario_diagnostics(
    scenario_df: pd.DataFrame,
    source_map: pd.DataFrame,
    scenario_cols: Sequence[str] | None = None,
    base_col: str = DEFAULT_BASE_COL,
) -> pd.DataFrame:
    if "times" not in scenario_df.columns:
        raise ValueError("scenario data missing required column: times")
    cols = list(scenario_cols or [col for col in scenario_df.columns if col.startswith(DEFAULT_SCENARIO_PREFIX)])
    if not cols:
        raise ValueError("scenario data has no residual scenario columns")
    if base_col not in scenario_df.columns:
        raise ValueError(f"scenario data missing base column: {base_col}")

    df = scenario_df.copy()
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    rows: list[dict[str, float | int | str]] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        scenario_values = group[cols].to_numpy(dtype=float).T
        base_values = group[base_col].to_numpy(dtype=float)
        day_source = source_map.loc[source_map["date"].astype(str).eq(str(date))]
        lag1 = np.asarray([_lag1_autocorr(values) for values in scenario_values], dtype=float)
        smoothness = np.asarray([_mean_abs_diff(values) for values in scenario_values], dtype=float)
        rows.append(
            {
                "date": str(date),
                "n_scenarios": int(len(cols)),
                "unique_source_days": int(day_source["source_date"].nunique()) if not day_source.empty else 0,
                "base_lag1_autocorr": _lag1_autocorr(base_values),
                "scenario_lag1_autocorr_mean": _nanmean(lag1),
                "scenario_lag1_autocorr_p10": _nanquantile(lag1, 0.10),
                "base_mean_abs_diff": _mean_abs_diff(base_values),
                "scenario_mean_abs_diff_mean": _nanmean(smoothness),
                "scenario_mean_abs_diff_p90": _nanquantile(smoothness, 0.90),
                "scenario_slot_std_mean": float(np.mean(np.std(scenario_values, axis=0))),
                "scenario_min": float(np.min(scenario_values)),
                "scenario_max": float(np.max(scenario_values)),
                "source_date_sample": ";".join(day_source["source_date"].astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows)


def residual_library_diagnostics(library: ResidualLibrary) -> pd.DataFrame:
    lag1 = np.asarray([_lag1_autocorr(values) for values in library.residuals], dtype=float)
    smoothness = np.asarray([_mean_abs_diff(values) for values in library.residuals], dtype=float)
    return pd.DataFrame(
        [
            {
                "residual_days": int(library.residuals.shape[0]),
                "slots_per_day": int(library.residuals.shape[1]),
                "residual_lag1_autocorr_mean": _nanmean(lag1),
                "residual_lag1_autocorr_p10": _nanquantile(lag1, 0.10),
                "residual_mean_abs_diff_mean": _nanmean(smoothness),
                "residual_slot_std_mean": float(np.mean(np.std(library.residuals, axis=0))),
                "residual_min": float(np.min(library.residuals)),
                "residual_max": float(np.max(library.residuals)),
                "prediction_cols": ",".join(library.prediction_cols),
                "true_col": library.true_col,
            }
        ]
    )


def scenario_columns(df: pd.DataFrame, prefix: str = DEFAULT_SCENARIO_PREFIX) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def _lag1_autocorr(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return float("nan")
    left = arr[:-1]
    right = arr[1:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _mean_abs_diff(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))))


def _nanmean(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _nanquantile(values: np.ndarray, quantile: float) -> float:
    if np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanquantile(values, quantile))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate whole-day residual resampling price scenarios.")
    parser.add_argument("--val-pred-csv", required=True)
    parser.add_argument("--test-pred-csv", required=True)
    parser.add_argument("--output", default="outputs/test_predictions_residual_scenarios.csv")
    parser.add_argument("--diagnostics-output", default="outputs/residual_scenario_diagnostics.csv")
    parser.add_argument("--library-diagnostics-output", default="outputs/residual_library_diagnostics.csv")
    parser.add_argument("--source-map-output", default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--true-col", default="A")
    parser.add_argument("--prediction-cols", default="")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260526)
    parser.add_argument("--scenario-prefix", default=DEFAULT_SCENARIO_PREFIX)
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_pred_csv)
    test_df = pd.read_csv(args.test_pred_csv)
    prediction_cols = detect_prediction_columns(val_df, args.prediction_cols)
    missing_in_test = [col for col in prediction_cols if col not in test_df.columns]
    if missing_in_test:
        raise ValueError(f"test data missing prediction columns used for residuals: {missing_in_test}")

    price_col = args.price_col or infer_price_column(test_df)
    library = build_daily_residual_library(
        val_df,
        true_col=args.true_col,
        prediction_cols=prediction_cols,
    )
    scenario_df, source_map = generate_residual_scenarios(
        test_df,
        library,
        n_scenarios=args.n_scenarios,
        random_seed=args.random_seed,
        prediction_cols=prediction_cols,
        price_col=price_col,
        scenario_prefix=args.scenario_prefix,
    )
    diagnostics = scenario_diagnostics(
        scenario_df,
        source_map,
        scenario_cols=scenario_columns(scenario_df, args.scenario_prefix),
    )
    library_diag = residual_library_diagnostics(library)

    for path_text, frame in [
        (args.output, scenario_df),
        (args.source_map_output, source_map),
        (args.diagnostics_output, diagnostics),
        (args.library_diagnostics_output, library_diag),
    ]:
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    print(
        f"residual_scenarios={args.output}, rows={len(scenario_df)}, "
        f"n_scenarios={args.n_scenarios}, residual_days={library.residuals.shape[0]}, "
        f"prediction_cols={','.join(prediction_cols)}"
    )
    print(
        f"diagnostics={args.diagnostics_output}, source_map={args.source_map_output}, "
        f"library_diagnostics={args.library_diagnostics_output}"
    )


if __name__ == "__main__":
    main()
