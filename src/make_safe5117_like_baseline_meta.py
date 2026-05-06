from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyze_submission_diff import compare_submissions, summarize_diff
from .storage_optimizer import infer_price_column, optimize_one_day


def _actions_from_submission(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "times" not in df.columns or "power" not in df.columns:
        raise ValueError(f"{path} must contain times and power columns")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)
    rows: list[dict[str, Any]] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times")
        power = group["power"].to_numpy(dtype=float)
        charge = np.flatnonzero(power < 0)
        discharge = np.flatnonzero(power > 0)
        rows.append(
            {
                "date": date,
                "charge_start": int(charge[0]) if len(charge) else np.nan,
                "discharge_start": int(discharge[0]) if len(discharge) else np.nan,
                "traded": bool(len(charge) and len(discharge)),
            }
        )
    return pd.DataFrame(rows)


def learn_charge_offset_map(
    safe_submission: str,
    source_submission: str,
    min_group_count: int = 2,
) -> tuple[dict[int, int], dict[str, Any]]:
    safe = _actions_from_submission(safe_submission).rename(
        columns={
            "charge_start": "safe_charge_start",
            "discharge_start": "safe_discharge_start",
        }
    )
    source = _actions_from_submission(source_submission).rename(
        columns={
            "charge_start": "source_charge_start",
            "discharge_start": "source_discharge_start",
        }
    )
    merged = safe.merge(source, on="date", how="inner")
    merged = merged.loc[merged["traded_x"].astype(bool) & merged["traded_y"].astype(bool)].copy()
    merged["charge_offset"] = (
        merged["safe_charge_start"].astype(int) - merged["source_charge_start"].astype(int)
    )
    global_offset = int(round(float(merged["charge_offset"].median())))
    offset_map: dict[int, int] = {}
    group_rows = []
    for discharge_start, group in merged.groupby("safe_discharge_start", sort=True):
        median_offset = int(round(float(group["charge_offset"].median())))
        count = int(len(group))
        chosen = median_offset if count >= int(min_group_count) else global_offset
        offset_map[int(discharge_start)] = int(chosen)
        group_rows.append(
            {
                "safe_discharge_start": int(discharge_start),
                "count": count,
                "median_charge_offset": median_offset,
                "chosen_charge_offset": int(chosen),
            }
        )
    diagnostics = {
        "global_median_charge_offset": global_offset,
        "groups": group_rows,
        "source_rows": merged.to_dict(orient="records"),
    }
    return offset_map, diagnostics


def generate_meta_from_predictions(
    price_csv: str,
    output: str,
    price_col: str = "",
    charge_start_min: int = 51,
    charge_start_max: int = 55,
    discharge_start_min: int = 67,
    discharge_start_max: int = 88,
    threshold: float = 0.0,
    charge_offset_by_discharge: dict[int, int] | None = None,
    block_size: int = 8,
) -> pd.DataFrame:
    price_df = pd.read_csv(price_csv)
    if "times" not in price_df.columns:
        raise ValueError(f"{price_csv} missing times column")
    price_df["times"] = pd.to_datetime(price_df["times"])
    price_col = infer_price_column(price_df, price_col or None)
    price_df["date"] = price_df["times"].dt.date.astype(str)
    offset_map = charge_offset_by_discharge or {}

    rows = []
    for date, group in price_df.groupby("date", sort=True):
        group = group.sort_values("times")
        if len(group) != 96:
            continue
        result = optimize_one_day(
            group[price_col].to_numpy(dtype=float),
            threshold=threshold,
            charge_start_min=charge_start_min,
            charge_start_max=charge_start_max,
            discharge_start_min=discharge_start_min,
            discharge_start_max=discharge_start_max,
        )
        if not result.traded or result.charge_start is None or result.discharge_start is None:
            continue
        discharge_start = int(result.discharge_start)
        raw_charge_start = int(result.charge_start)
        offset = int(offset_map.get(discharge_start, 0))
        corrected_charge_start = raw_charge_start + offset
        corrected_charge_start = max(0, min(discharge_start - block_size, corrected_charge_start))
        rows.append(
            {
                "date": str(date),
                "pred_best_profit": float(result.best_profit),
                "raw_charge_start": raw_charge_start,
                "raw_discharge_start": discharge_start,
                "charge_offset": offset,
                "charge_start": int(corrected_charge_start),
                "discharge_start": discharge_start,
                "traded": True,
            }
        )
    meta = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output, index=False)
    return meta


def _meta_to_submission_like(meta: pd.DataFrame, template_submission: str, output: str) -> None:
    template = pd.read_csv(template_submission)
    template["times"] = pd.to_datetime(template["times"])
    out = template.copy()
    out["power"] = 0.0
    meta_map = {str(row["date"]): row for _, row in meta.iterrows()}
    for date, group in out.groupby(out["times"].dt.date.astype(str), sort=True):
        if date not in meta_map:
            continue
        row = meta_map[date]
        idx = group.sort_values("times").index.to_numpy()
        power = np.zeros(96, dtype=float)
        c = int(row["charge_start"])
        d = int(row["discharge_start"])
        power[c : c + 8] = -1000.0
        power[d : d + 8] = 1000.0
        out.loc[idx, "power"] = power
    out.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate historical safe5117-like baseline meta from predictions.")
    parser.add_argument("--price-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--price-col", default="")
    parser.add_argument("--safe-submission", default="outputs/output_nwp_unconstrained_online5117.csv")
    parser.add_argument("--source-submission", default="outputs/output_nwp_constrained.csv")
    parser.add_argument("--charge-start-min", type=int, default=51)
    parser.add_argument("--charge-start-max", type=int, default=55)
    parser.add_argument("--discharge-start-min", type=int, default=67)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--min-group-count", type=int, default=2)
    parser.add_argument("--diagnostics-output", default="")
    parser.add_argument("--check-template-submission", default="")
    parser.add_argument("--check-submission-output", default="")
    args = parser.parse_args()

    offset_map, diagnostics = learn_charge_offset_map(
        args.safe_submission,
        args.source_submission,
        min_group_count=args.min_group_count,
    )
    meta = generate_meta_from_predictions(
        price_csv=args.price_csv,
        output=args.output,
        price_col=args.price_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        threshold=args.threshold,
        charge_offset_by_discharge=offset_map,
    )
    diagnostics["offset_map"] = {str(k): int(v) for k, v in offset_map.items()}
    diagnostics["output"] = args.output
    diagnostics["rows"] = int(len(meta))
    if args.diagnostics_output:
        Path(args.diagnostics_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.diagnostics_output).write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"saved_safe5117_like_meta={args.output}, rows={len(meta)}")

    if args.check_template_submission and args.check_submission_output:
        _meta_to_submission_like(meta, args.check_template_submission, args.check_submission_output)
        diff = compare_submissions(
            args.safe_submission,
            args.check_submission_output,
            reference_name="safe5117",
            candidate_name="safe5117_like",
        )
        print(pd.DataFrame([summarize_diff(diff)]).to_string(index=False))


if __name__ == "__main__":
    main()
