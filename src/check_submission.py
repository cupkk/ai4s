from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


ALLOWED_POWER = np.asarray([-1000.0, 0.0, 1000.0])


@dataclass
class CheckResult:
    errors: List[str]
    warnings: List[str]
    days: int
    rows: int
    traded_days: int


def _runs(mask: Iterable[bool]) -> List[tuple[int, int]]:
    runs: List[tuple[int, int]] = []
    start = None
    values = list(mask)
    for idx, flag in enumerate(values):
        if flag and start is None:
            start = idx
        if (not flag or idx == len(values) - 1) and start is not None:
            end = idx - 1 if not flag else idx
            runs.append((start, end))
            start = None
    return runs


def _power_allowed(values: pd.Series, tol: float = 1e-6) -> bool:
    arr = values.to_numpy(dtype=float)
    return bool(np.all(np.min(np.abs(arr[:, None] - ALLOWED_POWER[None, :]), axis=1) <= tol))


def check_submission(path: str, expected_rows: int | None = None) -> CheckResult:
    df = pd.read_csv(path)
    errors: List[str] = []
    warnings: List[str] = []

    if "times" not in df.columns:
        errors.append("missing required column: times")
    if "power" not in df.columns:
        errors.append("missing required column: power")
    if errors:
        return CheckResult(errors, warnings, days=0, rows=len(df), traded_days=0)

    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"row count mismatch: expected {expected_rows}, got {len(df)}")

    df = df.copy()
    df["times"] = pd.to_datetime(df["times"], errors="coerce")
    if df["times"].isna().any():
        errors.append("times contains unparsable values")
        bad_rows = int(df["times"].isna().sum())
        warnings.append(f"unparsable time rows: {bad_rows}")
        return CheckResult(errors, warnings, days=0, rows=len(df), traded_days=0)

    if df["times"].duplicated().any():
        errors.append(f"duplicated timestamps: {int(df['times'].duplicated().sum())}")
    if not df["times"].is_monotonic_increasing:
        errors.append("times must be sorted ascending")

    if not pd.api.types.is_numeric_dtype(df["power"]):
        errors.append("power column must be numeric")
        return CheckResult(errors, warnings, days=0, rows=len(df), traded_days=0)
    if not _power_allowed(df["power"]):
        bad = sorted(set(df.loc[~df["power"].isin([-1000, 0, 1000]), "power"].round(6).tolist()))
        errors.append(f"power must only contain -1000, 0, 1000; examples={bad[:10]}")

    df["__date__"] = df["times"].dt.date
    traded_days = 0
    for date, group in df.groupby("__date__", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        if len(group) != 96:
            errors.append(f"{date}: expected 96 rows, got {len(group)}")
            continue

        diffs = group["times"].diff().dropna()
        if not diffs.empty and not (diffs == pd.Timedelta(minutes=15)).all():
            errors.append(f"{date}: timestamps are not continuous 15-minute intervals")

        power = group["power"].to_numpy(dtype=float)
        charge_runs = _runs(power < 0)
        discharge_runs = _runs(power > 0)
        if power.any():
            traded_days += 1
        if len(charge_runs) > 1:
            errors.append(f"{date}: more than one charge block: {charge_runs}")
        if len(discharge_runs) > 1:
            errors.append(f"{date}: more than one discharge block: {discharge_runs}")
        for kind, runs in [("charge", charge_runs), ("discharge", discharge_runs)]:
            if runs:
                start, end = runs[0]
                length = end - start + 1
                if length != 8:
                    errors.append(f"{date}: {kind} block length must be 8, got {length}")
        if bool(charge_runs) != bool(discharge_runs):
            errors.append(f"{date}: charge and discharge blocks must appear together")
        if charge_runs and discharge_runs and charge_runs[0][1] >= discharge_runs[0][0]:
            errors.append(
                f"{date}: charge block must finish before discharge block starts "
                f"(charge={charge_runs[0]}, discharge={discharge_runs[0]})"
            )

    price_cols = [col for col in df.columns if col not in {"times", "power", "__date__"}]
    if not price_cols:
        warnings.append("no price/prediction column found; official format may still require one")

    return CheckResult(
        errors=errors,
        warnings=warnings,
        days=int(df["__date__"].nunique()),
        rows=int(len(df)),
        traded_days=traded_days,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check official submission format and storage actions.")
    parser.add_argument("--submission", required=True, help="Submission CSV to check.")
    parser.add_argument("--expected-rows", type=int, default=None)
    args = parser.parse_args()

    result = check_submission(args.submission, expected_rows=args.expected_rows)
    print(
        "submission_check="
        f"rows={result.rows}, days={result.days}, traded_days={result.traded_days}, "
        f"errors={len(result.errors)}, warnings={len(result.warnings)}"
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")
    if result.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
