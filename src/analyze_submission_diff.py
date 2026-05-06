from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _summarize_actions(path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "times" not in df.columns or "power" not in df.columns:
        raise ValueError(f"{path} must contain times and power columns")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)

    rows = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        power = group["power"].to_numpy(dtype=float)
        charge = [idx for idx, value in enumerate(power) if value < 0]
        discharge = [idx for idx, value in enumerate(power) if value > 0]
        rows.append(
            {
                "date": date,
                f"{prefix}_charge_start": charge[0] if charge else None,
                f"{prefix}_charge_end": charge[-1] if charge else None,
                f"{prefix}_discharge_start": discharge[0] if discharge else None,
                f"{prefix}_discharge_end": discharge[-1] if discharge else None,
                f"{prefix}_traded": bool(charge or discharge),
            }
        )
    return pd.DataFrame(rows)


def compare_submissions(reference: str, candidate: str, reference_name: str, candidate_name: str) -> pd.DataFrame:
    left = _summarize_actions(reference, reference_name)
    right = _summarize_actions(candidate, candidate_name)
    out = left.merge(right, on="date", how="outer")
    out["delta_charge_start"] = out[f"{candidate_name}_charge_start"] - out[f"{reference_name}_charge_start"]
    out["delta_discharge_start"] = (
        out[f"{candidate_name}_discharge_start"] - out[f"{reference_name}_discharge_start"]
    )
    out["same_charge"] = out[f"{reference_name}_charge_start"].eq(out[f"{candidate_name}_charge_start"])
    out["same_discharge"] = out[f"{reference_name}_discharge_start"].eq(
        out[f"{candidate_name}_discharge_start"]
    )
    out["same_both"] = out["same_charge"] & out["same_discharge"]
    return out


def summarize_diff(diff: pd.DataFrame) -> dict:
    changed = diff[~diff["same_both"]].copy()
    return {
        "days": int(len(diff)),
        "same_both_days": int(diff["same_both"].sum()),
        "same_charge_days": int(diff["same_charge"].sum()),
        "same_discharge_days": int(diff["same_discharge"].sum()),
        "changed_days": int((~diff["same_both"]).sum()),
        "mean_abs_charge_delta": float(changed["delta_charge_start"].abs().mean()) if not changed.empty else 0.0,
        "mean_abs_discharge_delta": float(changed["delta_discharge_start"].abs().mean())
        if not changed.empty
        else 0.0,
        "max_abs_charge_delta": float(diff["delta_charge_start"].abs().max()),
        "max_abs_discharge_delta": float(diff["delta_discharge_start"].abs().max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare daily charge/discharge windows between two submissions.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--output", default="")
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    diff = compare_submissions(
        args.reference,
        args.candidate,
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
    )
    summary = summarize_diff(diff)
    summary_df = pd.DataFrame([summary])
    print(summary_df.to_string(index=False))
    print(
        diff.loc[
            ~diff["same_both"],
            [
                "date",
                f"{args.reference_name}_charge_start",
                f"{args.candidate_name}_charge_start",
                "delta_charge_start",
                f"{args.reference_name}_discharge_start",
                f"{args.candidate_name}_discharge_start",
                "delta_discharge_start",
            ],
        ]
        .head(30)
        .to_string(index=False)
    )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        diff.to_csv(output_path, index=False)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
