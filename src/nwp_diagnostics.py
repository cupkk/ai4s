from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .features import TIME_COL
from .nwp_features import extract_one_nc, inspect_nc_layout


def diagnose_nwp(
    nwp_dir: str,
    output: str,
    max_files: int = 0,
    timezone_label: str = "BJT",
) -> pd.DataFrame:
    files = sorted(Path(nwp_dir).glob("*.nc"))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .nc files found in {nwp_dir}")

    rows: List[dict] = []
    for file in files:
        layout = inspect_nc_layout(file)
        frame = extract_one_nc(file)
        frame[TIME_COL] = pd.to_datetime(frame[TIME_COL])
        nwp_cols = [col for col in frame.columns if col.startswith("nwp_")]
        missing_ratio = float(frame[nwp_cols].isna().mean().mean()) if nwp_cols else 1.0

        if "nwp_ghi_mean" in frame.columns and frame["nwp_ghi_mean"].notna().any():
            ghi = frame[[TIME_COL, "nwp_ghi_mean"]].dropna().reset_index(drop=True)
            max_row = ghi.iloc[int(ghi["nwp_ghi_mean"].to_numpy().argmax())]
            night = ghi[ghi[TIME_COL].dt.hour.isin([0, 1, 2, 3, 4, 20, 21, 22, 23])]
            ghi_max_time = pd.Timestamp(max_row[TIME_COL])
            ghi_max_hour = int(ghi_max_time.hour)
            ghi_night_mean = float(night["nwp_ghi_mean"].mean()) if not night.empty else float("nan")
            ghi_max_value = float(max_row["nwp_ghi_mean"])
            ghi_time_ok = 10 <= ghi_max_hour <= 15
        else:
            ghi_max_time = pd.NaT
            ghi_max_hour = -1
            ghi_night_mean = float("nan")
            ghi_max_value = float("nan")
            ghi_time_ok = False

        wind_speed_mean = (
            float(frame["nwp_wind_speed_mean"].mean())
            if "nwp_wind_speed_mean" in frame.columns
            else float("nan")
        )
        rows.append(
            {
                "source_file": file.name,
                "date": str(frame[TIME_COL].dt.date.min()),
                "data_shape": "x".join(str(v) for v in layout["data_shape"]),
                "channel_axis_after_time": layout["channel_axis_after_time"],
                "hour_axis_after_time": layout["hour_axis_after_time"],
                "hour_count": layout["hour_count"],
                "timezone_label": timezone_label,
                "ghi_max_time": "" if pd.isna(ghi_max_time) else str(ghi_max_time),
                "ghi_max_hour": ghi_max_hour,
                "ghi_max_slot": -1
                if pd.isna(ghi_max_time)
                else int(ghi_max_time.hour * 4 + ghi_max_time.minute // 15),
                "ghi_max_value": ghi_max_value,
                "ghi_night_mean": ghi_night_mean,
                "ghi_peak_time_ok_10_15": bool(ghi_time_ok),
                "wind_speed_mean": wind_speed_mean,
                "missing_ratio": missing_ratio,
                "channels": "|".join(layout["channels"]),
            }
        )

    out = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose official NWP layout and time alignment.")
    parser.add_argument("--nwp-dir", required=True)
    parser.add_argument("--output", default="outputs/nwp_diagnostics.csv")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--timezone-label", default="BJT")
    args = parser.parse_args()

    result = diagnose_nwp(
        args.nwp_dir,
        args.output,
        max_files=args.max_files,
        timezone_label=args.timezone_label,
    )
    print(result.to_string(index=False))
    bad = result[~result["ghi_peak_time_ok_10_15"]]
    if not bad.empty:
        print(
            "WARNING: some GHI peak hours are outside 10:00-15:00; "
            "check timezone or lead_time alignment."
        )
    print(f"saved_nwp_diagnostics={args.output}")


if __name__ == "__main__":
    main()
