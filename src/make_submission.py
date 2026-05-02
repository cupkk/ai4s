from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .storage_optimizer import generate_strategy


def _threshold_from_file(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return None
    return float(text)


def _threshold_from_metadata(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    value = metadata.get("best_threshold")
    return None if value is None else float(value)


def _parse_threshold_by_month(text: str) -> dict[int, float]:
    result: dict[int, float] = {}
    if not text:
        return result
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        month_text, value_text = item.split(":", 1)
        month = int(month_text)
        if month < 1 or month > 12:
            raise ValueError(f"invalid month in threshold map: {month}")
        result[month] = float(value_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate output.csv from predicted prices.")
    parser.add_argument("--price-csv", required=True, help="CSV with times and predicted price.")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--price-col", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-by-month", default="")
    parser.add_argument("--threshold-file", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--charge-start-min", type=int, default=None)
    parser.add_argument("--charge-start-max", type=int, default=None)
    parser.add_argument("--discharge-start-min", type=int, default=None)
    parser.add_argument("--discharge-start-max", type=int, default=None)
    parser.add_argument("--meta-output", default="outputs/submission_strategy_meta.csv")
    args = parser.parse_args()

    threshold = args.threshold
    if threshold is None:
        threshold = _threshold_from_file(args.threshold_file)
    if threshold is None:
        threshold = _threshold_from_metadata(args.metadata)
    if threshold is None:
        threshold = 0.0
    threshold_by_month = _parse_threshold_by_month(args.threshold_by_month)

    constraints = {}
    if args.metadata:
        metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        constraints.update(metadata.get("strategy_constraints", {}))
    cli_constraints = {
        "charge_start_min": args.charge_start_min,
        "charge_start_max": args.charge_start_max,
        "discharge_start_min": args.discharge_start_min,
        "discharge_start_max": args.discharge_start_max,
    }
    constraints.update({k: v for k, v in cli_constraints.items() if v is not None})
    constraints.setdefault("charge_start_min", 0)
    constraints.setdefault("charge_start_max", 80)
    constraints.setdefault("discharge_start_min", 8)
    constraints.setdefault("discharge_start_max", 88)

    price_df = pd.read_csv(args.price_csv)
    out, meta = generate_strategy(
        price_df,
        threshold=threshold,
        price_col=args.price_col or None,
        threshold_by_month=threshold_by_month or None,
        **constraints,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    if args.meta_output:
        Path(args.meta_output).parent.mkdir(parents=True, exist_ok=True)
        meta.to_csv(args.meta_output, index=False)
    print(
        f"saved_submission={args.output}, rows={len(out)}, "
        f"threshold={threshold}, threshold_by_month={threshold_by_month}"
    )


if __name__ == "__main__":
    main()
