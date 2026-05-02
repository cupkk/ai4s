from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


FALLBACK_SUBMISSIONS = {
    "base_unconstrained": "outputs/output_base_unconstrained.csv",
    "nwp_unconstrained": "outputs/output_nwp_unconstrained.csv",
    "nwp_c0_55": "outputs/output_nwp_c0_55.csv",
    "nwp_c0_55_d72_88": "outputs/output_nwp_c0_55_d72_88.csv",
    "nwp_threshold_500": "outputs/output_nwp_unconstrained_t500.csv",
    "nwp_threshold_1000": "outputs/output_nwp_unconstrained_t1000.csv",
    "nwp_threshold_2000": "outputs/output_nwp_unconstrained_t2000.csv",
    "nwp_bias_unconstrained": "outputs/output_nwp_bias.csv",
    "nwp_exact_bias_unconstrained": "outputs/output_nwp_exact_bias.csv",
}


def select_best_submission(
    strategy_compare: str,
    output: str,
    report_output: str,
    min_trade_rate: float = 0.0,
    max_loss_days: int | None = None,
) -> pd.Series:
    compare = pd.read_csv(strategy_compare)
    if compare.empty:
        raise ValueError(f"strategy compare file is empty: {strategy_compare}")

    usable = compare.copy()
    if "available" in usable.columns:
        usable = usable[usable["available"].astype(bool)]
    if "error" in usable.columns:
        usable = usable[usable["error"].fillna("") == ""]
    if max_loss_days is not None and "loss_days" in usable.columns:
        usable = usable[usable["loss_days"] <= max_loss_days]
    if min_trade_rate > 0 and {"traded_days", "days"}.issubset(usable.columns):
        usable = usable[usable["traded_days"] / usable["days"] >= min_trade_rate]
    if usable.empty:
        raise ValueError("no usable strategy remains after selection filters")

    if "submission_csv" not in usable.columns:
        usable["submission_csv"] = usable["name"].map(FALLBACK_SUBMISSIONS)
    else:
        fallback = usable["name"].map(FALLBACK_SUBMISSIONS)
        usable["submission_csv"] = usable["submission_csv"].fillna(fallback)

    usable = usable[usable["submission_csv"].notna()].copy()
    usable["submission_exists"] = usable["submission_csv"].map(lambda path: Path(str(path)).exists())
    usable = usable[usable["submission_exists"]]
    if usable.empty:
        raise FileNotFoundError("no usable strategy has an existing submission_csv")

    sort_cols = []
    ascending = []
    for col, asc in [
        ("avg_profit", False),
        ("capture_ratio", False),
        ("loss_days", True),
        ("window_hit_2_rate", False),
    ]:
        if col in usable.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        raise ValueError("strategy compare lacks selection metrics")

    best = usable.sort_values(sort_cols, ascending=ascending).iloc[0]
    src = Path(str(best["submission_csv"]))
    dst = Path(output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)

    report = best.to_frame().T
    report_path = Path(report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Select final output.csv from strategy comparison results.")
    parser.add_argument("--strategy-compare", default="outputs/strategy_compare.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--report-output", default="outputs/selected_submission.csv")
    parser.add_argument("--min-trade-rate", type=float, default=0.0)
    parser.add_argument("--max-loss-days", type=int, default=None)
    args = parser.parse_args()

    best = select_best_submission(
        args.strategy_compare,
        args.output,
        args.report_output,
        min_trade_rate=args.min_trade_rate,
        max_loss_days=args.max_loss_days,
    )
    print(
        "selected_submission="
        f"name={best['name']}, source={best['submission_csv']}, output={args.output}, "
        f"avg_profit={best.get('avg_profit', '')}, capture_ratio={best.get('capture_ratio', '')}"
    )


if __name__ == "__main__":
    main()
