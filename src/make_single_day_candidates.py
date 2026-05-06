from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analyze_submission_diff import compare_submissions, summarize_diff
from .check_submission import check_submission


DEFAULT_BASELINE = "outputs/output_nwp_unconstrained_online5117.csv"
DEFAULT_RANKED_WINDOWS = "outputs/test_windows_window_ranker.csv"
DEFAULT_MANIFEST = "outputs/single_day_candidate_manifest.csv"
BLOCKED_DATES = {"2026-01-11"}


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def _repo_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _daily_actions_from_submission(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "times" not in df.columns or "power" not in df.columns:
        raise ValueError(f"{path} must contain times and power columns")
    df["times"] = pd.to_datetime(df["times"])
    df["date"] = df["times"].dt.date.astype(str)

    rows: list[dict[str, Any]] = []
    for date, group in df.groupby("date", sort=True):
        group = group.sort_values("times").reset_index(drop=True)
        power = group["power"].to_numpy(dtype=float)
        charge = np.flatnonzero(power < 0)
        discharge = np.flatnonzero(power > 0)
        rows.append(
            {
                "date": date,
                "baseline_charge_start": int(charge[0]) if len(charge) else np.nan,
                "baseline_discharge_start": int(discharge[0]) if len(discharge) else np.nan,
                "baseline_traded": bool(len(charge) or len(discharge)),
            }
        )
    return pd.DataFrame(rows)


def _load_recommendations(ranked_windows: str | Path) -> pd.DataFrame:
    windows = pd.read_csv(ranked_windows)
    if "pred_window_profit" not in windows.columns and "pred_expected_delta" in windows.columns:
        windows = windows.rename(columns={"pred_expected_delta": "pred_window_profit"})
    if "pred_window_profit" not in windows.columns and "pred_window_score" in windows.columns:
        windows = windows.rename(columns={"pred_window_score": "pred_window_profit"})
    required = {"date", "charge_start", "discharge_start", "pred_window_profit"}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"{ranked_windows} missing required columns: {sorted(missing)}")

    if "pred_rank" in windows.columns:
        selected = windows.loc[windows["pred_rank"].astype(float) == 1.0].copy()
    else:
        idx = windows.groupby("date", sort=True)["pred_window_profit"].idxmax()
        selected = windows.loc[idx].copy()

    if "pred_window_profit_std" not in selected.columns:
        if "pred_positive_proba_std" in selected.columns:
            selected["pred_window_profit_std"] = selected["pred_positive_proba_std"].astype(float)
        else:
            selected["pred_window_profit_std"] = 0.0
    if "top1_minus_top2_margin" not in selected.columns:
        selected["top1_minus_top2_margin"] = 0.0
    return selected.sort_values("date").reset_index(drop=True)


def _apply_day_window(
    baseline_df: pd.DataFrame,
    date: str,
    charge_start: int,
    discharge_start: int,
) -> pd.DataFrame:
    out = baseline_df.copy()
    times = pd.to_datetime(out["times"])
    mask = times.dt.date.astype(str) == date
    if int(mask.sum()) != 96:
        raise ValueError(f"{date}: baseline must contain 96 rows, got {int(mask.sum())}")

    day_index = out.loc[mask].sort_values("times").index.to_numpy()
    power = np.zeros(96, dtype=float)
    power[charge_start : charge_start + 8] = -1000.0
    power[discharge_start : discharge_start + 8] = 1000.0
    out.loc[day_index, "power"] = power
    return out


def _candidate_sort_key(df: pd.DataFrame) -> pd.Series:
    score_std = df["score_std"].astype(float).clip(lower=0.0)
    margin = df["top1_top2_margin"].astype(float)
    pred = df["pred_window_score"].astype(float)
    return pred + margin - 0.25 * score_std


def build_single_day_candidates(
    baseline: str = DEFAULT_BASELINE,
    ranked_windows: str = DEFAULT_RANKED_WINDOWS,
    output_dir: str = "outputs",
    manifest_output: str = DEFAULT_MANIFEST,
    max_candidates: int = 3,
    tag: str = "df",
    min_margin: float = 0.0,
    max_score_std: float | None = None,
    min_pred_score: float = 0.0,
    allow_submission: bool = False,
) -> pd.DataFrame:
    baseline_df = pd.read_csv(baseline)
    if "times" not in baseline_df.columns or "power" not in baseline_df.columns:
        raise ValueError(f"{baseline} must contain times and power columns")
    baseline_df["times"] = pd.to_datetime(baseline_df["times"])

    baseline_actions = _daily_actions_from_submission(baseline)
    recs = _load_recommendations(ranked_windows).rename(
        columns={
            "charge_start": "candidate_charge_start",
            "discharge_start": "candidate_discharge_start",
            "pred_window_profit": "pred_window_score",
            "pred_window_profit_std": "score_std",
            "top1_minus_top2_margin": "top1_top2_margin",
        }
    )
    recs = recs.drop(
        columns=[
            col
            for col in recs.columns
            if col.startswith("baseline_") or col in {"delta_charge_start", "delta_discharge_start"}
        ],
        errors="ignore",
    )
    merged = recs.merge(baseline_actions, on="date", how="inner")
    merged["candidate_charge_start"] = merged["candidate_charge_start"].astype(int)
    merged["candidate_discharge_start"] = merged["candidate_discharge_start"].astype(int)
    merged["baseline_charge_start"] = merged["baseline_charge_start"].astype(int)
    merged["baseline_discharge_start"] = merged["baseline_discharge_start"].astype(int)
    merged["same_action"] = (
        (merged["candidate_charge_start"] == merged["baseline_charge_start"])
        & (merged["candidate_discharge_start"] == merged["baseline_discharge_start"])
    )

    eligible = merged.loc[
        (~merged["date"].isin(BLOCKED_DATES))
        & (~merged["same_action"])
        & (merged["top1_top2_margin"].astype(float) >= float(min_margin))
        & (merged["pred_window_score"].astype(float) >= float(min_pred_score))
    ].copy()
    if max_score_std is not None:
        eligible = eligible.loc[eligible["score_std"].astype(float) <= float(max_score_std)].copy()
    if eligible.empty:
        raise ValueError("no eligible single-day replacement candidates")

    eligible["selection_score"] = _candidate_sort_key(eligible)
    eligible = eligible.sort_values(
        ["selection_score", "top1_top2_margin", "pred_window_score"],
        ascending=[False, False, False],
    ).head(int(max_candidates))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    for _, row in eligible.iterrows():
        date = str(row["date"])
        charge_start = int(row["candidate_charge_start"])
        discharge_start = int(row["candidate_discharge_start"])
        safe_date = date.replace("-", "")
        candidate_path = output_path / f"output_df_single_{safe_date}_{tag}.csv"
        candidate_df = _apply_day_window(baseline_df, date, charge_start, discharge_start)
        candidate_df.to_csv(candidate_path, index=False)

        check = check_submission(str(candidate_path))
        diff = compare_submissions(
            baseline,
            str(candidate_path),
            reference_name="safe5117",
            candidate_name=f"single_{safe_date}_{tag}",
        )
        summary = summarize_diff(diff)
        if check.errors or check.warnings or int(summary["changed_days"]) != 1:
            raise ValueError(
                f"generated invalid candidate {candidate_path}: "
                f"errors={check.errors}, warnings={check.warnings}, changed_days={summary['changed_days']}"
            )

        reason = (
            "单日替换候选：模型推荐窗口不同于5117保底，"
            f"margin={float(row['top1_top2_margin']):.3f}, "
            f"std={float(row['score_std']):.3f}; 不涉及跳过2026-01-11。"
        )
        if "pred_positive_proba" in row.index and pd.notna(row["pred_positive_proba"]):
            reason = (
                "单日替换候选：保守替换分类器认为该日近保底窗口有正 delta 机会，"
                f"p_positive={float(row['pred_positive_proba']):.3f}, "
                f"expected_delta={float(row['pred_window_score']):.3f}, "
                f"std={float(row['score_std']):.3f}; 不涉及跳过2026-01-11。"
            )
        if not allow_submission:
            reason += " 当前为研究候选，默认阻止提交；通过滚动验证和人工确认后才能重新生成可提交 manifest。"
        manifest_rows.append(
            {
                "candidate_csv": _repo_path(candidate_path),
                "candidate_sha256": _sha256(candidate_path),
                "baseline_csv": _repo_path(baseline),
                "baseline_sha256": _sha256(baseline),
                "date": date,
                "changed_days": int(summary["changed_days"]),
                "baseline_charge_start": int(row["baseline_charge_start"]),
                "baseline_discharge_start": int(row["baseline_discharge_start"]),
                "candidate_charge_start": charge_start,
                "candidate_discharge_start": discharge_start,
                "pred_window_score": float(row["pred_window_score"]),
                "pred_delta_profit": float(row["pred_window_score"]),
                "pred_positive_proba": float(row["pred_positive_proba"])
                if "pred_positive_proba" in row.index and pd.notna(row["pred_positive_proba"])
                else np.nan,
                "score_std": float(row["score_std"]),
                "top1_top2_margin": float(row["top1_top2_margin"]),
                "selection_score": float(row["selection_score"]),
                "blocked": not allow_submission,
                "reason": reason,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = Path(manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate manifest-backed one-day replacements from a ranked window model."
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--ranked-windows", default=DEFAULT_RANKED_WINDOWS)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--tag", default="df")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-score-std", type=float, default=None)
    parser.add_argument(
        "--min-pred-score",
        type=float,
        default=0.0,
        help="Minimum predicted score. For baseline-delta windows this is minimum predicted gain.",
    )
    parser.add_argument(
        "--allow-submission",
        action="store_true",
        help="Mark generated manifest rows as unblocked. Use only after validation review.",
    )
    args = parser.parse_args()

    manifest = build_single_day_candidates(
        baseline=args.baseline,
        ranked_windows=args.ranked_windows,
        output_dir=args.output_dir,
        manifest_output=args.manifest_output,
        max_candidates=args.max_candidates,
        tag=args.tag,
        min_margin=args.min_margin,
        max_score_std=args.max_score_std,
        min_pred_score=args.min_pred_score,
        allow_submission=args.allow_submission,
    )
    print(f"single_day_candidates={len(manifest)}")
    print(manifest[["candidate_csv", "date", "changed_days", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
