from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from .analyze_submission_diff import compare_submissions, summarize_diff
from .check_submission import check_submission


BASELINE_SCORE = 5117.832037755039
DEFAULT_REFERENCE = "outputs/output_nwp_unconstrained_online5117.csv"
DEFAULT_MANIFEST = "outputs/single_day_candidate_manifest.csv"
BLOCKED_SINGLE_DAY_DATES = {"2026-01-11"}

ALLOWED_CANDIDATES = {
    "outputs/output_nwp_unconstrained_online5117.csv": {
        "expected_changed_days": 0,
        "stage": "fallback",
        "next_step": "Use as the verified fallback baseline.",
    },
}

BLOCKED_CANDIDATES = {
    "outputs/output_safe5117_skip_t500.csv": "online score 4987.610162489461; skipping 2026-01-11 hurt vs 5117 baseline",
    "outputs/output_safe5117_skip_t1000.csv": "blocked because t500 failed; do not expand skip threshold",
    "outputs/output_safe5117_skip_t1500.csv": "blocked because t500 failed; do not expand skip threshold",
    "outputs/output_nwp_c0_55_d72_88.csv": "online score 3798.629342284567; overfit and changed all 59 days",
    "outputs/output_blend_fine_w025_t1000.csv": "online score 4703.505815153465; blend overfit",
    "outputs/output_nwp_unconstrained_t2000.csv": "online score 4903.504068225546; threshold too defensive",
    "outputs/output_residual_nwp.csv": "local validation weaker than main line",
    "outputs/output_window_ranker_c055_d7288.csv": "local validation weaker than main line",
}


def _as_repo_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def _format_window(row: pd.Series, prefix: str) -> str:
    traded = bool(row[f"{prefix}_traded"])
    if not traded:
        return "no_trade"
    charge_start = int(row[f"{prefix}_charge_start"])
    charge_end = int(row[f"{prefix}_charge_end"])
    discharge_start = int(row[f"{prefix}_discharge_start"])
    discharge_end = int(row[f"{prefix}_discharge_end"])
    return f"charge={charge_start}-{charge_end};discharge={discharge_start}-{discharge_end}"


def _changed_actions(diff: pd.DataFrame, reference_name: str, candidate_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed = diff.loc[~diff["same_both"]].copy()
    for _, row in changed.iterrows():
        rows.append(
            {
                "date": row["date"],
                "reference_action": _format_window(row, reference_name),
                "candidate_action": _format_window(row, candidate_name),
                "same_charge": bool(row["same_charge"]),
                "same_discharge": bool(row["same_discharge"]),
            }
        )
    return rows


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "blocked"}


def _load_manifest_row(manifest_path: str, repo_candidate: str, candidate_sha256: str) -> pd.Series | None:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = pd.read_csv(path)
    if "candidate_csv" not in manifest.columns:
        raise ValueError(f"manifest missing required column: candidate_csv")

    rows = []
    for _, row in manifest.iterrows():
        manifest_candidate = _as_repo_path(row["candidate_csv"])
        if manifest_candidate == repo_candidate:
            rows.append(row)
            continue
        if "candidate_sha256" in manifest.columns and str(row["candidate_sha256"]).upper() == candidate_sha256:
            rows.append(row)
    if not rows:
        return None
    if len(rows) > 1:
        raise ValueError(f"manifest has multiple rows for candidate: {repo_candidate}")
    return rows[0]


def _guard_manifest_policy(
    manifest_row: pd.Series | None,
    summary: dict[str, Any],
    changed_actions: list[dict[str, Any]],
    candidate_sha256: str,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    info: dict[str, Any] = {"manifest_status": "not_used"}
    if manifest_row is None:
        errors.append("candidate is not in static allowlist and no matching manifest row was found")
        info["manifest_status"] = "missing_row"
        return errors, info

    info["manifest_status"] = "matched"
    info["stage"] = "manifest_single_day"
    info["next_step"] = "Manifest matched; submit only if final decision=PASS and model evidence was reviewed."
    manifest_candidate_sha = str(manifest_row.get("candidate_sha256", "")).upper()
    if manifest_candidate_sha and manifest_candidate_sha != candidate_sha256:
        errors.append(
            f"manifest candidate_sha256 mismatch: manifest={manifest_candidate_sha}, actual={candidate_sha256}"
        )

    if _truthy(manifest_row.get("blocked", False)):
        errors.append("manifest marks candidate as blocked")
        info["next_step"] = "Do not submit: manifest marks this candidate as blocked."

    manifest_date = str(manifest_row.get("date", "")).strip()
    info["manifest_date"] = manifest_date
    if manifest_date in BLOCKED_SINGLE_DAY_DATES:
        errors.append(f"manifest date is blocked: {manifest_date}")

    changed_days = int(summary["changed_days"])
    if changed_days != 1:
        errors.append(f"manifest-backed candidates must change exactly 1 day, got changed_days={changed_days}")
    if len(changed_actions) != 1:
        errors.append(f"manifest-backed candidates must have exactly 1 changed action, got {len(changed_actions)}")
        return errors, info

    changed = changed_actions[0]
    changed_date = str(changed["date"])
    info["changed_date"] = changed_date
    if manifest_date and changed_date != manifest_date:
        errors.append(f"manifest date {manifest_date} does not match changed date {changed_date}")
    if changed_date in BLOCKED_SINGLE_DAY_DATES:
        errors.append(f"changed date is blocked: {changed_date}")
    if changed["candidate_action"] == "no_trade":
        errors.append("manifest-backed candidate changes the day to no_trade, which is not allowed")

    if "changed_days" in manifest_row.index:
        try:
            manifest_changed_days = int(manifest_row["changed_days"])
            if manifest_changed_days != 1:
                errors.append(f"manifest changed_days must be 1, got {manifest_changed_days}")
        except (TypeError, ValueError):
            errors.append(f"manifest changed_days is not an integer: {manifest_row['changed_days']}")

    for column in [
        "baseline_charge_start",
        "baseline_discharge_start",
        "candidate_charge_start",
        "candidate_discharge_start",
        "pred_window_score",
        "score_std",
        "top1_top2_margin",
        "reason",
    ]:
        if column not in manifest_row.index or pd.isna(manifest_row[column]):
            errors.append(f"manifest missing required value: {column}")
    return errors, info


def _guard_policy(
    repo_candidate: str,
    summary: dict[str, Any],
    max_changed_days: int,
    changed_actions: list[dict[str, Any]],
    candidate_sha256: str,
    manifest: str = "",
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    info: dict[str, Any] = {"manifest_status": "not_required"}
    if repo_candidate in BLOCKED_CANDIDATES:
        errors.append(f"candidate is blocked: {BLOCKED_CANDIDATES[repo_candidate]}")

    changed_days = int(summary["changed_days"])
    if changed_days > max_changed_days:
        errors.append(f"changed_days={changed_days} exceeds max_changed_days={max_changed_days}")

    rule = ALLOWED_CANDIDATES.get(repo_candidate)
    if rule:
        expected = rule.get("expected_changed_days")
        if expected is not None and changed_days != int(expected):
            errors.append(f"changed_days={changed_days} does not match expected_changed_days={expected}")
        max_for_candidate = rule.get("max_changed_days")
        if max_for_candidate is not None and changed_days > int(max_for_candidate):
            errors.append(
                f"changed_days={changed_days} exceeds candidate max_changed_days={max_for_candidate}"
            )
        return errors, info

    if repo_candidate not in BLOCKED_CANDIDATES:
        if not manifest:
            errors.append("candidate is not in static allowlist; pass --manifest for one-day model candidates")
            return errors, info
        try:
            manifest_row = _load_manifest_row(manifest, repo_candidate, candidate_sha256)
            manifest_errors, manifest_info = _guard_manifest_policy(
                manifest_row,
                summary,
                changed_actions,
                candidate_sha256,
            )
            errors.extend(manifest_errors)
            info.update(manifest_info)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
            info["manifest_status"] = "error"
    return errors, info


def guard_candidate(
    candidate: str,
    reference: str = DEFAULT_REFERENCE,
    max_changed_days: int = 5,
    reference_name: str = "safe5117",
    candidate_name: str = "candidate",
    manifest: str = "",
) -> tuple[dict[str, Any], pd.DataFrame]:
    repo_candidate = _as_repo_path(candidate)
    repo_reference = _as_repo_path(reference)
    candidate_sha256 = _sha256(candidate)

    check = check_submission(candidate)
    diff = compare_submissions(reference, candidate, reference_name, candidate_name)
    summary = summarize_diff(diff)
    changed_actions = _changed_actions(diff, reference_name, candidate_name)
    policy_errors, policy_info = _guard_policy(
        repo_candidate,
        summary,
        max_changed_days,
        changed_actions,
        candidate_sha256,
        manifest=manifest,
    )

    errors = list(check.errors) + policy_errors
    warnings = list(check.warnings)
    if warnings:
        errors.append("submission warnings are not allowed for steady-push candidates")

    rule = ALLOWED_CANDIDATES.get(repo_candidate, {})
    report = {
        "decision": "PASS" if not errors else "FAIL",
        "candidate": repo_candidate,
        "reference": repo_reference,
        "candidate_sha256": candidate_sha256,
        "reference_sha256": _sha256(reference),
        "baseline_score": BASELINE_SCORE,
        "stage": rule.get("stage", "not_allowed"),
        "next_step": rule.get("next_step", "Do not submit unless policy is updated."),
        **policy_info,
        "rows": check.rows,
        "days": check.days,
        "traded_days": check.traded_days,
        "check_errors": len(check.errors),
        "check_warnings": len(check.warnings),
        **summary,
        "changed_actions": changed_actions,
        "errors": errors,
    }
    return report, diff


def _print_report(report: dict[str, Any]) -> None:
    scalar_keys = [
        "decision",
        "candidate",
        "candidate_sha256",
        "reference",
        "reference_sha256",
        "baseline_score",
        "stage",
        "manifest_status",
        "manifest_date",
        "changed_date",
        "rows",
        "days",
        "traded_days",
        "check_errors",
        "check_warnings",
        "same_both_days",
        "changed_days",
        "next_step",
    ]
    for key in scalar_keys:
        if key in report:
            print(f"{key}={report[key]}")
    if report["changed_actions"]:
        print("changed_actions:")
        for row in report["changed_actions"]:
            print(
                f"  {row['date']}: {row['reference_action']} -> "
                f"{row['candidate_action']}"
            )
    if report["errors"]:
        print("errors:")
        for error in report["errors"]:
            print(f"  {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Guard a 2026-05-03 steady-push submission candidate.")
    parser.add_argument("--candidate", required=True, help="Candidate submission CSV.")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE, help="Verified 5117 baseline CSV.")
    parser.add_argument("--candidate-name", default="", help="Name used in diff output columns.")
    parser.add_argument("--reference-name", default="safe5117")
    parser.add_argument("--max-changed-days", type=int, default=5)
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest CSV for dynamically allowed one-day model candidates.",
    )
    parser.add_argument("--diff-output", default="", help="Optional path for full daily action diff CSV.")
    parser.add_argument("--summary-output", default="", help="Optional path for one-row guard summary CSV.")
    args = parser.parse_args()

    candidate_name = args.candidate_name or Path(args.candidate).stem
    report, diff = guard_candidate(
        candidate=args.candidate,
        reference=args.reference,
        max_changed_days=args.max_changed_days,
        reference_name=args.reference_name,
        candidate_name=candidate_name,
        manifest=args.manifest,
    )
    _print_report(report)

    if args.diff_output:
        diff_path = Path(args.diff_output)
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        diff.to_csv(diff_path, index=False)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        flat_report = {k: v for k, v in report.items() if k not in {"changed_actions", "errors"}}
        flat_report["changed_actions"] = "; ".join(
            f"{row['date']}:{row['reference_action']}->{row['candidate_action']}"
            for row in report["changed_actions"]
        )
        flat_report["errors"] = "; ".join(report["errors"])
        pd.DataFrame([flat_report]).to_csv(summary_path, index=False)

    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
