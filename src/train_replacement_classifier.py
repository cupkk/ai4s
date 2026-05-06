from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, build_features, fit_history_stats
from .nwp_features import load_or_build_nwp_features, merge_nwp_features
from .price_history_features import add_price_history_features, fit_price_history_features
from .replacement_classifier import (
    add_baseline_stability_features,
    CLASSIFIER_PARAMS,
    POSITIVE_LABEL_COL,
    PRED_EXPECTED_DELTA_COL,
    PRED_PROBA_COL,
    RISK_PROBA_COL,
    add_classifier_predictions,
    add_positive_delta_label,
    add_rule_risk_predictions_to_scored_windows,
    aggregate_replacement_metrics,
    baseline_meta_from_attached_windows,
    drop_attached_baseline_columns,
    fit_rule_risk_gate,
    fit_delta_calibrator,
    prepare_replacement_candidates,
    predict_binary,
    replacement_feature_columns,
    select_daily_replacements,
    select_stage1_candidate_rows,
)
from .safe5117_baseline import fit_safe5117_source_baseline
from .train_lgb import load_training_frame, params_for_seed, parse_seeds, split_by_day, train_booster
from .make_safe5117_like_baseline_meta import learn_charge_offset_map
from .train_window_ranker import (
    DEFAULT_BASELINE_WINDOW_SCORE_COL,
    DEFAULT_POINT_FEATURES,
    attach_baseline_window_context,
    build_window_dataset,
    prepare_baseline_windows,
)


def _window_dataset_for_frame(
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    history_stats: dict[str, object],
    price_stats: dict[str, object],
    target_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    baseline_window_score_col: str,
    baseline_meta: str = "",
    include_target: bool = True,
) -> pd.DataFrame:
    model_with_price = add_price_history_features(model_df, price_stats)
    features = build_features(model_with_price, history_stats=history_stats)
    point_features = [col for col in DEFAULT_POINT_FEATURES if col in features.frame.columns]
    if not point_features:
        raise ValueError("no point features available for replacement classifier")
    windows, _ = build_window_dataset(
        raw_df,
        features.frame,
        point_features,
        target_col,
        charge_start_min,
        charge_start_max,
        discharge_start_min,
        discharge_start_max,
        include_target=include_target,
    )
    baseline = prepare_baseline_windows(
        windows,
        score_col=baseline_window_score_col,
        meta_path=baseline_meta,
    )
    return attach_baseline_window_context(windows, baseline)


def _test_window_dataset(
    test_df: pd.DataFrame,
    history_stats: dict[str, object],
    price_stats: dict[str, object],
    target_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    test_baseline_submission: str,
    baseline_window_score_col: str,
) -> pd.DataFrame:
    model_df = add_price_history_features(test_df, price_stats)
    features = build_features(model_df, history_stats=history_stats)
    point_features = [col for col in DEFAULT_POINT_FEATURES if col in features.frame.columns]
    if not point_features:
        raise ValueError("no point features available for replacement classifier test data")
    windows, _ = build_window_dataset(
        test_df,
        features.frame,
        point_features,
        target_col,
        charge_start_min,
        charge_start_max,
        discharge_start_min,
        discharge_start_max,
        include_target=False,
    )
    baseline = prepare_baseline_windows(
        windows,
        score_col=baseline_window_score_col,
        submission_path=test_baseline_submission,
    )
    return attach_baseline_window_context(windows, baseline)


def _make_safe5117_like_meta_from_windows(
    windows: pd.DataFrame,
    safe_submission: str,
    source_submission: str,
    min_group_count: int,
) -> pd.DataFrame:
    offset_map, _ = learn_charge_offset_map(
        safe_submission,
        source_submission,
        min_group_count=min_group_count,
    )
    idx = windows.groupby("date", sort=True)[DEFAULT_BASELINE_WINDOW_SCORE_COL].idxmax()
    selected = windows.loc[idx].copy().sort_values("date").reset_index(drop=True)
    rows = []
    for _, row in selected.iterrows():
        discharge_start = int(row["discharge_start"])
        raw_charge_start = int(row["charge_start"])
        offset = int(offset_map.get(discharge_start, 0))
        corrected_charge = max(0, min(discharge_start - 8, raw_charge_start + offset))
        rows.append(
            {
                "date": str(row["date"]),
                "raw_charge_start": raw_charge_start,
                "raw_discharge_start": discharge_start,
                "charge_offset": offset,
                "baseline_charge_start": int(corrected_charge),
                "baseline_discharge_start": discharge_start,
            }
        )
    return pd.DataFrame(rows)


def _filter_windows_to_baseline_dates(windows: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    baseline_dates = set(baseline["date"].astype(str))
    out = windows.loc[windows["date"].astype(str).isin(baseline_dates)].copy()
    if out.empty:
        raise ValueError("no window rows remain after filtering to baseline meta dates")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a conservative classifier for one-day replacements near the safe baseline window."
    )
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--test-feature", default="")
    parser.add_argument("--target-col", default=TARGET_COL)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--test-nwp-cache", default="outputs/nwp_features_all.csv")
    parser.add_argument("--val-start-date", default="")
    parser.add_argument("--val-end-date", default="")
    parser.add_argument("--val-days", type=int, default=59)
    parser.add_argument("--seeds", default="42,2024,2026")
    parser.add_argument("--max-shift", type=int, default=8)
    parser.add_argument(
        "--daily-top-k",
        type=int,
        default=0,
        help="Keep only this many prior-ranked near-baseline windows per day before classifier training. 0 disables filtering.",
    )
    parser.add_argument("--topk-score-col", default="daily_topk_prior_score")
    parser.add_argument("--positive-delta-threshold", type=float, default=0.0)
    parser.add_argument("--proba-threshold", type=float, default=0.70)
    parser.add_argument("--min-expected-delta", type=float, default=0.0)
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--max-proba-std", type=float, default=None)
    parser.add_argument("--use-rule-risk-gate", action="store_true")
    parser.add_argument("--risk-proba-threshold", type=float, default=1.0)
    parser.add_argument("--min-risk-expected-delta", type=float, default=0.0)
    parser.add_argument("--use-baseline-stability-gate", action="store_true")
    parser.add_argument("--baseline-stability-max-abs-delta", type=int, default=2)
    parser.add_argument("--charge-start-min", type=int, default=0)
    parser.add_argument("--charge-start-max", type=int, default=80)
    parser.add_argument("--discharge-start-min", type=int, default=8)
    parser.add_argument("--discharge-start-max", type=int, default=88)
    parser.add_argument("--baseline-window-score-col", default=DEFAULT_BASELINE_WINDOW_SCORE_COL)
    parser.add_argument(
        "--baseline-mode",
        choices=["proxy", "safe5117-like", "safe5117-source-model"],
        default="proxy",
    )
    parser.add_argument("--safe-submission", default="outputs/output_nwp_unconstrained_online5117.csv")
    parser.add_argument("--source-submission", default="outputs/output_nwp_constrained.csv")
    parser.add_argument("--min-group-count", type=int, default=2)
    parser.add_argument("--source-num-boost-round", type=int, default=500)
    parser.add_argument("--source-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--source-val-days", type=int, default=59)
    parser.add_argument(
        "--source-train-baseline-mode",
        choices=["recent-oof", "in-sample"],
        default="recent-oof",
    )
    parser.add_argument("--source-threshold", type=float, default=-1.0e18)
    parser.add_argument("--train-baseline-meta", default="")
    parser.add_argument("--val-baseline-meta", default="")
    parser.add_argument("--test-baseline-submission", default="")
    parser.add_argument("--num-boost-round", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument("--params-json", default="")
    parser.add_argument("--model-output", default="outputs/replacement_classifier_model.txt")
    parser.add_argument("--metadata-output", default="outputs/replacement_classifier_metadata.json")
    parser.add_argument("--val-window-output", default="outputs/val_windows_replacement_classifier.csv")
    parser.add_argument("--val-day-metrics-output", default="outputs/val_replacement_classifier_day_metrics.csv")
    parser.add_argument("--test-window-output", default="outputs/test_windows_replacement_classifier.csv")
    args = parser.parse_args()

    params = CLASSIFIER_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))
    seeds = parse_seeds(args.seeds)

    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=args.target_col,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    train_df, val_df = split_by_day(
        df,
        val_ratio=0.2,
        val_start_date=args.val_start_date,
        val_end_date=args.val_end_date,
        val_days=args.val_days,
    )
    price_stats = fit_price_history_features(train_df, target_col=args.target_col)
    hist_stats = fit_history_stats(add_price_history_features(train_df, price_stats), target_col=args.target_col)

    train_windows = _window_dataset_for_frame(
        raw_df=train_df,
        model_df=train_df,
        history_stats=hist_stats,
        price_stats=price_stats,
        target_col=args.target_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        baseline_window_score_col=args.baseline_window_score_col,
        baseline_meta=args.train_baseline_meta,
        include_target=True,
    )
    val_windows = _window_dataset_for_frame(
        raw_df=val_df,
        model_df=val_df,
        history_stats=hist_stats,
        price_stats=price_stats,
        target_col=args.target_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        baseline_window_score_col=args.baseline_window_score_col,
        baseline_meta=args.val_baseline_meta,
        include_target=True,
    )
    train_stability_reference = baseline_meta_from_attached_windows(train_windows)
    val_stability_reference = baseline_meta_from_attached_windows(val_windows)
    if args.baseline_mode == "safe5117-like":
        train_source = _make_safe5117_like_meta_from_windows(
            train_windows,
            safe_submission=args.safe_submission,
            source_submission=args.source_submission,
            min_group_count=args.min_group_count,
        )
        val_source = _make_safe5117_like_meta_from_windows(
            val_windows,
            safe_submission=args.safe_submission,
            source_submission=args.source_submission,
            min_group_count=args.min_group_count,
        )
        train_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(train_windows),
            train_source,
        )
        val_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(val_windows),
            val_source,
        )
    elif args.baseline_mode == "safe5117-source-model":
        source_baseline = fit_safe5117_source_baseline(
            train_df=train_df,
            predict_df=val_df,
            target_col=args.target_col,
            seeds=seeds,
            num_boost_round=args.source_num_boost_round,
            early_stopping_rounds=args.source_early_stopping_rounds,
            source_val_days=args.source_val_days,
            train_baseline_mode=args.source_train_baseline_mode,
            threshold=args.source_threshold,
        )
        train_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(
                _filter_windows_to_baseline_dates(train_windows, source_baseline.train_meta)
            ),
            source_baseline.train_meta,
        )
        val_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(val_windows),
            source_baseline.predict_meta,
        )
        source_artifact_prefix = Path(args.val_window_output).with_suffix("")
        source_baseline.train_meta.to_csv(
            source_artifact_prefix.with_name(f"{source_artifact_prefix.name}_source_train_meta.csv"),
            index=False,
        )
        source_baseline.predict_meta.to_csv(
            source_artifact_prefix.with_name(f"{source_artifact_prefix.name}_source_val_meta.csv"),
            index=False,
        )
        source_baseline.train_predictions.to_csv(
            source_artifact_prefix.with_name(f"{source_artifact_prefix.name}_source_train_predictions.csv"),
            index=False,
        )
        source_baseline.predict_predictions.to_csv(
            source_artifact_prefix.with_name(f"{source_artifact_prefix.name}_source_val_predictions.csv"),
            index=False,
        )
    train_candidates = prepare_replacement_candidates(
        train_windows,
        max_shift=args.max_shift,
        daily_top_k=args.daily_top_k,
        topk_score_col=args.topk_score_col,
    )
    val_candidates = prepare_replacement_candidates(
        val_windows,
        max_shift=args.max_shift,
        daily_top_k=args.daily_top_k,
        topk_score_col=args.topk_score_col,
    )
    if args.use_baseline_stability_gate:
        train_candidates = add_baseline_stability_features(
            train_candidates,
            train_stability_reference,
            max_abs_delta=args.baseline_stability_max_abs_delta,
        )
        val_candidates = add_baseline_stability_features(
            val_candidates,
            val_stability_reference,
            max_abs_delta=args.baseline_stability_max_abs_delta,
        )
    train_near = add_positive_delta_label(
        train_candidates,
        positive_delta_threshold=args.positive_delta_threshold,
    )
    val_near = add_positive_delta_label(
        val_candidates,
        positive_delta_threshold=args.positive_delta_threshold,
    )
    feature_columns = replacement_feature_columns(train_near)
    train_x = train_near[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    val_x = val_near[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_y = train_near[POSITIVE_LABEL_COL].to_numpy(dtype=int)
    val_y = val_near[POSITIVE_LABEL_COL].to_numpy(dtype=int)

    val_preds = []
    train_preds = []
    best_iterations: list[int] = []
    final_train_models = []
    for seed in seeds:
        model = train_booster(
            train_x,
            train_y,
            val_x,
            val_y,
            params=params_for_seed(params, seed),
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        best_iteration = int(model.best_iteration or args.num_boost_round)
        best_iterations.append(best_iteration)
        train_preds.append(predict_binary(model, train_x, best_iteration))
        val_preds.append(predict_binary(model, val_x, best_iteration))
        final_train_models.append(model)

    train_proba = np.mean(np.vstack(train_preds), axis=0)
    calibrator = fit_delta_calibrator(
        train_proba,
        train_near["true_delta_profit"].to_numpy(dtype=float),
    )
    val_scored = add_classifier_predictions(val_near, val_preds, seeds, calibrator)
    rule_risk_gate = None
    if args.use_rule_risk_gate:
        train_scored = add_classifier_predictions(train_near, train_preds, seeds, calibrator)
        train_stage1 = select_stage1_candidate_rows(
            train_scored,
            proba_threshold=args.proba_threshold,
            min_expected_delta=args.min_expected_delta,
            min_margin=args.min_margin,
            max_proba_std=args.max_proba_std,
            require_baseline_stability=args.use_baseline_stability_gate,
        )
        rule_risk_gate = fit_rule_risk_gate(train_stage1)
        val_scored = add_rule_risk_predictions_to_scored_windows(val_scored, rule_risk_gate)
    day_metrics = select_daily_replacements(
        val_scored,
        proba_threshold=args.proba_threshold,
        min_expected_delta=args.min_expected_delta,
        min_margin=args.min_margin,
        max_proba_std=args.max_proba_std,
        risk_proba_threshold=args.risk_proba_threshold if args.use_rule_risk_gate else None,
        min_risk_expected_delta=args.min_risk_expected_delta if args.use_rule_risk_gate else None,
        require_baseline_stability=args.use_baseline_stability_gate,
    )
    summary = aggregate_replacement_metrics(day_metrics)
    print("replacement_classifier_validation=" + ", ".join(f"{k}={v}" for k, v in summary.items()))

    Path(args.val_window_output).parent.mkdir(parents=True, exist_ok=True)
    val_scored.to_csv(args.val_window_output, index=False)
    Path(args.val_day_metrics_output).parent.mkdir(parents=True, exist_ok=True)
    day_metrics.to_csv(args.val_day_metrics_output, index=False)

    full_price_stats = fit_price_history_features(df, target_col=args.target_col)
    full_hist_stats = fit_history_stats(add_price_history_features(df, full_price_stats), target_col=args.target_col)
    full_windows = _window_dataset_for_frame(
        raw_df=df,
        model_df=df,
        history_stats=full_hist_stats,
        price_stats=full_price_stats,
        target_col=args.target_col,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        baseline_window_score_col=args.baseline_window_score_col,
        baseline_meta=args.train_baseline_meta,
        include_target=True,
    )
    full_stability_reference = baseline_meta_from_attached_windows(full_windows)
    if args.baseline_mode == "safe5117-like":
        full_source = _make_safe5117_like_meta_from_windows(
            full_windows,
            safe_submission=args.safe_submission,
            source_submission=args.source_submission,
            min_group_count=args.min_group_count,
        )
        full_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(full_windows),
            full_source,
        )
    elif args.baseline_mode == "safe5117-source-model":
        full_source_baseline = fit_safe5117_source_baseline(
            train_df=df,
            predict_df=df,
            target_col=args.target_col,
            seeds=seeds,
            num_boost_round=args.source_num_boost_round,
            early_stopping_rounds=args.source_early_stopping_rounds,
            source_val_days=args.source_val_days,
            train_baseline_mode=args.source_train_baseline_mode,
            threshold=args.source_threshold,
        )
        full_windows = attach_baseline_window_context(
            drop_attached_baseline_columns(
                _filter_windows_to_baseline_dates(full_windows, full_source_baseline.train_meta)
            ),
            full_source_baseline.train_meta,
        )
        full_source_prefix = Path(args.model_output).with_suffix("")
        full_source_baseline.train_meta.to_csv(
            full_source_prefix.with_name(f"{full_source_prefix.name}_safe5117_source_full_meta.csv"),
            index=False,
        )
        full_source_baseline.train_predictions.to_csv(
            full_source_prefix.with_name(f"{full_source_prefix.name}_safe5117_source_full_predictions.csv"),
            index=False,
        )
    full_candidates = prepare_replacement_candidates(
        full_windows,
        max_shift=args.max_shift,
        daily_top_k=args.daily_top_k,
        topk_score_col=args.topk_score_col,
    )
    if args.use_baseline_stability_gate:
        full_candidates = add_baseline_stability_features(
            full_candidates,
            full_stability_reference,
            max_abs_delta=args.baseline_stability_max_abs_delta,
        )
    full_near = add_positive_delta_label(
        full_candidates,
        positive_delta_threshold=args.positive_delta_threshold,
    )
    full_feature_columns = replacement_feature_columns(full_near)
    full_x = full_near[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    full_y = full_near[POSITIVE_LABEL_COL].to_numpy(dtype=int)

    model_paths = []
    final_models = []
    full_train_preds = []
    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    for seed, rounds in zip(seeds, best_iterations):
        model = train_booster(
            full_x,
            full_y,
            None,
            None,
            params=params_for_seed(params, seed),
            num_boost_round=rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        path = Path(args.model_output)
        model_path = str(path.with_name(f"{path.stem}_seed{seed}{path.suffix}"))
        model.save_model(model_path)
        model_paths.append(model_path)
        final_models.append(model)
        full_train_preds.append(predict_binary(model, full_x, rounds))

    full_calibrator = fit_delta_calibrator(
        np.mean(np.vstack(full_train_preds), axis=0),
        full_near["true_delta_profit"].to_numpy(dtype=float),
    )
    full_rule_risk_gate = None
    if args.use_rule_risk_gate:
        full_scored_for_gate = add_classifier_predictions(
            full_near,
            full_train_preds,
            seeds,
            full_calibrator,
        )
        full_stage1_for_gate = select_stage1_candidate_rows(
            full_scored_for_gate,
            proba_threshold=args.proba_threshold,
            min_expected_delta=args.min_expected_delta,
            min_margin=args.min_margin,
            max_proba_std=args.max_proba_std,
            require_baseline_stability=args.use_baseline_stability_gate,
        )
        full_rule_risk_gate = fit_rule_risk_gate(full_stage1_for_gate)

    metadata = {
        "model_paths": model_paths,
        "seeds": seeds,
        "feature_columns": full_feature_columns,
        "max_shift": args.max_shift,
        "daily_top_k": args.daily_top_k,
        "topk_score_col": args.topk_score_col,
        "positive_delta_threshold": args.positive_delta_threshold,
        "proba_threshold": args.proba_threshold,
        "min_expected_delta": args.min_expected_delta,
        "min_margin": args.min_margin,
        "max_proba_std": args.max_proba_std,
        "use_rule_risk_gate": args.use_rule_risk_gate,
        "risk_proba_threshold": args.risk_proba_threshold,
        "min_risk_expected_delta": args.min_risk_expected_delta,
        "use_baseline_stability_gate": args.use_baseline_stability_gate,
        "baseline_stability_max_abs_delta": args.baseline_stability_max_abs_delta,
        "validation_rule_risk_gate": rule_risk_gate,
        "full_rule_risk_gate": full_rule_risk_gate,
        "baseline_window_score_col": args.baseline_window_score_col,
        "baseline_mode": args.baseline_mode,
        "safe_submission": args.safe_submission,
        "source_submission": args.source_submission,
        "min_group_count": args.min_group_count,
        "source_num_boost_round": args.source_num_boost_round,
        "source_early_stopping_rounds": args.source_early_stopping_rounds,
        "source_val_days": args.source_val_days,
        "source_train_baseline_mode": args.source_train_baseline_mode,
        "source_threshold": args.source_threshold,
        "best_iterations": best_iterations,
        "validation_summary": summary,
        "calibrator": full_calibrator.to_jsonable(),
    }
    Path(args.metadata_output).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.test_feature:
        if not args.test_baseline_submission:
            raise ValueError("--test-baseline-submission is required when --test-feature is used")
        test_df = pd.read_csv(args.test_feature)
        test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL])
        if args.nwp_dir:
            nwp = load_or_build_nwp_features(
                args.nwp_dir,
                args.test_nwp_cache,
                start_time=str(test_df[TIME_COL].min()),
                end_time=str(test_df[TIME_COL].max()),
            )
            test_df = merge_nwp_features(test_df, nwp)
        test_stability_reference = None
        if args.use_baseline_stability_gate and args.baseline_mode == "safe5117-source-model":
            test_source_baseline = fit_safe5117_source_baseline(
                train_df=df,
                predict_df=test_df,
                target_col=args.target_col,
                seeds=seeds,
                num_boost_round=args.source_num_boost_round,
                early_stopping_rounds=args.source_early_stopping_rounds,
                source_val_days=args.source_val_days,
                train_baseline_mode=args.source_train_baseline_mode,
                threshold=args.source_threshold,
            )
            test_stability_reference = test_source_baseline.predict_meta
            test_source_prefix = Path(args.test_window_output).with_suffix("")
            test_source_baseline.predict_meta.to_csv(
                test_source_prefix.with_name(f"{test_source_prefix.name}_source_test_meta.csv"),
                index=False,
            )
            test_source_baseline.predict_predictions.to_csv(
                test_source_prefix.with_name(f"{test_source_prefix.name}_source_test_predictions.csv"),
                index=False,
            )
        test_windows = _test_window_dataset(
            test_df=test_df,
            history_stats=full_hist_stats,
            price_stats=full_price_stats,
            target_col=args.target_col,
            charge_start_min=args.charge_start_min,
            charge_start_max=args.charge_start_max,
            discharge_start_min=args.discharge_start_min,
            discharge_start_max=args.discharge_start_max,
            test_baseline_submission=args.test_baseline_submission,
            baseline_window_score_col=args.baseline_window_score_col,
        )
        if test_stability_reference is None:
            test_stability_reference = baseline_meta_from_attached_windows(test_windows)
        test_near = prepare_replacement_candidates(
            test_windows,
            max_shift=args.max_shift,
            daily_top_k=args.daily_top_k,
            topk_score_col=args.topk_score_col,
        )
        if args.use_baseline_stability_gate:
            test_near = add_baseline_stability_features(
                test_near,
                test_stability_reference,
                max_abs_delta=args.baseline_stability_max_abs_delta,
            )
        test_x = test_near[full_feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        test_preds = [
            predict_binary(model, test_x, rounds)
            for model, rounds in zip(final_models, best_iterations)
        ]
        test_scored = add_classifier_predictions(test_near, test_preds, seeds, full_calibrator)
        if args.use_rule_risk_gate:
            if full_rule_risk_gate is None:
                raise ValueError("rule risk gate was not fitted")
            test_scored = add_rule_risk_predictions_to_scored_windows(
                test_scored,
                full_rule_risk_gate,
            )
        test_selected = select_daily_replacements(
            test_scored,
            proba_threshold=args.proba_threshold,
            min_expected_delta=args.min_expected_delta,
            min_margin=args.min_margin,
            max_proba_std=args.max_proba_std,
            risk_proba_threshold=args.risk_proba_threshold if args.use_rule_risk_gate else None,
            min_risk_expected_delta=args.min_risk_expected_delta if args.use_rule_risk_gate else None,
            require_baseline_stability=args.use_baseline_stability_gate,
        )
        test_scored["pred_rank"] = 999999.0
        selected = test_selected.loc[test_selected["proposed"].astype(bool)].copy()
        if not selected.empty:
            rank_keys = selected.loc[:, ["date", "candidate_charge_start", "candidate_discharge_start"]].rename(
                columns={
                    "candidate_charge_start": "charge_start",
                    "candidate_discharge_start": "discharge_start",
                }
            )
            rank_keys["date"] = rank_keys["date"].astype(str)
            rank_keys["charge_start"] = rank_keys["charge_start"].astype(int)
            rank_keys["discharge_start"] = rank_keys["discharge_start"].astype(int)
            test_scored = test_scored.merge(
                rank_keys.assign(__selected_rank__=1.0),
                on=["date", "charge_start", "discharge_start"],
                how="left",
            )
            test_scored.loc[test_scored["__selected_rank__"].eq(1.0), "pred_rank"] = 1.0
            test_scored = test_scored.drop(columns=["__selected_rank__"])
        Path(args.test_window_output).parent.mkdir(parents=True, exist_ok=True)
        test_scored.to_csv(args.test_window_output, index=False)
        print(f"saved_replacement_classifier_test_windows={args.test_window_output}")


if __name__ == "__main__":
    main()
