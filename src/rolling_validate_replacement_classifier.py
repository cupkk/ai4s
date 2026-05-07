from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .features import TARGET_COL, TIME_COL, build_features, fit_history_stats
from .price_history_features import add_price_history_features, fit_price_history_features
from .replacement_classifier import (
    add_baseline_stability_features,
    CLASSIFIER_PARAMS,
    POSITIVE_LABEL_COL,
    RISK_EXPECTED_DELTA_COL,
    RISK_PROBA_COL,
    add_risk_predictions_to_scored_windows,
    add_risk_regression_predictions_to_scored_windows,
    add_rule_risk_predictions_to_scored_windows,
    add_classifier_predictions,
    add_positive_delta_label,
    aggregate_replacement_metrics,
    baseline_meta_from_attached_windows,
    drop_attached_baseline_columns,
    fit_delta_calibrator,
    fit_rule_risk_gate,
    prepare_replacement_candidates,
    predict_binary,
    replacement_feature_columns,
    risk_feature_columns,
    select_daily_replacements,
    select_stage1_candidate_rows,
)
from .rolling_validate_window_ranker import DEFAULT_FOLDS, parse_folds
from .safe5117_baseline import fit_safe5117_source_baseline
from .train_lgb import load_training_frame, params_for_seed, parse_seeds, train_booster
from .train_window_ranker import (
    DEFAULT_BASELINE_WINDOW_SCORE_COL,
    DEFAULT_POINT_FEATURES,
    attach_baseline_window_context,
    build_window_dataset,
    prepare_baseline_windows,
)
from .make_safe5117_like_baseline_meta import learn_charge_offset_map


def _build_windows(
    raw_df: pd.DataFrame,
    price_stats: dict[str, object],
    hist_stats: dict[str, object],
    target_col: str,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    baseline_window_score_col: str,
    baseline_meta: str = "",
) -> pd.DataFrame:
    model_df = add_price_history_features(raw_df, price_stats)
    features = build_features(model_df, history_stats=hist_stats)
    point_features = [col for col in DEFAULT_POINT_FEATURES if col in features.frame.columns]
    if not point_features:
        raise ValueError("no point features available for rolling replacement classifier")
    windows, _ = build_window_dataset(
        raw_df,
        features.frame,
        point_features,
        target_col,
        charge_start_min,
        charge_start_max,
        discharge_start_min,
        discharge_start_max,
        include_target=True,
    )
    baseline = prepare_baseline_windows(
        windows,
        score_col=baseline_window_score_col,
        meta_path=baseline_meta,
    )
    return attach_baseline_window_context(windows, baseline)


def _make_safe5117_like_meta_from_windows(
    windows: pd.DataFrame,
    safe_submission: str,
    source_submission: str,
    min_group_count: int,
) -> pd.DataFrame:
    if DEFAULT_BASELINE_WINDOW_SCORE_COL not in windows.columns:
        raise ValueError(f"safe5117-like source windows missing {DEFAULT_BASELINE_WINDOW_SCORE_COL}")
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


def aggregate_by_fold(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, group in detail.groupby("fold", sort=True):
        row = aggregate_replacement_metrics(group)
        row.update(
            {
                "fold": int(fold),
                "val_start": group["val_start"].iloc[0],
                "val_end": group["val_end"].iloc[0],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)[
        [
            "fold",
            "val_start",
            "val_end",
            "days",
            "proposed_days",
            "positive_selected_days",
            "false_positive_days",
            "false_positive_rate",
            "avg_delta_all_days",
            "avg_delta_proposed_days",
            "total_delta_profit",
            "worst_selected_delta",
            "avg_pred_positive_proba",
            "avg_pred_expected_delta",
        ]
    ]


def run_rolling_validation(
    df: pd.DataFrame,
    folds: Sequence[tuple[pd.Timestamp, pd.Timestamp]],
    seeds: Sequence[int],
    params: dict[str, object],
    max_shift: int,
    daily_top_k: int,
    topk_score_col: str,
    positive_delta_threshold: float,
    proba_threshold: float,
    min_expected_delta: float,
    min_margin: float,
    max_proba_std: float | None,
    use_risk_gate: bool,
    risk_objective: str,
    risk_proba_threshold: float,
    min_risk_expected_delta: float,
    rule_shape_spike_max: float | None,
    rule_shape_plateau_min: float | None,
    rule_shape_balance_min: float | None,
    use_baseline_stability_gate: bool,
    baseline_stability_max_abs_delta: int,
    charge_start_min: int,
    charge_start_max: int,
    discharge_start_min: int,
    discharge_start_max: int,
    baseline_window_score_col: str,
    baseline_mode: str,
    safe_submission: str,
    source_submission: str,
    min_group_count: int,
    source_num_boost_round: int,
    source_early_stopping_rounds: int,
    source_val_days: int,
    source_train_baseline_mode: str,
    source_threshold: float,
    num_boost_round: int,
    early_stopping_rounds: int,
    scored_output_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_day_metrics = []
    scored_dir = Path(scored_output_dir) if scored_output_dir else None
    if scored_dir:
        scored_dir.mkdir(parents=True, exist_ok=True)

    for fold_index, (start, end) in enumerate(folds, start=1):
        train_df = df[df[TIME_COL] < start].copy()
        val_df = df[(df[TIME_COL] >= start) & (df[TIME_COL] <= end + pd.Timedelta(hours=23, minutes=45))].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(f"fold {fold_index} has empty train or validation data: {start}..{end}")

        price_stats = fit_price_history_features(train_df, target_col=TARGET_COL)
        hist_stats = fit_history_stats(add_price_history_features(train_df, price_stats), target_col=TARGET_COL)
        train_windows = _build_windows(
            train_df,
            price_stats,
            hist_stats,
            TARGET_COL,
            charge_start_min,
            charge_start_max,
            discharge_start_min,
            discharge_start_max,
            baseline_window_score_col,
        )
        val_windows = _build_windows(
            val_df,
            price_stats,
            hist_stats,
            TARGET_COL,
            charge_start_min,
            charge_start_max,
            discharge_start_min,
            discharge_start_max,
            baseline_window_score_col,
        )
        train_stability_reference = baseline_meta_from_attached_windows(train_windows)
        val_stability_reference = baseline_meta_from_attached_windows(val_windows)
        if baseline_mode == "safe5117-like":
            train_source = _make_safe5117_like_meta_from_windows(
                train_windows,
                safe_submission=safe_submission,
                source_submission=source_submission,
                min_group_count=min_group_count,
            )
            val_source = _make_safe5117_like_meta_from_windows(
                val_windows,
                safe_submission=safe_submission,
                source_submission=source_submission,
                min_group_count=min_group_count,
            )
            train_windows = attach_baseline_window_context(
                drop_attached_baseline_columns(train_windows),
                train_source,
            )
            val_windows = attach_baseline_window_context(
                drop_attached_baseline_columns(val_windows),
                val_source,
            )
        elif baseline_mode == "safe5117-source-model":
            source_baseline = fit_safe5117_source_baseline(
                train_df=train_df,
                predict_df=val_df,
                target_col=TARGET_COL,
                seeds=seeds,
                num_boost_round=source_num_boost_round,
                early_stopping_rounds=source_early_stopping_rounds,
                source_val_days=source_val_days,
                train_baseline_mode=source_train_baseline_mode,
                threshold=source_threshold,
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
            if scored_dir:
                source_baseline.train_meta.to_csv(
                    scored_dir / f"fold_{fold_index:02d}_safe5117_source_train_meta.csv",
                    index=False,
                )
                source_baseline.predict_meta.to_csv(
                    scored_dir / f"fold_{fold_index:02d}_safe5117_source_val_meta.csv",
                    index=False,
                )
                source_baseline.train_predictions.to_csv(
                    scored_dir / f"fold_{fold_index:02d}_safe5117_source_train_predictions.csv",
                    index=False,
                )
                source_baseline.predict_predictions.to_csv(
                    scored_dir / f"fold_{fold_index:02d}_safe5117_source_val_predictions.csv",
                    index=False,
                )
        train_candidates = prepare_replacement_candidates(
            train_windows,
            max_shift=max_shift,
            daily_top_k=daily_top_k,
            topk_score_col=topk_score_col,
        )
        val_candidates = prepare_replacement_candidates(
            val_windows,
            max_shift=max_shift,
            daily_top_k=daily_top_k,
            topk_score_col=topk_score_col,
        )
        if use_baseline_stability_gate:
            train_candidates = add_baseline_stability_features(
                train_candidates,
                train_stability_reference,
                max_abs_delta=baseline_stability_max_abs_delta,
            )
            val_candidates = add_baseline_stability_features(
                val_candidates,
                val_stability_reference,
                max_abs_delta=baseline_stability_max_abs_delta,
            )
        train_near = add_positive_delta_label(
            train_candidates,
            positive_delta_threshold=positive_delta_threshold,
        )
        val_near = add_positive_delta_label(
            val_candidates,
            positive_delta_threshold=positive_delta_threshold,
        )
        feature_columns = replacement_feature_columns(train_near)
        train_x = train_near[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        val_x = val_near[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        train_y = train_near[POSITIVE_LABEL_COL].to_numpy(dtype=int)
        val_y = val_near[POSITIVE_LABEL_COL].to_numpy(dtype=int)

        train_preds = []
        val_preds = []
        for seed in seeds:
            model = train_booster(
                train_x,
                train_y,
                val_x,
                val_y,
                params=params_for_seed(params, seed),
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
            )
            best_iteration = int(model.best_iteration or num_boost_round)
            train_preds.append(predict_binary(model, train_x, best_iteration))
            val_preds.append(predict_binary(model, val_x, best_iteration))

        calibrator = fit_delta_calibrator(
            np.mean(np.vstack(train_preds), axis=0),
            train_near["true_delta_profit"].to_numpy(dtype=float),
        )
        scored = add_classifier_predictions(val_near, val_preds, seeds, calibrator)
        risk_feature_cols: list[str] = []
        if use_risk_gate:
            train_scored = add_classifier_predictions(train_near, train_preds, seeds, calibrator)
            train_stage1 = select_stage1_candidate_rows(
                train_scored,
                proba_threshold=proba_threshold,
                min_expected_delta=min_expected_delta,
                min_margin=min_margin,
                max_proba_std=max_proba_std,
                require_baseline_stability=use_baseline_stability_gate,
            )
            val_stage1 = select_stage1_candidate_rows(
                scored,
                proba_threshold=proba_threshold,
                min_expected_delta=min_expected_delta,
                min_margin=min_margin,
                max_proba_std=max_proba_std,
                require_baseline_stability=use_baseline_stability_gate,
            )
            if train_stage1.empty or val_stage1.empty:
                scored[RISK_PROBA_COL] = 0.0
                scored[f"{RISK_PROBA_COL}_std"] = 0.0
                scored[RISK_EXPECTED_DELTA_COL] = 0.0
            else:
                risk_feature_cols = risk_feature_columns(train_stage1)
                risk_train_x = train_stage1[risk_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                risk_val_x = val_stage1[risk_feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if risk_objective == "rule":
                    gate = fit_rule_risk_gate(train_stage1)
                    scored = add_rule_risk_predictions_to_scored_windows(
                        scored,
                        gate,
                        post_shape_spike_max=rule_shape_spike_max,
                        post_shape_plateau_min=rule_shape_plateau_min,
                        post_shape_balance_min=rule_shape_balance_min,
                    )
                    risk_feature_cols = [
                        "delta_vs_baseline_spread_net_load",
                        "delta_vs_baseline_spread_hist_slot_mean_daily_centered",
                    ]
                elif risk_objective == "classification":
                    risk_train_y = train_stage1[POSITIVE_LABEL_COL].to_numpy(dtype=int)
                    risk_val_y = val_stage1[POSITIVE_LABEL_COL].to_numpy(dtype=int)
                    risk_params = params
                elif risk_objective == "regression":
                    risk_train_y = train_stage1["true_delta_profit"].to_numpy(dtype=float)
                    risk_val_y = val_stage1["true_delta_profit"].to_numpy(dtype=float)
                    risk_params = {
                        key: value
                        for key, value in params.items()
                        if key not in {"objective", "metric"}
                    }
                    risk_params.update({"objective": "regression", "metric": "rmse"})
                else:
                    raise ValueError(f"unsupported risk objective: {risk_objective}")
                if risk_objective != "rule":
                    risk_train_preds = []
                    risk_val_preds = []
                    for seed in seeds:
                        risk_model = train_booster(
                            risk_train_x,
                            risk_train_y,
                            risk_val_x,
                            risk_val_y,
                            params=params_for_seed(risk_params, seed),
                            num_boost_round=num_boost_round,
                            early_stopping_rounds=early_stopping_rounds,
                        )
                        risk_best_iteration = int(risk_model.best_iteration or num_boost_round)
                        risk_train_preds.append(predict_binary(risk_model, risk_train_x, risk_best_iteration))
                        risk_val_preds.append(predict_binary(risk_model, risk_val_x, risk_best_iteration))
                    if risk_objective == "classification":
                        risk_calibrator = fit_delta_calibrator(
                            np.mean(np.vstack(risk_train_preds), axis=0),
                            train_stage1["true_delta_profit"].to_numpy(dtype=float),
                        )
                        scored = add_risk_predictions_to_scored_windows(
                            scored,
                            val_stage1,
                            risk_val_preds,
                            seeds,
                            risk_calibrator,
                        )
                    else:
                        scored = add_risk_regression_predictions_to_scored_windows(
                            scored,
                            val_stage1,
                            risk_val_preds,
                            seeds,
                        )
        if scored_dir:
            scored.to_csv(scored_dir / f"fold_{fold_index:02d}_scored_windows.csv", index=False)
            if use_risk_gate and risk_feature_cols:
                (scored_dir / f"fold_{fold_index:02d}_risk_feature_columns.json").write_text(
                    json.dumps(risk_feature_cols, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        day_metrics = select_daily_replacements(
            scored,
            proba_threshold=proba_threshold,
            min_expected_delta=min_expected_delta,
            min_margin=min_margin,
            max_proba_std=max_proba_std,
            risk_proba_threshold=risk_proba_threshold if use_risk_gate else None,
            min_risk_expected_delta=min_risk_expected_delta if use_risk_gate else None,
            require_baseline_stability=use_baseline_stability_gate,
        )
        day_metrics.insert(0, "fold", fold_index)
        day_metrics.insert(1, "val_start", str(start.date()))
        day_metrics.insert(2, "val_end", str(end.date()))
        all_day_metrics.append(day_metrics)

    detail = pd.concat(all_day_metrics, ignore_index=True)
    return detail, aggregate_by_fold(detail)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling validation for conservative near-baseline replacements.")
    parser.add_argument("--train-feature", required=True)
    parser.add_argument("--train-label", required=True)
    parser.add_argument("--nwp-dir", default="")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features_train.csv")
    parser.add_argument("--folds", default=DEFAULT_FOLDS)
    parser.add_argument("--seeds", default="42")
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
    parser.add_argument("--use-risk-gate", action="store_true")
    parser.add_argument(
        "--risk-objective",
        choices=["classification", "regression", "rule"],
        default="classification",
    )
    parser.add_argument("--risk-proba-threshold", type=float, default=0.60)
    parser.add_argument("--min-risk-expected-delta", type=float, default=0.0)
    parser.add_argument("--rule-shape-spike-max", type=float, default=None)
    parser.add_argument("--rule-shape-plateau-min", type=float, default=None)
    parser.add_argument("--rule-shape-balance-min", type=float, default=None)
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
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--params-json", default="")
    parser.add_argument("--output", default="outputs/replacement_classifier_rolling_day_metrics.csv")
    parser.add_argument("--aggregate-output", default="outputs/replacement_classifier_rolling_summary.csv")
    parser.add_argument("--scored-output-dir", default="")
    args = parser.parse_args()

    params = CLASSIFIER_PARAMS.copy()
    if args.params_json:
        params.update(json.loads(args.params_json))

    df = load_training_frame(
        args.train_feature,
        args.train_label,
        target_col=TARGET_COL,
        nwp_dir=args.nwp_dir,
        nwp_cache=args.nwp_cache,
    )
    detail, aggregate = run_rolling_validation(
        df=df,
        folds=parse_folds(args.folds),
        seeds=parse_seeds(args.seeds),
        params=params,
        max_shift=args.max_shift,
        daily_top_k=args.daily_top_k,
        topk_score_col=args.topk_score_col,
        positive_delta_threshold=args.positive_delta_threshold,
        proba_threshold=args.proba_threshold,
        min_expected_delta=args.min_expected_delta,
        min_margin=args.min_margin,
        max_proba_std=args.max_proba_std,
        use_risk_gate=args.use_risk_gate,
        risk_objective=args.risk_objective,
        risk_proba_threshold=args.risk_proba_threshold,
        min_risk_expected_delta=args.min_risk_expected_delta,
        rule_shape_spike_max=args.rule_shape_spike_max,
        rule_shape_plateau_min=args.rule_shape_plateau_min,
        rule_shape_balance_min=args.rule_shape_balance_min,
        use_baseline_stability_gate=args.use_baseline_stability_gate,
        baseline_stability_max_abs_delta=args.baseline_stability_max_abs_delta,
        charge_start_min=args.charge_start_min,
        charge_start_max=args.charge_start_max,
        discharge_start_min=args.discharge_start_min,
        discharge_start_max=args.discharge_start_max,
        baseline_window_score_col=args.baseline_window_score_col,
        baseline_mode=args.baseline_mode,
        safe_submission=args.safe_submission,
        source_submission=args.source_submission,
        min_group_count=args.min_group_count,
        source_num_boost_round=args.source_num_boost_round,
        source_early_stopping_rounds=args.source_early_stopping_rounds,
        source_val_days=args.source_val_days,
        source_train_baseline_mode=args.source_train_baseline_mode,
        source_threshold=args.source_threshold,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        scored_output_dir=args.scored_output_dir,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False)
    Path(args.aggregate_output).parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(args.aggregate_output, index=False)
    print(aggregate.to_string(index=False))
    print(f"saved_replacement_classifier_rolling_detail={args.output}")
    print(f"saved_replacement_classifier_rolling_summary={args.aggregate_output}")


if __name__ == "__main__":
    main()
