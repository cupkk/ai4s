from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import TIME_COL, align_feature_frame, build_features, ensure_datetime
from .nwp_features import load_or_build_nwp_features, merge_nwp_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict test prices with a trained LightGBM model.")
    parser.add_argument("--test-feature", required=True, help="Test boundary condition CSV.")
    parser.add_argument("--model", default="outputs/lgb_model.txt")
    parser.add_argument("--metadata", default="outputs/lgb_model_metadata.json")
    parser.add_argument("--nwp-dir", default="", help="Optional official all_nc directory.")
    parser.add_argument("--nwp-cache", default="outputs/nwp_features.csv")
    parser.add_argument("--output", default="outputs/test_predictions.csv")
    args = parser.parse_args()

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed; run: pip install -r requirements.txt") from exc

    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    history_stats = metadata.get("history_stats")
    feature_options = metadata.get("feature_options", {})

    test_df = pd.read_csv(args.test_feature)
    test_df = ensure_datetime(test_df).sort_values(TIME_COL).reset_index(drop=True)
    if args.nwp_dir:
        nwp = load_or_build_nwp_features(
            args.nwp_dir,
            args.nwp_cache,
            start_time=str(test_df[TIME_COL].min()),
            end_time=str(test_df[TIME_COL].max()),
        )
        test_df = merge_nwp_features(test_df, nwp)
    feature_result = build_features(
        test_df,
        history_stats=history_stats,
        use_exact_calendar_history=bool(feature_options.get("use_exact_calendar_history", False)),
        use_forecast_bias=bool(feature_options.get("use_forecast_bias", False)),
    )
    x_test = align_feature_frame(feature_result.frame, feature_columns)

    model_paths = metadata.get("model_paths") or [args.model]
    preds = []
    for model_path in model_paths:
        model = lgb.Booster(model_file=model_path)
        preds.append(model.predict(x_test))
    pred = np.mean(np.vstack(preds), axis=0)
    out = pd.DataFrame({"times": test_df[TIME_COL].to_numpy(), "实时价格": pred})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"saved_predictions={args.output}, rows={len(out)}")


if __name__ == "__main__":
    main()
