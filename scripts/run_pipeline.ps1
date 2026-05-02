param(
  [string]$TrainFeature = "to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv",
  [string]$TrainLabel = "to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv",
  [string]$TestFeature = "to_sais_new/to_sais_new/test/test_in_feature_ori.csv",
  [string]$NwpDir = "to_sais_new/to_sais_new/all_nc",
  [string]$Output = "output.csv",
  [string]$OnlineVerifiedOutput = "outputs/output_nwp_unconstrained_online5117.csv",
  [switch]$UseLocalBest
)

python -m src.train_lgb `
  --train-feature $TrainFeature `
  --train-label $TrainLabel `
  --model-output outputs/lgb_model.txt `
  --metadata-output outputs/lgb_model_metadata.json `
  --val-pred-output outputs/val_predictions.csv `
  --threshold-output outputs/best_threshold.txt `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --charge-start-min 51 `
  --charge-start-max 55 `
  --discharge-start-min 66 `
  --discharge-start-max 88

python -m src.predict `
  --test-feature $TestFeature `
  --metadata outputs/lgb_model_metadata.json `
  --output outputs/test_predictions.csv

python -m src.train_lgb `
  --train-feature $TrainFeature `
  --train-label $TrainLabel `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_train.csv `
  --model-output outputs/lgb_model_nwp.txt `
  --metadata-output outputs/lgb_model_nwp_metadata.json `
  --val-pred-output outputs/val_predictions_nwp.csv `
  --threshold-output outputs/best_threshold_nwp.txt `
  --feature-importance-output outputs/feature_importance_nwp.csv `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026

python -m src.train_lgb `
  --train-feature $TrainFeature `
  --train-label $TrainLabel `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_train.csv `
  --model-output outputs/lgb_model_nwp_bias.txt `
  --metadata-output outputs/lgb_model_nwp_bias_metadata.json `
  --val-pred-output outputs/val_predictions_nwp_bias.csv `
  --threshold-output outputs/best_threshold_nwp_bias.txt `
  --feature-importance-output outputs/feature_importance_nwp_bias.csv `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --use-forecast-bias

python -m src.train_lgb `
  --train-feature $TrainFeature `
  --train-label $TrainLabel `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_train.csv `
  --model-output outputs/lgb_model_nwp_exact_bias.txt `
  --metadata-output outputs/lgb_model_nwp_exact_bias_metadata.json `
  --val-pred-output outputs/val_predictions_nwp_exact_bias.csv `
  --threshold-output outputs/best_threshold_nwp_exact_bias.txt `
  --feature-importance-output outputs/feature_importance_nwp_exact_bias.csv `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --use-exact-calendar-history `
  --use-forecast-bias

python -m src.predict `
  --test-feature $TestFeature `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_all.csv `
  --output outputs/test_predictions_nwp.csv

python -m src.predict `
  --test-feature $TestFeature `
  --metadata outputs/lgb_model_nwp_bias_metadata.json `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_all.csv `
  --output outputs/test_predictions_nwp_bias.csv

python -m src.predict `
  --test-feature $TestFeature `
  --metadata outputs/lgb_model_nwp_exact_bias_metadata.json `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_all.csv `
  --output outputs/test_predictions_nwp_exact_bias.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions.csv `
  --metadata outputs/lgb_model_metadata.json `
  --threshold 0 `
  --output outputs/output_base_unconstrained.csv `
  --meta-output outputs/base_unconstrained_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 0 `
  --output outputs/output_nwp_unconstrained.csv `
  --meta-output outputs/nwp_unconstrained_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 0 `
  --charge-start-min 0 `
  --charge-start-max 55 `
  --output outputs/output_nwp_c0_55.csv `
  --meta-output outputs/nwp_c0_55_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 0 `
  --charge-start-min 0 `
  --charge-start-max 55 `
  --discharge-start-min 72 `
  --discharge-start-max 88 `
  --output outputs/output_nwp_c0_55_d72_88.csv `
  --meta-output outputs/nwp_c0_55_d72_88_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 500 `
  --output outputs/output_nwp_unconstrained_t500.csv `
  --meta-output outputs/nwp_unconstrained_t500_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 1000 `
  --output outputs/output_nwp_unconstrained_t1000.csv `
  --meta-output outputs/nwp_unconstrained_t1000_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 2000 `
  --output outputs/output_nwp_unconstrained_t2000.csv `
  --meta-output outputs/nwp_unconstrained_t2000_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp_bias.csv `
  --metadata outputs/lgb_model_nwp_bias_metadata.json `
  --threshold 0 `
  --output outputs/output_nwp_bias.csv `
  --meta-output outputs/nwp_bias_meta.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp_exact_bias.csv `
  --metadata outputs/lgb_model_nwp_exact_bias_metadata.json `
  --threshold 0 `
  --output outputs/output_nwp_exact_bias.csv `
  --meta-output outputs/nwp_exact_bias_meta.csv

python -m src.check_submission --submission outputs/output_nwp_unconstrained.csv
python -m src.check_submission --submission outputs/output_nwp_c0_55.csv
python -m src.check_submission --submission outputs/output_nwp_c0_55_d72_88.csv
python -m src.check_submission --submission outputs/output_nwp_bias.csv
python -m src.check_submission --submission outputs/output_nwp_exact_bias.csv
python -m src.compare_strategies --output outputs/strategy_compare.csv
if ((-not $UseLocalBest) -and (Test-Path $OnlineVerifiedOutput)) {
  Copy-Item -LiteralPath $OnlineVerifiedOutput -Destination $Output -Force
  "selected_submission=online_verified, source=$OnlineVerifiedOutput, output=$Output" | Out-File -FilePath outputs/selected_submission.txt -Encoding utf8
} else {
  python -m src.select_best_submission --strategy-compare outputs/strategy_compare.csv --output $Output --report-output outputs/selected_submission.csv
}
python -m src.check_submission --submission $Output
