param(
  [string]$TrainFeature = "to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv",
  [string]$TrainLabel = "to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv",
  [string]$TestFeature = "to_sais_new/to_sais_new/test/test_in_feature_ori.csv",
  [string]$NwpDir = "to_sais_new/to_sais_new/all_nc",
  [string]$Output = "output.csv"
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

python -m src.predict `
  --test-feature $TestFeature `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --nwp-dir $NwpDir `
  --nwp-cache outputs/nwp_features_all.csv `
  --output outputs/test_predictions_nwp.csv

python -m src.make_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --metadata outputs/lgb_model_nwp_metadata.json `
  --threshold 0 `
  --charge-start-min 0 `
  --charge-start-max 55 `
  --discharge-start-min 72 `
  --discharge-start-max 88 `
  --output $Output `
  --meta-output outputs/nwp_c0_55_d72_88_meta.csv

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
  --discharge-start-min 72 `
  --discharge-start-max 88 `
  --output outputs/output_nwp_c0_55_d72_88.csv `
  --meta-output outputs/nwp_c0_55_d72_88_meta.csv

python -m src.check_submission --submission $Output
python -m src.check_submission --submission outputs/output_nwp_unconstrained.csv
python -m src.check_submission --submission outputs/output_nwp_c0_55_d72_88.csv
python -m src.compare_strategies --output outputs/strategy_compare.csv
