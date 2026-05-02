# 优化实验记录 2026-05-02

本轮目标不是盲目堆模型，而是把几个高优先级方向逐一验证：滚动验证、窗口细搜、不确定性惩罚、residual 模型、直接窗口收益模型。

## 1. 滚动验证

运行命令：

```powershell
python -m src.rolling_validate `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --nwp-cache outputs\nwp_features_train.csv `
  --seeds 42 `
  --num-boost-round 700 `
  --early-stopping-rounds 60 `
  --output outputs\rolling_validation_current.csv `
  --aggregate-output outputs\rolling_validation_current_summary.csv `
  --pred-output-dir outputs\rolling_predictions_current
```

输出文件：

```text
outputs/rolling_validation_current.csv
outputs/rolling_validation_current_summary.csv
```

关键结果：

| strategy | folds | avg_profit_mean | capture_ratio_mean | loss_days_mean |
|---|---:|---:|---:|---:|
| `c0_55_d72_88` | 4 | `5048.6000` | `0.5125` | `6.00` |
| `c0_55` | 4 | `4183.7238` | `0.3576` | `8.75` |
| `unconstrained` | 4 | `4158.3681` | `0.3528` | `8.50` |
| `threshold_1000` | 4 | `1589.7775` | `0.1347` | `3.75` |

结论：`c0_55_d72_88` 不只是 2025 年 1-2 月验证窗口好看，在 4、7、10、12 月滚动切片中也保持领先。因此当前窗口约束仍然是主线。

## 2. 窗口细网格搜索

新增脚本：

```text
src/tune_strategy_windows.py
```

运行命令：

```powershell
python -m src.tune_strategy_windows `
  --pred-csv outputs\val_predictions_nwp.csv `
  --test-price-csv outputs\test_predictions_nwp.csv `
  --output outputs\window_constraint_search.csv `
  --candidate-manifest outputs\window_constraint_candidates.csv `
  --submission-prefix output_nwp_window `
  --top-k 5
```

关键结果：

| candidate | avg_profit | capture_ratio | loss_days |
|---|---:|---:|---:|
| `c0_55_d72_88_t100` | `13370.6122` | `0.8432` | `0` |
| `c0_55_d72_88_t0` | `13370.5786` | `0.8432` | `1` |
| `c0_55_d68_88_t0` | `13340.3553` | `0.8266` | `0` |

结论：原窗口几乎就是最优。`threshold=100` 线下略稳，但测试集上与当前 `output.csv` 完全相同，因为测试期每天预测价差都超过 100。

## 3. 不确定性惩罚

新增脚本：

```text
src/tune_robust_strategy.py
```

运行命令：

```powershell
python -m src.tune_robust_strategy `
  --pred-csv outputs\val_predictions_nwp.csv `
  --test-price-csv outputs\test_predictions_nwp.csv `
  --output outputs\robust_strategy_search.csv `
  --candidate-manifest outputs\robust_strategy_candidates.csv `
  --submission-prefix output_nwp_robust `
  --top-k 5
```

关键结果：

| lambda | threshold | avg_profit | capture_ratio | loss_days |
|---:|---:|---:|---:|---:|
| `0.00` | `100` | `13370.6122` | `0.8432` | `0` |
| `0.00` | `0` | `13370.5786` | `0.8432` | `1` |
| `0.25` | `0` | `13360.5001` | `0.8426` | `0` |
| `0.50` | `0` | `12345.0212` | `0.7785` | `0` |

结论：seed 分歧惩罚没有明显增益。当前多 seed 的均值预测已经足够，强行惩罚会减少可盈利交易。

## 4. Residual LightGBM

运行命令：

```powershell
python -m src.train_residual_lgb `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --test-feature to_sais_new\to_sais_new\test\test_in_feature_ori.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --charge-start-min 0 `
  --charge-start-max 55 `
  --discharge-start-min 72 `
  --discharge-start-max 88 `
  --threshold-grid 0,100,200,500,1000
```

关键结果：

```text
residual_validation_rmse=1.235403
best avg_profit=9906.3860
best threshold=100
```

结论：residual 单模型弱于主线，不作为提交文件。它后续可以尝试做 stacking 的辅助输入，但不能单独替换当前模型。

## 5. 窗口收益模型

运行命令：

```powershell
python -m src.train_window_ranker `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --test-feature to_sais_new\to_sais_new\test\test_in_feature_ori.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --charge-start-min 0 `
  --charge-start-max 55 `
  --discharge-start-min 72 `
  --discharge-start-max 88
```

关键结果：

```text
window_ranker_validation=avg_profit=7712.807117
capture_ratio=0.486411
```

结论：直接窗口收益模型当前没有跑赢主线。早停发生在第 1-5 轮，说明候选窗口特征对窗口收益的解释能力不足，暂时不适合作为提交策略。

## 当前提交建议

当前 `output.csv` 仍然是冲分候选，文件哈希与 `outputs/output_nwp_c0_55_d72_88.csv` 一致。

```text
3EF3D82B60E05C9CCD5533558BC64D392FFAF10F150F7752B01EA0BD09E78E47
```

如果线上分数低于 `5117.8320`，保底文件是：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

## 下一步

1. 线上提交当前 `output.csv`，记录分数。
2. 如果低于保底分，立即提交 `outputs/output_nwp_unconstrained_online5117.csv`。
3. 后续冲榜不要再优先做复杂模型，优先做：
   - 对 `output.csv` 与保底文件逐日差异分析；
   - 针对 2026 年 1-2 月测试期做更稳的季节迁移验证；
   - 尝试用主线预测和保底预测做“按日选择”，而不是全局 blend。
