# AI4S 蒙西电力现货储能策略

本项目用于完成赛题初赛提交：根据边界条件、NWP 气象数据和历史电价规律预测电价，再生成储能充放电策略 `output.csv`。

## 当前版本能力

1. LightGBM 多 seed 集成预测。
2. 支持官方 `.nc` NWP 气象特征，并带有维度自适应读取。
3. 支持 NWP 诊断，检查 GHI 峰值时间、夜间 GHI、缺失率和数据维度。
4. 支持历史价格统计特征，包括同月同日同 slot、month-slot 分位数、冬季 slot、最近 7/14/28 天 slot 统计。
5. 支持 `oracle_profit`、`capture_ratio`、`regret`、`window_hit` 等收益诊断。
6. 支持多策略统一比较、提交格式检查和充放电合法性检查。
7. 支持窗口细网格搜索、不确定性惩罚调参、residual LightGBM 和窗口收益模型实验。

## 数据目录

默认使用官方数据路径：

```text
to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv
to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv
to_sais_new/to_sais_new/test/test_in_feature_ori.csv
to_sais_new/to_sais_new/all_nc/*.nc
```

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 一键训练与生成

```powershell
.\scripts\run_pipeline.ps1
```

脚本会依次完成：

1. 训练基础 LightGBM、NWP LightGBM、NWP bias、NWP exact bias 候选。
2. 预测测试集。
3. 生成多个候选提交文件。
4. 运行提交合法性检查。
5. 生成 `outputs/strategy_compare.csv`。
6. 默认把线上验证过的保底文件 `outputs/output_nwp_unconstrained_online5117.csv` 复制为 `output.csv`。

默认不再用本地验证最高候选覆盖 `output.csv`，因为 `nwp_c0_55_d72_88` 已在线上验证为低分 `3798.629342284567`。如果明确要用本地验证最优候选覆盖 `output.csv`，需要显式传入：

```powershell
.\scripts\run_pipeline.ps1 -UseLocalBest
```

## 推荐提交文件

当前最终提交入口：

```text
output.csv
```

当前 `output.csv` 已恢复为线上验证过的保底文件：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

已知线上分数：

```text
5117.832037755039
```

不要再提交以下已验证低分或本地验证失败的候选：

```text
outputs/output_nwp_c0_55_d72_88.csv
outputs/output_blend_fine_w025_t1000.csv
outputs/output_nwp_unconstrained_t2000.csv
outputs/output_residual_nwp.csv
outputs/output_window_ranker_c055_d7288.csv
```

## 常用诊断命令

检查 NWP 维度和时间对齐：

```powershell
python -m src.nwp_diagnostics `
  --nwp-dir to_sais_new/to_sais_new/all_nc `
  --output outputs/nwp_diagnostics.csv
```

检查提交合法性：

```powershell
python -m src.check_submission --submission output.csv
```

比较策略：

```powershell
python -m src.compare_strategies --output outputs/strategy_compare.csv
```

滚动验证：

```powershell
python -m src.rolling_validate `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc `
  --output outputs/rolling_validation.csv `
  --aggregate-output outputs/rolling_validation_summary.csv
```

窗口约束搜索：

```powershell
python -m src.tune_strategy_windows `
  --pred-csv outputs/val_predictions_nwp.csv `
  --test-price-csv outputs/test_predictions_nwp.csv `
  --output outputs/window_constraint_search.csv
```

不确定性惩罚搜索：

```powershell
python -m src.tune_robust_strategy `
  --pred-csv outputs/val_predictions_nwp.csv `
  --test-price-csv outputs/test_predictions_nwp.csv `
  --output outputs/robust_strategy_search.csv
```

## 关键文件

| 文件 | 作用 |
|---|---|
| `src/train_lgb.py` | 主线 LightGBM 训练，含 NWP 和历史价格统计特征。 |
| `src/predict.py` | 根据模型和元数据预测测试集。 |
| `src/nwp_features.py` | 读取和缓存 NWP 特征。 |
| `src/nwp_diagnostics.py` | 检查 NWP 维度和时间对齐。 |
| `src/price_history_features.py` | 构造无泄漏历史价格统计特征。 |
| `src/storage_optimizer.py` | 枚举每天一次充放电的最优动作。 |
| `src/validate_profit.py` | 回测收益、oracle、regret 和窗口命中率。 |
| `src/compare_strategies.py` | 汇总比较多个候选策略。 |
| `src/select_best_submission.py` | 从比较表自动选择本地最优候选。 |
| `src/check_submission.py` | 检查提交文件格式和策略合法性。 |
| `src/tune_strategy_windows.py` | 在验证集上搜索充放电窗口边界和低阈值。 |
| `src/tune_robust_strategy.py` | 用多 seed 分歧调参稳健策略。 |
| `src/train_residual_lgb.py` | 训练历史价格先验上的 residual LightGBM。 |
| `src/train_window_ranker.py` | 直接训练窗口收益模型。 |

## 当前注意事项

1. `output.csv` 是最终提交文件。默认 pipeline 会恢复线上验证过的保底版本。
2. 只有使用 `-UseLocalBest` 才会用本地验证最高候选覆盖 `output.csv`。
3. `outputs/` 下大部分实验结果只用于本地复盘。
4. 每次线上提交结果都要写入 `docs/research/`，并更新候选状态表。

## Open Source Maintenance

This repository is maintained as an open-source AI-for-science energy storage strategy pipeline. See:

- [Open Source Maintenance](docs/OPEN_SOURCE_MAINTENANCE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [MIT License](LICENSE)
