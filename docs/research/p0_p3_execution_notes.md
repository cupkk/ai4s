# P0-P3 执行说明

更新日期：2026-04-30

## 核心结论

当前线上已验证主线仍然是：

```text
outputs/output_nwp_unconstrained.csv
```

对应线上分数：

```text
5117.832037755039
```

当前仓库里的 `output.csv` 与 `outputs/output_nwp_c0_55_d72_88.csv` 一致，是滚动验证更强的冲分候选；保底文件仍然是 `outputs/output_nwp_unconstrained.csv`。原因很简单：之前 `output_blend_fine_w025_t1000.csv` 本地看起来更好，但线上只有 `4703.505815153465`，说明单一验证窗口存在过拟合风险。

## P0：先防止提交混乱

### 做了什么

1. 修改 `scripts/run_pipeline.ps1`，去掉默认 `threshold=2000`。
2. 新增 `src/check_submission.py`，检查提交文件格式和充放电合法性。
3. 新增 `src/select_best_submission.py`，根据 `outputs/strategy_compare.csv` 自动选择最终 `output.csv`。
4. 新增 `src/nwp_diagnostics.py`，检查 NWP 维度、GHI 峰值时间、夜间 GHI 和缺失率。
5. 增强 `src/validate_profit.py`，新增：
   - `oracle_profit`：当天真实价格下理论最优收益。
   - `capture_ratio`：我们捕获了理论最优收益的比例。
   - `regret`：理论最优收益减去实际策略收益。
   - `window_hit`：充放电窗口是否命中理论最优窗口。
6. 新增 `docs/research/candidate_status.md`，明确哪些文件可提交、哪些只是候选。

### NWP 诊断结论

本地抽样诊断输出在：

```text
outputs/nwp_diagnostics.csv
```

当前官方样本文件的实际维度为：

```text
1 x 24 x 7 x 104 x 225
```

即 `time, hour, channel, lat, lon`。脚本已改成自动识别 `hour/channel` 轴，不再写死。抽样文件 GHI 峰值均在北京时间 `13:00`，夜间 GHI 均值为 `0`，暂未发现 8 小时错位证据。

### 为什么这样做

线上分数由充放电动作决定，不是由 RMSE 单独决定。只看平均收益容易漏掉两个问题：第一，策略可能错过真正高峰；第二，低分候选可能被脚本重新覆盖到 `output.csv`。P0 的目标是先把这个风险关掉。

## P1：建立稳定的本地评估

### 做了什么

1. 新增 `src/compare_strategies.py`，生成统一比较表：

```powershell
python -m src.compare_strategies --output outputs/strategy_compare.csv
```

2. 新增 `src/rolling_validate.py`，做扩展窗口滚动验证：

```powershell
python -m src.rolling_validate `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc `
  --output outputs/rolling_validation.csv `
  --aggregate-output outputs/rolling_validation_summary.csv
```

本轮已用单 seed、300 轮上限跑过一次滚动验证，输出：

```text
outputs/rolling_validation.csv
outputs/rolling_validation_summary.csv
```

结果摘要：`c0_55_d72_88` 在 4 个滚动窗口上平均收益最高，`avg_profit_mean=5783.5865`；无约束策略为 `5045.5199`。这说明窗口约束不只是 1-2 月验证期有效，但仍需要线上提交确认。

3. 新增 `src/diagnose_intertie_sign.py`，验证联络线正负号：

```powershell
python -m src.diagnose_intertie_sign `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv
```

4. 保守升级 `src/nwp_features.py`：
   - 增加 NWP 空间分位数 `p10/p50/p90`。
   - 增加 `nwp_wind_speed_cube_mean`。
   - 增加 `nwp_ghi_positive_ratio`。
   - 不改 NWP 时间对齐逻辑。

### 为什么这样做

之前的 2025 年 1-2 月验证存在一个重要问题：如果训练集用了 2025 年 3-12 月，再验证 1-2 月，就相当于用未来数据帮助验证。最终测试是 2026 年，所以更合理的本地验证应该是“用过去训练，预测未来”。滚动验证就是为了解决这个问题。

## P2：增加更贴近业务的模型

### 做了什么

1. 增强历史价格统计特征：
   - `hist_slot_std`
   - `hist_slot_p10`
   - `hist_slot_p90`
   - `hist_month_slot_std/p10/p90`
   - `hist_dow_slot_std/p10/p90`

2. 新增无泄漏历史价格特征模块 `src/price_history_features.py`：
   - `price_hist_same_month_day_slot`
   - `price_hist_month_slot_median/p10/p90`
   - `price_hist_winter_slot_*`
   - `price_hist_recent_7d/14d/28d_slot_*`

这些特征由训练期标签拟合，再映射到验证/测试期，避免直接使用验证期真实滞后价格。

3. 新增 residual LightGBM：

```powershell
python -m src.train_residual_lgb `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --test-feature to_sais_new/to_sais_new/test/test_in_feature_ori.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc
```

4. 新增 LightGBM ranker：

```powershell
python -m src.train_lgb_ranker `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --test-feature to_sais_new/to_sais_new/test/test_in_feature_ori.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc
```

### 为什么这样做

Residual 模型把“历史规律”和“边界条件导致的偏离”分开学，适合电价这种季节性、时段性很强的数据。Ranker 不再只追求逐点价格误差，而是直接学习一天内价格高低排序，更贴近“低价充电、高价放电”的评分目标。

## P3：高风险候选实验

### 做了什么

1. 新增交易分类模型：

```powershell
python -m src.train_trade_classifier `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --test-feature to_sais_new/to_sais_new/test/test_in_feature_ori.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc
```

2. 新增 Quantile LightGBM：

```powershell
python -m src.train_quantile_lgb `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --test-feature to_sais_new/to_sais_new/test/test_in_feature_ori.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc
```

3. 新增 seed 不确定性稳健策略：

```powershell
python -m src.make_robust_submission `
  --price-csv outputs/test_predictions_nwp.csv `
  --lambda-uncertainty 1.0 `
  --output outputs/output_nwp_robust_l1.csv
```

`predict.py` 现在会在多 seed 模型下输出 `pred_price_seed*` 和 `pred_std`。稳健策略用 `spread_mean - lambda * spread_uncertainty` 选择窗口，避免多个 seed 分歧很大时过度激进。

4. 新增窗口收益模型：

```powershell
python -m src.train_window_ranker `
  --train-feature to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv `
  --test-feature to_sais_new/to_sais_new/test/test_in_feature_ori.csv `
  --nwp-dir to_sais_new/to_sais_new/all_nc
```

它不再先预测 96 个点价格，而是直接枚举每天所有合法 `(charge_start, discharge_start)`，用真实窗口收益训练 LightGBM 回归模型。这个方向更贴近最终评分，但计算量更大，需要滚动验证后再考虑提交。

### 为什么这样做

交易分类模型直接学习“这个点是不是最优充电/放电窗口”，理论上更贴近得分，但也更容易受标签构造影响。Quantile 模型能给出不确定性，后续可以做稳健策略：如果 q90-q10 很宽，说明预测不稳，策略就应该降低激进度。

PatchTST、TFT、TimesNet 暂时不接入主线。原因是当前训练数据只有 2025 年一年，深度时序模型需要更多数据和更严格的滚动验证，否则很容易本地好看、线上变差。

本轮对 residual、ranker、trade classifier、quantile、window ranker 脚本都做了小轮数 smoke test，确认脚本能跑通；smoke 输出只用于检查程序，不作为正式提交结果。

## 下一步提交建议

1. 若想冲分，提交当前 `output.csv`，它等同于 `outputs/output_nwp_c0_55_d72_88.csv`，滚动验证平均收益最高。
2. 若想保底，提交 `outputs/output_nwp_unconstrained.csv`，这是已验证的 5117 分主线。
3. 先看 `outputs/strategy_compare.csv` 和 `outputs/rolling_validation_summary.csv`，只有跨窗口稳定提升的候选才值得占用线上提交次数。
4. 不要再提交 `threshold=2000`、`blend_fine_w025_t1000` 这类已经被线上证明低分的文件。
