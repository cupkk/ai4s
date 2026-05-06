# AI4S 研究进展日志 20260504

## 总体进展

当前线上可信最高分仍是 `5117.832037755039`，对应文件为 `outputs/output_nwp_unconstrained_online5117.csv`。根目录 `output.csv` 保持与该保底文件完全一致，SHA256 为：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

本轮目标不是继续降低阈值硬凑单日候选，而是实现“日级 top-K 替换候选筛选 + 更强校准特征”，让训练和验证更贴近真实提交策略：线上每次最多只替换 1 天，因此模型也应先在每天少量高质量近邻窗口中筛选，再判断是否值得替换保底窗口。

## 2026-05-04 更新：日级 top-K 替换候选筛选

### 做了什么

新增并接入以下能力：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

核心变化：

1. 新增 `add_replacement_calibration_features`，为近保底窗口生成更强的校准特征。
2. 新增 `filter_daily_topk_replacement_candidates`，支持每天只保留 top-K 个先验候选。
3. 新增 `prepare_replacement_candidates`，统一执行：

```text
near-baseline 过滤 -> 校准特征生成 -> 日级 top-K 预筛选
```

4. `rolling_validate_replacement_classifier.py` 和 `train_replacement_classifier.py` 均新增参数：

```text
--daily-top-k
--topk-score-col
```

### 校准特征说明

新增特征不使用 `true_delta_profit` 做筛选，避免标签泄漏。当前 top-K 先验主要来自以下相对保底窗口的非真实标签信号：

```text
delta_vs_baseline_spread_price_hist_recent_28d_slot_mean
delta_vs_baseline_spread_price_hist_month_slot_median
delta_vs_baseline_spread_hist_slot_mean
delta_vs_baseline_spread_hist_month_slot_mean
delta_vs_baseline_spread_price_hist_same_month_day_slot
```

同时加入移动幅度和方向校准特征：

```text
total_abs_shift
max_abs_shift
shift_balance_abs_diff
same_direction_shift
opposite_direction_shift
charge_shift_sign
discharge_shift_sign
candidate_gap_delta_abs
prior_rank_mean / prior_z_mean
daily_topk_prior_score
```

通俗解释：先用历史价差和窗口移动幅度筛掉明显不像好替换的窗口，再让分类器判断“这个替换是否真的有正收益概率”。这样比把每天几百个近邻窗口都喂给分类器更符合线上只改 1 天的策略。

### 验证命令

严格阈值验证：

```powershell
python -m src.rolling_validate_replacement_classifier `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --seeds 42 `
  --max-shift 8 `
  --daily-top-k 10 `
  --baseline-mode safe5117-source-model `
  --source-train-baseline-mode recent-oof `
  --source-val-days 59 `
  --source-threshold -1000000000000000000 `
  --proba-threshold 0.60 `
  --min-expected-delta 0 `
  --source-num-boost-round 500 `
  --source-early-stopping-rounds 50 `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --output outputs/replacement_classifier_topk10_safe5117source_ms8_p060_rolling_day_metrics.csv `
  --aggregate-output outputs/replacement_classifier_topk10_safe5117source_ms8_p060_rolling_summary.csv `
  --scored-output-dir outputs/replacement_classifier_topk10_safe5117source_ms8_p060_rolling_scored
```

结果：

```text
2025-04 proposed_days=0
2025-07 proposed_days=0
2025-10 proposed_days=0
2025-12 proposed_days=0
```

说明严格阈值下仍没有足够把握的替换候选。

宽松诊断验证：

```powershell
--daily-top-k 10
--proba-threshold 0.40
--min-expected-delta -100000
```

结果：

```text
2025-04 proposed_days=30, false_positive_rate=0.433333, total_delta_profit=20059.409832
2025-07 proposed_days=31, false_positive_rate=0.290323, total_delta_profit=24572.005315
2025-10 proposed_days=31, false_positive_rate=0.516129, total_delta_profit=-5487.903253
2025-12 proposed_days=31, false_positive_rate=0.580645, total_delta_profit=-2508.036149
```

再测 `daily_top_k=5` 后，10 月和 12 月仍为负，说明仅缩小候选池还不能解决误报问题。

### 阈值网格诊断

已输出：

```text
outputs/replacement_classifier_topk10_safe5117source_threshold_grid.csv
outputs/replacement_classifier_topk5_safe5117source_threshold_grid.csv
```

关键结论：

- `topK=10, p=0.45, min_expected_delta=-100000`：4 个 fold 都触发，但 10 月 fold 仍亏损，最大 fold 误报率高。
- `topK=10, p=0.50/0.55`：总收益为正，但只在 2 个 fold 中触发，泛化不稳。
- `topK=5`：没有明显改善，10 月 fold 仍是主要风险。
- 加 `min_margin > 0` 后多数设置直接没有候选，少数候选反而全误报。

### 决策

本轮不生成 `blocked=False` manifest，不生成新的线上提交候选。

原因：top-K 和校准特征是正确方向，但 rolling validation 尚未达到放行标准。宽松阈值下仍有 10 月、12 月负收益和较高误报率。如果现在硬生成单日候选，本质仍是在赌 hidden test，而不是稳健冲分。

### 当前保留产物

保留以下产物用于后续分析：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p060_rolling_summary.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_diag_rolling_summary.csv
outputs/replacement_classifier_topk5_safe5117source_ms8_p040_diag_rolling_summary.csv
outputs/replacement_classifier_topk10_safe5117source_threshold_grid.csv
outputs/replacement_classifier_topk5_safe5117source_threshold_grid.csv
```

### 下一步

1. 不继续降阈值。
2. 从 10 月、12 月误报样本反查特征，找出模型为什么把亏损替换判成正向。
3. 增加“月份/季节风险校准”和“候选相对日内 top1 的保守距离”特征。
4. 考虑训练二阶段校准器：

```text
第一阶段：日级 top-K 候选生成
第二阶段：只对每日 top1/top2 候选做 fold-aware 风险分类
```

5. 只有当 rolling validation 出现少量候选且满足以下条件，才允许重新生成 `blocked=False` manifest：

```text
proposed_days > 0
false_positive_rate 明显低，最好所有触发 fold <= 0.25
total_delta_profit > 0
worst_fold_profit >= 0 或亏损很小
```

## 验证状态

代码编译通过：

```powershell
python -m compileall src
```

当前保底提交文件仍合法：

```powershell
python -m src.check_submission --submission output.csv
```

结果：

```text
rows=5664
days=59
traded_days=59
errors=0
warnings=0
```

## 2026-05-04 更新：分析 2025-10/12 误报并加入二阶段风险 gate

### 背景

用户要求直接分析：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_diag_rolling_scored
```

中的 2025-10 和 2025-12 误报样本，并做二阶段风险校准；明确不要继续降阈值硬出单日候选。

### 误报分析

先从 `p040 diag` 的 stage1 候选中抽取每天实际被选中的窗口，输出：

```text
outputs/replacement_classifier_topk10_stage1_selected_rows.csv
```

共 123 个 stage1 日级候选：

```text
positive=67
false_positive=56
```

2025-10 误报特点：

- 大亏误报多来自窗口大幅移动，尤其放电窗口提前或后移 8 个 15 分钟格。
- 单看历史价差提升不可靠，例如部分 `delta_vs_baseline_spread_price_hist_recent_28d_slot_mean` 很高，但真实 delta 仍大亏。
- 更有效的信号是候选相对保底窗口的供需/净负荷改善。

2025-12 误报特点：

- 单日亏损幅度小于 10 月，但正负例的先验分数非常接近。
- 原一阶段概率和 expected_delta 基本无法区分正负例。
- 分类式二阶段 LightGBM 在小样本下退化成 fold 内常数，只能拦住整月，不能做日内区分。

### 尝试过的二阶段模型

已实现二阶段接口：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

新增列：

```text
risk_positive_proba
risk_expected_delta
rule_risk_score
```

新增能力：

```text
select_stage1_candidate_rows
risk_feature_columns
add_risk_predictions_to_scored_windows
add_risk_regression_predictions_to_scored_windows
fit_rule_risk_gate
add_rule_risk_predictions_to_scored_windows
```

二阶段分类 gate 结果：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_risk060_rolling_summary.csv

2025-04 proposed_days=30, false_positive_rate=0.433333, total_delta_profit=20059.409832
2025-07 proposed_days=31, false_positive_rate=0.290323, total_delta_profit=24572.005315
2025-10 proposed_days=0
2025-12 proposed_days=0
```

解释：分类 gate 能拦住 10/12 月，但 4 月仍全量放行，误报率仍高；原因是 stage1 日级样本很少，LightGBM 二阶段分类容易输出 fold 内常数。

### 规则型二阶段风险 gate

用已有 scored 文件做无泄漏 train-quantile 回放：每个验证 fold 的 gate 阈值只从其它 fold 的 stage1 候选中估计，再应用到当前 fold。

输出：

```text
outputs/replacement_classifier_topk10_stage1_train_quantile_risk_gates.csv
outputs/replacement_classifier_topk10_rule_gate_replay_summary.csv
```

当前最稳规则：

```text
delta_vs_baseline_spread_net_load >= 训练期 stage1 的 0.85 分位
且
delta_vs_baseline_spread_hist_slot_mean_daily_centered >= 训练期 stage1 的 0.80 分位
```

回放结果：

```text
fold=1 proposed_days=13, false_positive_days=1, false_positive_rate=0.076923, total_delta_profit=45353.953321
fold=2 proposed_days=2, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=4540.805471
fold=3 proposed_days=1, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=14291.491802
fold=4 proposed_days=0, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=0.000000

aggregate proposed_days=16
aggregate false_positive_days=1
aggregate total_delta_profit=64186.250594
```

通俗解释：这个规则只允许“供需改善非常强，并且历史同 slot 改善也不弱”的替换候选通过。它会明显减少 10/12 月误报，但也会放弃很多中等强度候选，属于保守 gate。

### 正式 rolling 状态

尝试跑正式 rolling：

```powershell
python -m src.rolling_validate_replacement_classifier `
  --daily-top-k 10 `
  --baseline-mode safe5117-source-model `
  --proba-threshold 0.40 `
  --min-expected-delta -100000 `
  --use-risk-gate `
  --risk-objective rule `
  --risk-proba-threshold 1.0 `
  --min-risk-expected-delta 0
```

结果：命令超时，未生成完整 summary。此前还遇到一次 NWP 缓存覆盖不足导致重新读取 nc 文件并触发内存不足：

```text
numpy.core._exceptions._ArrayMemoryError
```

原因：`outputs/nwp_features_train.csv` 从 `2025-01-02` 开始，训练数据需要覆盖 `2025-01-01`，loader 判定缓存不覆盖起点后尝试重建 NWP 特征。

### 决策

当前不生成 `blocked=False` manifest，也不生成线上提交候选。

原因：

1. 规则型二阶段 gate 的 scored replay 信号很好，但正式 rolling 因资源超时未完成。
2. 线上提交必须遵守更严格证据链，不能把 replay 直接当作放行许可。
3. 下一步应先修复/补全 NWP 缓存覆盖，或清理残留 Python 训练进程后再跑正式 rolling。

### 当前可保留的代码能力

二阶段 gate 已接入：

```text
src/rolling_validate_replacement_classifier.py:
  --use-risk-gate
  --risk-objective classification|regression|rule

src/train_replacement_classifier.py:
  --use-rule-risk-gate
  --risk-proba-threshold
  --min-risk-expected-delta
```

后续如果正式 rolling 通过，可以用最终训练脚本生成测试期 scored windows，但仍需先过：

```powershell
python -m src.check_submission --submission <candidate>
python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate <candidate>
python -m src.guard_submission_candidate --candidate <candidate> --manifest <manifest>
```

### 下一步

1. 补全 `outputs/nwp_features_train.csv` 对 `2025-01-01` 的覆盖，或让 loader 在缓存覆盖主体训练期时不重建全部 nc。
2. 清理确认无效的残留 Python 进程，避免内存不足。
3. 重新跑正式 `risk-objective rule` rolling。
4. 只有正式 rolling 的 `false_positive_rate` 和 `worst_fold_profit` 达标，才允许训练测试期 scored windows。
5. 即使测试期 scored windows 出现候选，也先生成 `blocked=True` manifest，人工复核后再决定是否放行。

## 2026-05-04 更新：NWP 缓存补齐并完成正式 rule rolling

### 做了什么

1. 先核验了当前保底锚点，没有动 `output.csv`：
```text
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

2. 补齐了 `outputs/nwp_features_train.csv` 对 `2025-01-01` 的覆盖，避免 loader 因起点缺失去重建 `.nc` 文件：
   - 原始缓存备份：`outputs/nwp_features_train_original_from_20250102.csv`
   - 新缓存：`outputs/nwp_features_train_with_20250101.csv`
   - 默认缓存：`outputs/nwp_features_train.csv`
   - 补法：把最早可用的 `2025-01-02 00:00:00` 行按 15 分钟粒度回填到 `2025-01-01 00:00:00` 到 `23:45:00`，等价于开头 bfill，只用于防止缓存重建，不再读 `.nc`。

3. 重跑了正式 rolling，使用缓存路径：
```powershell
python -m src.rolling_validate_replacement_classifier `
  --nwp-cache outputs\nwp_features_train.csv `
  --use-risk-gate `
  --risk-objective rule
```

### 结果

正式 rolling 已跑通，不再触发 `.nc` 重建或内存错误。汇总结果如下：

```text
proposed_days_sum=8
positive_selected_days_sum=4
false_positive_days_sum=4
total_delta_profit_sum=19540.0802914541
worst_selected_delta_min=-5768.976570827298
false_positive_rate_max=0.75
false_positive_rate_mean=0.3125
```

### 决策

这次正式 rolling 仍然不达标，所以继续保持：

```text
submit_allowed=false
```

原因很直接：
1. 规则 gate 仍有明显误报，`false_positive_days` 不是 0。
2. `worst_selected_delta` 仍为负，说明少数替换会亏。
3. 在 formal rolling 没过线之前，不生成任何新候选，不放行 `blocked=False` manifest。

### 保留的产物

```text
outputs/nwp_features_train_original_from_20250102.csv
outputs/nwp_features_train_with_20250101.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_day_metrics_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_scored_cachefix_20260504/
```

### 下一步

只继续做一件事：分析这次 formal rolling 里仍然冒出来的误报 fold，继续加二阶段风险校准特征；在 rolling 达标前，不提交任何新候选。

## 2026-05-04 更新：结构性二阶段 gate 正式 rolling 复跑通过

### 做了什么

在前一版规则 gate 的基础上，继续拆 2025-10 / 2025-12 误报样本后，发现主要误报不是单纯阈值问题，而是结构形态问题：

1. 有些候选只移动了充电或放电其中一个窗口，真实收益不稳定。
2. 有些候选把充电和放电窗口同时提前，历史价差信号看起来好，但在 10 月、12 月容易误报。

因此在 `src/replacement_classifier.py` 的规则型风险 gate 中加入结构性约束：

```text
require_both_windows_moved = true
block_both_windows_earlier = true
risk_rule_structural_pass
```

通俗解释：以后不再放行“只挪一边窗口”的替换，也不放行“充电和放电一起往前挪”的替换。这样做不是继续降阈值硬筛，而是把历史误报里最明显的坏形态拦掉。

### 验证命令

已重新运行正式 rolling：

```powershell
python -m src.rolling_validate_replacement_classifier `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --nwp-cache outputs\nwp_features_train.csv `
  --seeds 42 `
  --max-shift 8 `
  --daily-top-k 10 `
  --baseline-mode safe5117-source-model `
  --source-train-baseline-mode recent-oof `
  --source-val-days 59 `
  --source-threshold -1000000000000000000 `
  --proba-threshold 0.40 `
  --min-expected-delta -100000 `
  --use-risk-gate `
  --risk-objective rule `
  --risk-proba-threshold 1.0 `
  --min-risk-expected-delta 0 `
  --source-num-boost-round 500 `
  --source-early-stopping-rounds 50 `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --output outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_day_metrics_structural_20260504.csv `
  --aggregate-output outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_structural_20260504.csv `
  --scored-output-dir outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_scored_structural_20260504
```

### 结果

正式 rolling 与此前基于 scored folds 的无泄漏 replay 完全一致，关键列逐项相等：

```text
fold=1 2025-04 proposed_days=1 false_positive_days=0 total_delta_profit=10535.642825 worst_selected_delta=10535.642825
fold=2 2025-07 proposed_days=1 false_positive_days=0 total_delta_profit=2628.647939  worst_selected_delta=2628.647939
fold=3 2025-10 proposed_days=1 false_positive_days=0 total_delta_profit=14291.491802 worst_selected_delta=14291.491802
fold=4 2025-12 proposed_days=1 false_positive_days=0 total_delta_profit=210.676144   worst_selected_delta=210.676144

aggregate proposed_days=4
aggregate positive_selected_days=4
aggregate false_positive_days=0
aggregate total_delta_profit=27666.458709
aggregate worst_selected_delta=210.676144
```

本轮还复核了：

```powershell
python -m compileall src
python -m src.check_submission --submission output.csv
```

结果：

```text
compileall passed
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

### 决策

正式 rolling 现在达标，但本轮仍不直接生成线上候选、不覆盖 `output.csv`。

原因：

1. 用户当前明确要求先拆正式 rolling 的误报 fold 并做二阶段校准；这一步已经完成。
2. 下一步应单独进入“测试期 scored windows 生成 + 单日 manifest + guard 复核”阶段，不能把验证通过和候选提交混成一步。
3. 即使下一步出现 `blocked=False` 候选，也必须先和 `outputs/output_nwp_unconstrained_online5117.csv` 做逐日 action diff，再跑 `guard_submission_candidate.py`。

### 保留的产物

```text
outputs/replacement_classifier_rulegate_structural_replay_summary_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_day_metrics_structural_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_structural_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_scored_structural_20260504/
```

### 下一步

下一步可以开始测试期候选生成，但必须保持小步：

1. 用同一套正式达标参数训练全量 replacement classifier，并输出测试期 scored windows。
2. 先检查测试期是否有 `pred_rank=1` 且通过 `risk_rule_structural_pass` 的日期。
3. 若有候选，最多生成 1 到 3 个单日替换 manifest，默认先人工复核。
4. 每个候选必须运行：

```powershell
python -m src.check_submission --submission <candidate>
python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate <candidate>
python -m src.guard_submission_candidate --candidate <candidate> --manifest <manifest>
```

5. 根目录 `output.csv` 继续保持 5117 保底，不由训练或候选脚本自动覆盖。

## 2026-05-04 更新：测试期 scored windows 生成，未产生可提交单日候选

### 做了什么

在结构性二阶段 gate 正式 rolling 达标后，用同一套主参数训练全量 replacement classifier，并输出测试期 scored windows：

```powershell
python -m src.train_replacement_classifier `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --test-feature to_sais_new\to_sais_new\test\test_in_feature_ori.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --nwp-cache outputs\nwp_features_train.csv `
  --test-nwp-cache outputs\nwp_features_all.csv `
  --seeds 42 `
  --max-shift 8 `
  --daily-top-k 10 `
  --baseline-mode safe5117-source-model `
  --source-train-baseline-mode recent-oof `
  --source-val-days 59 `
  --source-threshold -1000000000000000000 `
  --proba-threshold 0.40 `
  --min-expected-delta -100000 `
  --use-rule-risk-gate `
  --risk-proba-threshold 1.0 `
  --min-risk-expected-delta 0 `
  --source-num-boost-round 500 `
  --source-early-stopping-rounds 50 `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --test-baseline-submission outputs\output_nwp_unconstrained_online5117.csv `
  --model-output outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504_model.txt `
  --metadata-output outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504_metadata.json `
  --val-window-output outputs\val_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504.csv `
  --val-day-metrics-output outputs\val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504_day_metrics.csv `
  --test-window-output outputs\test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504.csv
```

### 结果

测试期 scored windows 已生成：

```text
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504.csv
rows=590
unique_days=59
pred_rank=1 days=0
```

也就是说，在当前正式 gate 下，测试期没有任何一天被放行为单日替换候选。因此没有运行 `make_single_day_candidates --allow-submission`，也没有生成新的 `blocked=False` manifest。

同时，全量训练脚本自带的尾部验证暴露了一个新风险：

```text
validation_summary:
proposed_days=8
positive_selected_days=3
false_positive_days=5
false_positive_rate=0.625
total_delta_profit=-574.604850
worst_selected_delta=-836.087342
```

这些误报主要集中在 2025-11 / 2025-12。更重要的是，`safe5117-source-model` 在 full train / tail validation 下的 baseline window 会和 formal rolling fold 中同一天的 baseline window 发生漂移。例如：

```text
formal fold 2025-12-05:
baseline=charge 36, discharge 69
candidate=charge 40, discharge 71
true_delta_profit=+210.676144

full tail validation 2025-12-05:
baseline=charge 37, discharge 71
candidate=charge 45, discharge 73
true_delta_profit=-836.087342
```

通俗解释：同一个日期在不同训练切分下，源模型生成的“同源保底窗口”会变，所以结构 gate 不能只看窗口形态，还要防 baseline 锚点漂移。

### 额外风险网格

对 `min-risk-expected-delta` 做了小网格回放，发现：

```text
min-risk-expected-delta=0.10

formal rolling:
proposed_days=1
false_positive_days=0
total_delta_profit=14291.491802
worst_selected_delta=14291.491802

full tail validation:
proposed_days=1
false_positive_days=0
total_delta_profit=535.836774
worst_selected_delta=535.836774

test:
proposed_days=0
```

这说明更严格的风险分数阈值能压掉 11/12 月误报，但当前测试期仍无可提交候选。

### 决策

保持：

```text
submit_allowed=false
output.csv=5117 baseline
```

原因：

1. 测试期当前没有任何 `pred_rank=1` 的可放行日期。
2. 全量训练尾部验证暴露了 baseline 漂移和 11/12 月误报风险。
3. 不能为了冲分去降阈值硬造单日候选；这会回到前面 `skip_t500` 的错误模式。

### 下一步

下一步不是提交，而是修正训练/候选生成闭环：

1. 把 `min-risk-expected-delta=0.10` 作为下一轮正式 rolling 的候选守门参数重跑。
2. 专门分析 `safe5117-source-model` baseline 在 full train 与 rolling fold 之间的窗口漂移。
3. 若要出候选，优先要求测试期候选同时满足：
   - `risk_expected_delta >= 0.10`
   - `risk_rule_structural_pass = 1`
   - `changed_days = 1`
   - `guard_submission_candidate.py` 输出 `decision=PASS`
4. 若仍为 0 候选，就继续保留 5117，不提交新文件。
