# AI4S 研究进展日志 20260506

## 总体研究进展

当前线上可信最高分仍是 `5117.832037755039`，对应保底文件：

```text
outputs/output_nwp_unconstrained_online5117.csv
output.csv
```

根目录 `output.csv` 当前仍保持 5117 保底，SHA256：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

当前研究主线已经从“跳过低收益日”切换到“相对保底窗口的单日替换分类器”。目标不是生成整套新策略，而是在 5117 保底文件基础上，只找经过 rolling validation、风险校准、action diff、guard 全部放行的 1 天替换候选。

截至本日志，本轮结论为：

```text
submit_allowed=false
recommended_submission=output.csv
new_candidate_count=0
```

原因是更严格的 `min-risk-expected-delta=0.10` 下，测试期仍没有任何可放行单日替换候选；同时 `safe5117-source-model` 的历史 baseline 锚点在 rolling fold 与 full-tail validation 之间存在明显漂移。

## 2026-05-06 更新：接续严格 min-risk=0.10 复核

### 做了什么

首先确认了根目录保底提交文件未被覆盖：

```powershell
Get-FileHash output.csv -Algorithm SHA256
python -m src.check_submission --submission output.csv
```

结果：

```text
SHA256=AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

随后复核了上一轮严格模型产物：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_summary_20260505.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_day_metrics_20260505.csv
outputs/val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505_day_metrics.csv
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505.csv
```

严格参数为：

```text
baseline_mode=safe5117-source-model
daily_top_k=10
max_shift=8
proba_threshold=0.40
risk_objective=rule
min-risk-expected-delta=0.10
```

正式 rolling 汇总：

```text
fold=2025-10 proposed_days=1 false_positive_days=0 total_delta_profit=14291.49180180204
其他 fold proposed_days=0
overall false_positive_days=0
```

全量训练尾部验证：

```text
proposed_days=1
date=2025-12-24
selected_true_delta_profit=535.8367742571918
false_positive_days=0
```

测试期打分：

```text
rows=590
days=59
pred_rank=1 rows=0
pred_rank=1 days=0
structural_pass_rows=294
structural_pass_days=48
risk_expected_delta>=0 rows=7
risk_expected_delta>=0 days=2
risk_expected_delta>=0.10 rows=0
risk_expected_delta>=0.10 days=0
max_risk_expected_delta=0.0350848884796989
```

通俗解释：测试期确实有少量窗口在结构规则上看起来接近可用，主要集中在 `2026-02-21` 和 `2026-02-24`，但最高风险期望分只有 `0.035085`，低于正式守门线 `0.10`。这不是“差一点就可以提交”，而是风险校准明确没有放行。

### 新增 baseline 漂移诊断脚本

新增文件：

```text
src/analyze_baseline_drift.py
```

用途：比较 `safe5117-source-model` 在不同训练切分下生成的 baseline 窗口是否一致，并把测试期候选 gate 状态一并汇总。该脚本只读已有输出，不训练模型，不生成 submission，不覆盖 `output.csv`。

运行命令：

```powershell
python -m src.analyze_baseline_drift `
  --rolling-meta outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_scored_20260505\fold_04_safe5117_source_val_meta.csv `
  --full-meta outputs\val_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505_source_val_meta.csv `
  --rolling-label fold04_dec `
  --full-label full_tail `
  --rolling-day-metrics outputs\replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_day_metrics_20260505.csv `
  --full-day-metrics outputs\val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505_day_metrics.csv `
  --test-windows outputs\test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505.csv `
  --detail-output outputs\safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_20260506.csv `
  --summary-output outputs\safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_summary_20260506.csv
```

新增输出：

```text
outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_20260506.csv
outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_summary_20260506.csv
```

### 漂移诊断结果

12 月 formal rolling fold 与 full-tail validation 的 baseline 窗口比较：

```text
overlap_days=31
changed_days=28
same_days=3
changed_rate=0.9032258064516129
mean_abs_charge_delta=2.6129032258064515
mean_abs_discharge_delta=5.354838709677419
max_abs_charge_delta=38
max_abs_discharge_delta=41
large_drift_days_ge_8=8
```

典型日期：

```text
2025-12-05:
formal rolling baseline=charge 36, discharge 69
full-tail baseline=charge 37, discharge 71

2025-12-24:
formal rolling baseline=charge 40, discharge 70
full-tail baseline=charge 42, discharge 68
full-tail proposed candidate=charge 50, discharge 72
true_delta_profit=+535.8367742571918

2025-12-18:
formal rolling baseline=charge 38, discharge 66
full-tail baseline=charge 0, discharge 25
baseline drift max_abs_delta=41
```

### 决策

继续不生成新提交候选，不允许降低阈值硬凑单日文件。

原因：

1. 测试期 `pred_rank=1 days=0`，没有任何正式放行候选。
2. 测试期 `risk_expected_delta>=0.10 days=0`，严格风险阈值下没有候选。
3. `safe5117-source-model` 的 baseline 在 12 月有明显漂移，说明历史验证中的“相对保底收益差”锚点仍不够稳定。
4. 如果强行降阈值，会重新回到 `skip_t500` 那种“为了出候选而冒险”的模式，线上风险大于收益。

当前应保留：

```text
output.csv
outputs/output_nwp_unconstrained_online5117.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_summary_20260505.csv
outputs/val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505_day_metrics.csv
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505.csv
outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_summary_20260506.csv
```

当前不应提交：

```text
outputs/output_safe5117_skip_t500.csv
outputs/output_safe5117_skip_t1000.csv
outputs/output_safe5117_skip_t1500.csv
任何没有 blocked=False manifest 的新单日候选
任何 changed_days > 1 的候选
```

### 下一步

下一步仍是研究，不是提交：

1. 给替换候选增加 baseline 稳定性 gate：候选日的 source-model baseline 必须与稳定锚点足够接近。
2. 研究是否能用真实 5117 提交动作构造历史同源锚点，减少 source-model split drift。
3. 若测试期仍为 0 候选，就继续不冒险，保留 5117。
4. 任何新候选都必须先通过：

```powershell
python -m src.check_submission --submission <candidate>
python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate <candidate>
python -m src.guard_submission_candidate --candidate <candidate> --manifest <manifest>
```

## 2026-05-06 更新：baseline 稳定性 gate 接入与复核

### 做了什么

按当前方向实现 baseline 稳定性 gate，而不是放宽 `min-risk-expected-delta`。

改动文件：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

新增核心能力：

```text
baseline_meta_from_attached_windows
normalize_baseline_meta
add_baseline_stability_features
--use-baseline-stability-gate
--baseline-stability-max-abs-delta
```

当前 gate 使用：

```text
baseline_stability_max_abs_delta <= 2
```

训练 / rolling 阶段：先用原始 proxy baseline 作为稳定参考，再切换到 `safe5117-source-model` baseline 训练相对替换。若 source baseline 与 proxy reference 的充放电窗口相差超过 2 个 15 分钟格，则该日候选不能进入最终选择。

测试阶段：用真实 5117 提交动作作为 baseline，再额外训练 source-model 对测试期生成 `source_test_meta`。若 5117 动作和 source-model baseline 相差超过 2 格，则该日候选也不能进入最终选择。

同时修正了测试期 `pred_rank` 标记逻辑：以前可能把同日一阶段最高但风险不通过的行标成 `pred_rank=1`；现在只给最终通过全部 gate 的行标 `pred_rank=1`。

### 正式 rolling 结果

运行命令核心参数：

```text
--daily-top-k 10
--baseline-mode safe5117-source-model
--proba-threshold 0.40
--min-expected-delta -100000
--use-risk-gate
--risk-objective rule
--risk-proba-threshold 1.0
--min-risk-expected-delta 0.10
--use-baseline-stability-gate
--baseline-stability-max-abs-delta 2
```

输出：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v2_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v2_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v2_rolling_scored_20260506/
```

结果：

```text
fold=2025-04 proposed_days=0
fold=2025-07 proposed_days=0
fold=2025-10 proposed_days=0
fold=2025-12 proposed_days=0
false_positive_days=0
```

重要解释：原先 `min-risk=0.10` 下 2025-10 的大正样本 `2025-10-06` 被挡掉了。原因是该日 source baseline 为 `28/60`，proxy 稳定参考为 `54/69`，`baseline_stability_max_abs_delta=26`，锚点严重不一致。这个结果说明稳定性 gate 确实在拦截“锚点漂移”，不是在放宽风险。

### 全量训练与测试期结果

输出：

```text
outputs/val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506_day_metrics.csv
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506.csv
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506_source_test_meta.csv
```

全量尾部验证出现 1 个正向研究信号：

```text
date=2025-12-10
baseline=50/69
candidate=51/73
true_delta_profit=1036.2522658178368
risk_expected_delta=0.11418190279787432
false_positive_days=0
```

测试期出现 1 个研究信号：

```text
date=2026-01-03
baseline=50/68
source_reference=50/68
candidate=51/70
delta_charge_start=1
delta_discharge_start=2
baseline_stability_max_abs_delta=0
risk_expected_delta=0.104161
pred_rank=1
```

### 决策

不生成提交文件，不生成 `blocked=False` manifest，不覆盖 `output.csv`。

原因：

1. 正式 rolling 在稳定性 gate 下 `proposed_days=0`，没有跨 fold 放行证据。
2. 全量尾部和测试期各出现 1 个信号，但这还不能证明可泛化。
3. `2026-01-03 51/70` 是小幅移动研究信号，可继续分析，但不能直接提交。
4. 当前线上最高仍是 5117，任何新候选都必须先满足正式 rolling 有足够证据，再进入 manifest + guard。

当前状态：

```text
submit_allowed=false
recommended_submission=output.csv
research_signal=2026-01-03 50/68 -> 51/70
```

### 下一步

下一步仍然不是提交，而是评估稳定性 gate 的阈值形状：

1. 做 `baseline_stability_max_abs_delta` 网格回放，例如 0、1、2、4、8。
2. 不降低 `min-risk-expected-delta=0.10`。
3. 观察 formal rolling 是否能在低误报前提下恢复少量正样本。
4. 若测试期候选仍只有单个孤立信号，继续不提交。

## 2026-05-06 更新：baseline stability 网格回放

### 做了什么

按计划固定 `min-risk-expected-delta=0.10` 不变，只回放：

```text
baseline_stability_max_abs_delta = 0, 1, 2, 4, 8
```

公共参数保持：

```text
baseline_mode=safe5117-source-model
daily_top_k=10
max_shift=8
proba_threshold=0.40
min_expected_delta=-100000
risk_objective=rule
risk_proba_threshold=1.0
min-risk-expected-delta=0.10
```

新增输出：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability0_grid_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability0_grid_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability1_grid_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability1_grid_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_grid_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_grid_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability8_grid_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability8_grid_rolling_summary_20260506.csv
```

汇总输出：

```text
outputs/replacement_classifier_baseline_stability_grid_minrisk010_summary_20260506.csv
outputs/replacement_classifier_baseline_stability_grid_minrisk010_proposals_20260506.csv
```

### 结果

```text
baseline_stability_max_abs_delta=0 proposed_days=0 false_positive_days=0 total_delta_profit=0
baseline_stability_max_abs_delta=1 proposed_days=0 false_positive_days=0 total_delta_profit=0
baseline_stability_max_abs_delta=2 proposed_days=0 false_positive_days=0 total_delta_profit=0
baseline_stability_max_abs_delta=4 proposed_days=2 false_positive_days=1 total_delta_profit=-2354.2077596053086
baseline_stability_max_abs_delta=8 proposed_days=0 false_positive_days=0 total_delta_profit=0
```

`max_abs_delta=4` 是唯一恢复提案的阈值，但结果失败：

```text
2025-12-30:
baseline=48/68
candidate=49/71
true_delta_profit=-3626.752494527096
risk_expected_delta=0.1155377470930429

2025-12-31:
baseline=48/66
candidate=49/71
true_delta_profit=+1272.5447349217875
risk_expected_delta=0.1775353928299283
```

通俗解释：阈值 4 确实让模型恢复了候选，但同时放出一个大亏损日，合计收益为负。阈值 8 不是更好，而是因为训练阶段的 stability gate 变宽后，rule gate 的分位数阈值重新校准，最终反而无提案。这个非单调现象说明稳定性 gate 会改变二阶段规则分布，不能只看“阈值越大越宽松”。

### 决策

当前不提交任何新候选，也不为 `2026-01-03 50/68 -> 51/70` 生成可提交 manifest。

原因：

1. `0/1/2` 全部 0 提案，没有 rolling 正样本支撑。
2. `4` 明确出现 12 月大误报，且总 delta 为负。
3. `8` 无提案，不能证明恢复了稳健信号。
4. 当前网格没有找到“低误报 + 恢复少量正样本”的阈值。

状态保持：

```text
submit_allowed=false
recommended_submission=output.csv
```

## 2026-05-06 更新：二阶段加入放电尖峰风险/平台强度后的 formal rolling 结果

### 操作

按当前冲分规则，先把“移出放电窗口尖峰风险 / 新窗口平台强度”加入二阶段风险校准，再重跑正式 rolling。核心改动已经进入代码：

```text
src/train_window_ranker.py
src/replacement_classifier.py
```

新增的形态信息包括：

```text
charge/discharge/spread 的 std/min/max 窗口统计
discharge_move_spike_risk_proxy
discharge_move_plateau_strength_proxy
discharge_shape_risk_balance
risk_rule_discharge_spike_margin
risk_rule_discharge_plateau_margin
risk_rule_discharge_shape_balance_margin
```

通俗解释：这次不是继续放宽阈值，而是让二阶段 gate 明确回答两个问题：旧放电窗口里有没有会被移出的尖峰，新放电窗口是不是稳定高价平台。只有“没有明显错过尖峰风险，且新窗口平台强度够”的替换才可能被放行。

### 验证

已重新确认根目录提交文件没有被覆盖：

```text
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
python -m compileall src
passed
```

### formal rolling 结果

第一版只加入 spike/plateau margin，但没有强制 shape balance，仍失败：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_rolling_summary_20260506.csv

aggregate proposed_days=2
aggregate positive_selected_days=1
aggregate false_positive_days=1
aggregate total_delta_profit=-2036.07157587486
```

失败原因：12 月 fold 仍放出一个大额误报，说明单独看 spike/plateau 分位数还不够稳。

加入更硬的 `discharge_shape_risk_balance >= 0` 后，误报被压住，但也没有任何正式提案：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_rolling_summary_20260506.csv

fold=1 2025-04 proposed_days=0 false_positive_days=0
fold=2 2025-07 proposed_days=0 false_positive_days=0
fold=3 2025-10 proposed_days=0 false_positive_days=0
fold=4 2025-12 proposed_days=0 false_positive_days=0

aggregate proposed_days=0
aggregate positive_selected_days=0
aggregate false_positive_days=0
aggregate total_delta_profit=0
```

### 决策

本轮不生成测试期单日 manifest，也不生成新提交文件。

原因：用户给出的提交门槛要求同时满足 `proposed_days>0`、无大额误报、测试期 `blocked=False` 单日 manifest、`changed_days=1`。当前 shape-balance 版本只满足“无误报”，但 `proposed_days=0`，因此没有通过第一道必要条件。

当前状态保持：

```text
status=formal_zero_proposals_after_shape_balance_gate
submit_allowed=false
recommended_submission=output.csv
reason=二阶段形态 gate 压住误报，但 formal rolling 没有恢复任何正样本，不能生成 blocked=False manifest
```

### 下一步

不要为了出候选继续硬降阈值。下一步应做二阶段风险校准的可解释拆分：

1. 对 shape-balance 版本中被挡掉但真实为正的样本做诊断，确认是 `min-risk-expected-delta=0.10` 太严格，还是 shape balance 过硬。
2. 仍保持 `min-risk-expected-delta=0.10`，优先尝试更细的形态条件，而不是整体放宽。
3. 如果 formal rolling 仍是 0 候选，就继续不冒险提交。

## 2026-05-06 更新：拆解 shape-balance 挡掉的真实正样本，并复跑 hard-pass 版本

### 操作

按用户要求，没有降低 `min-risk-expected-delta=0.10`，而是从已生成的 scored windows 中复现正式日级流程：

1. 先过一阶段 `pred_positive_proba >= 0.40`、`pred_expected_delta >= -100000`、`baseline_stability_pass=1`。
2. 每天只取正式流程会看的 diagnostic selected window。
3. 对真实 `true_delta_profit > 0` 但二阶段没有放行的样本做 blocker 归因。
4. 再试一个更细的二阶段修正：`shape_balance >= 0` 仍作为硬安全条件，但不再参与 `risk_expected_delta` 的最小值打分；也就是不降 `0.10`，只避免把“形态安全余量”误当成“收益安全余量”。

新增诊断产物：

```text
outputs/shape_balance_blocked_true_positive_summary_20260506.csv
outputs/shape_balance_blocked_true_positive_diagnostics_20260506.csv
outputs/shape_balance_near_miss_positive_windows_20260506.csv
outputs/shape_balance_blocked_true_positive_blocker_counts_20260506.csv
outputs/shape_balance_blocked_and_false_positive_source_terms_20260506.csv
outputs/shape_balance_vs_hardpass_stage1_day_drift_20260506.csv
outputs/shape_balance_vs_hardpass_s42_stage1_day_drift_20260506.csv
```

代码改动：

```text
src/replacement_classifier.py
```

### 被挡掉的真实正样本

在旧 shape-balance 产物中，日级正式流程会考虑、且真实收益为正但被二阶段挡住的样本只有 3 个：

```text
fold=2 date=2025-07-25 baseline=48/80 candidate=49/79 true_delta_profit=+2.729608
primary_blocker=shape_balance_too_low

fold=4 date=2025-12-27 baseline=49/69 candidate=51/71 true_delta_profit=+1510.593316
primary_blocker=shape_balance_too_low

fold=4 date=2025-12-31 baseline=48/66 candidate=51/71 true_delta_profit=+1081.663025
primary_blocker=min_risk_expected_delta_buffer
risk_expected_delta=0.092418, gap_to_0.10=0.007582
```

解释：

1. `2025-07-25` 的正收益只有 `+2.73`，几乎没有线上价值；这个样本被挡住不是问题。
2. `2025-12-27` 真实收益较高，但多个源特征显示平台强度不足，`plateau_proxy=0`、`shape_balance=-0.174340`，目前不适合直接放行。
3. `2025-12-31` 是最接近可放行的样本：尖峰风险低、平台强度为正，但 `shape_balance=0.092418` 小于 `0.10`。因此这一步说明原来的 hard score 可能过苛刻，但不能直接降 `min-risk`。

### hard-pass formal rolling 复跑

复跑命令保持 `min-risk-expected-delta=0.10`，只改变二阶段打分方式。先用多 seed，后用历史文档中更常用的 `--seeds 42` 复核：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_hardpass_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_hardpass_s42_rolling_summary_20260506.csv
```

两次 formal rolling 都没有恢复提案：

```text
hardpass multi-seed:
aggregate proposed_days=0
aggregate false_positive_days=0
aggregate total_delta_profit=0

hardpass seeds=42:
aggregate proposed_days=0
aggregate false_positive_days=0
aggregate total_delta_profit=0
```

### 重要发现：复跑稳定性不足

旧 shape-balance 产物中的日级候选包括：

```text
2025-12-27 candidate=51/71 true_delta_profit=+1510.593316
2025-12-31 candidate=51/71 true_delta_profit=+1081.663025
```

但在当前代码和相同 `--seeds 42` 下复跑，日级一阶段候选集合发生漂移，12 月这些候选没有再出现。新的 s42 复跑只出现：

```text
2025-04-25 candidate=54/74 true_delta_profit=-2996.842851
2025-10-12 candidate=53/71 true_delta_profit=-475.473615
```

这说明当前最大问题不是简单调二阶段形态条件，而是 formal rolling 复跑不稳定：一阶段日级候选集合会随着训练重跑发生明显变化。这个信号比单个 12-31 near miss 更重要，因为线上测试期也可能遇到同类漂移。

### 决策

仍然不生成测试期 `blocked=False` manifest，不生成新提交候选，不修改 `output.csv`。

当前状态：

```text
status=blocked_by_zero_proposals_and_stage1_rerun_drift
submit_allowed=false
recommended_submission=output.csv
reason=shape-balance 真实正样本诊断完成，但 hard-pass formal rolling 仍 proposed_days=0，且复跑显示一阶段候选集合漂移；未满足 proposed_days>0
```

### 下一步

下一步不要降阈值，也不要直接训练测试期候选。更合理的方向是：

1. 固定 scored windows 做纯规则 replay，确认二阶段规则本身能否在同一候选集合上稳定地区分 12-30 与 12-31。
2. 再做一阶段候选稳定性校准，例如只接受跨 seed / 跨 rerun 都出现的日级候选。
3. 只有 formal rolling 在稳定候选集合上恢复 `proposed_days>0` 且无大额误报，再进入测试期 manifest。

### 下一步

下一步不要继续简单调 `baseline_stability_max_abs_delta`。更合理的是拆 `max_abs_delta=4` 的误报原因，尤其是：

1. 2025-12-30 和 2025-12-31 都是 `49/71`，但一个大亏一个盈利，说明日内价格形态特征还区分不开。
2. 增加“相邻日同窗口一致但真实 delta 分化”的风险特征。
3. 针对 12 月末日历/节假日/年末负荷形态做额外校准。
4. 测试期 `2026-01-03` 仍只保留为研究信号，不进入提交。

## 2026-05-06 更新：拆解 `max_abs_delta=4` 的 12 月末 `49/71` 误报

### 操作

针对用户要求，停止继续调 `baseline_stability_max_abs_delta`，改为直接拆解 formal rolling 中 `max_abs_delta=4` 放出的两个 12 月末样本：

```text
2025-12-30 baseline=48/68 candidate=49/71 true_delta_profit=-3626.752495
2025-12-31 baseline=48/66 candidate=49/71 true_delta_profit=+1272.544735
```

已生成诊断输出：

```text
outputs/dec30_dec31_4971_profit_decomposition_20260506.csv
outputs/dec30_dec31_4971_slot_prices_20260506.csv
outputs/dec30_dec31_4971_shape_features_20260506.csv
```

同时确认：

```text
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

### 真实收益拆解

`2025-12-30`：

```text
baseline 48/68 profit = 7094.436897
candidate 49/71 profit = 3467.684403
charge move contribution = +254.508947
discharge move contribution = -3881.261442
total delta = -3626.752495
```

亏损几乎全部来自放电窗口后移。`68/69/70` 被移出，其中 slot 69 的真实电价为 `4.772043`，是一个尖峰；新加入的 `76/77/78` 只有 `1.272545 / 1.049849 / 1.113477`。所以这天不是“49/71 窗口整体差”，而是后移放电错过了 17:15 尖峰。

`2025-12-31`：

```text
baseline 48/66 profit = 3244.989074
candidate 49/71 profit = 4517.533809
charge move contribution = 0.000000
discharge move contribution = +1272.544735
total delta = +1272.544735
```

这天充电窗口移动不影响收益，关键仍是放电窗口。移出的 `66-70` 是约 `2.16-2.67` 的较平平台，新加入的 `74-78` 全部是 `2.672344` 高价平台，因此后移放电盈利。

### 为什么同样 `49/71` 会一亏一赚

这两个样本表面上都是“充电 +1、放电后移到 71”，但真实风险结构不同：

1. `2025-12-30` 是尖峰型风险：原放电窗口早段有单点尖峰，后移会错过峰值。
2. `2025-12-31` 是平台型机会：后段高价更稳定，后移扩大了高价覆盖。
3. 当前二阶段 rule gate 更相信历史/净负荷边际信号，无法识别“移出窗口里是否有尖峰”。
4. 两天一阶段模型其实都不乐观：`pred_expected_delta=-436.408217`，`pred_positive_proba=0.432208`。真正放行的是 `risk_expected_delta` 规则分数，因此问题集中在二阶段风险校准，而不是继续硬调稳定阈值。

关键形态特征：

```text
2025-12-30 removed_discharge_max=4.772043, added_discharge_mean=1.145290, removed_discharge_spike_risk=3.626752
2025-12-31 removed_discharge_max=2.672344, added_discharge_mean=2.672344, removed_discharge_spike_risk=0.000000
```

### 决策

当前仍不提交任何新候选。

原因不是“不想冲分”，而是现在已有证据说明：`max_abs_delta=4` 能恢复候选，但恢复的是一个“一个大亏、一个小赚”的组合，平均后为负；如果今天拿这个逻辑生成测试期单日，很可能重演 `t500` 的线上回撤。

### 下一步

下一步不再继续调 `baseline_stability_max_abs_delta`，而是给二阶段风险校准增加“窗口形态”特征，再重跑 formal rolling：

1. `removed_discharge_max - added_discharge_mean`：识别后移是否会错过尖峰。
2. `removed_discharge_max_share`：识别原放电窗口收益是否由单点尖峰贡献。
3. `added_discharge_min - removed_discharge_mean`：识别新放电窗口是否真的是稳定高价平台。
4. 若候选是放电后移，要求 `removed_discharge_spike_risk` 不能过高，或者 `added_late_plateau_strength` 必须为正。

提交条件保持不变：

```text
submit_allowed=false
recommended_submission=output.csv
```
