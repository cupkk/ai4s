# AI4S 研究进展日志 20260507

## 总体研究进展

当前线上可信最高分仍是：

```text
5117.832037755039
```

保底文件仍为：

```text
output.csv
outputs/output_nwp_unconstrained_online5117.csv
```

本日开始前已再次确认两者 SHA256 完全一致：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

并复核提交格式：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

本日只做诊断、formal rolling 和规则 replay，没有生成 submission，没有覆盖 `output.csv`。

当前结论：

```text
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
p1_allowed=false
```

原因很直接：charge-only 分支在正式二阶段风险门下没有恢复候选；source-model baseline 在 rolling 与 recent-oof/full-tail 口径之间仍有明显漂移，不能直接作为测试期稳定锚点。

## 2026-05-07 更新：charge-only formal rolling

### 做了什么

延续前一轮“不要降阈值硬出候选”的原则，本轮把 charge-only 从普通 replay 升级为正式 formal rolling 检查。

charge-only 的定义是：

```text
放电窗口保持不变
只小幅移动充电窗口
```

这样做的原因是：此前大额误报多来自“移出真实放电尖峰”或“换到不够强的新放电平台”。如果只动充电窗口，理论上不会错过原来的放电峰，风险更低。

先前有一次 formal rolling 命令因为未启用 risk gate，导致 `risk_expected_delta` 列缺失并报错。本轮改用正式二阶段规则口径重跑：

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
  --use-baseline-stability-gate `
  --baseline-stability-max-abs-delta 4 `
  --use-risk-gate `
  --risk-objective rule `
  --risk-proba-threshold 1.0 `
  --min-risk-expected-delta 0.10 `
  --rule-shape-spike-max 0.10 `
  --rule-shape-plateau-min 0.10 `
  --source-num-boost-round 500 `
  --source-early-stopping-rounds 50 `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --output outputs\charge_only_formal_rule_rolling_day_metrics_20260507.csv `
  --aggregate-output outputs\charge_only_formal_rule_rolling_summary_20260507.csv `
  --scored-output-dir outputs\charge_only_formal_rule_rolling_scored_20260507
```

### formal rolling 结果

输出文件：

```text
outputs/charge_only_formal_rule_rolling_day_metrics_20260507.csv
outputs/charge_only_formal_rule_rolling_summary_20260507.csv
outputs/charge_only_formal_rule_rolling_scored_20260507/
```

正式 rolling 汇总：

```text
fold_1 2025-04: proposed_days=0
fold_2 2025-07: proposed_days=0
fold_3 2025-10: proposed_days=0
fold_4 2025-12: proposed_days=1, positive_selected_days=1, false_positive_days=0
```

唯一被正式 rolling 选中的样本：

```text
date=2025-12-31
baseline=48/66
candidate=49/71
true_delta_profit=+1272.544735
risk_positive_proba=1.0
risk_expected_delta=0.182651
```

注意：这一天不是 charge-only，因为放电窗口从 66 移到了 71。

## 2026-05-07 更新：charge-only replay 和小网格

### 脚本修复

更新脚本：

```text
src/replay_charge_only_replacement_rules.py
```

修复内容：

```text
新增 --risk-proba-threshold
新增 --min-risk-expected-delta
```

原因：之前 charge-only replay 只看一阶段概率和 baseline stability，没有真正执行二阶段 `risk_expected_delta>=0.10`。这会把“正式风险门会拦住”的样本误当成可用信号。

### 一阶段 replay

命令：

```powershell
python -m src.replay_charge_only_replacement_rules `
  --scored-dir outputs\charge_only_formal_rule_rolling_scored_20260507 `
  --source-name charge_earlier_abs1_formal_stage1_only `
  --proba-threshold 0.40 `
  --min-expected-delta -100000 `
  --require-baseline-stability `
  --direction earlier `
  --max-abs-charge-delta 1 `
  --detail-output outputs\charge_only_formal_stage1_detail_earlier_abs1_20260507.csv `
  --summary-output outputs\charge_only_formal_stage1_summary_earlier_abs1_20260507.csv
```

结果：

```text
days=1
positive_days=1
false_positive_days=0
total_delta_profit=+63.627237
```

该样本为：

```text
date=2025-10-12
baseline=52/71
candidate=51/71
true_delta_profit=+63.627237
```

但它的二阶段风险结果为：

```text
risk_positive_proba=0.0
risk_expected_delta=-999.0
```

所以它不能算正式可提交信号。

### 正式 risk010 replay

命令：

```powershell
python -m src.replay_charge_only_replacement_rules `
  --scored-dir outputs\charge_only_formal_rule_rolling_scored_20260507 `
  --source-name charge_earlier_abs1_formal_risk010 `
  --proba-threshold 0.40 `
  --min-expected-delta -100000 `
  --require-baseline-stability `
  --direction earlier `
  --max-abs-charge-delta 1 `
  --risk-proba-threshold 1.0 `
  --min-risk-expected-delta 0.10 `
  --detail-output outputs\charge_only_formal_risk010_detail_earlier_abs1_20260507.csv `
  --summary-output outputs\charge_only_formal_risk010_summary_earlier_abs1_20260507.csv
```

结果：

```text
days=0
positive_days=0
false_positive_days=0
total_delta_profit=0
```

### 小网格 replay

输出文件：

```text
outputs/charge_only_formal_rule_grid_20260507.csv
outputs/charge_only_formal_rule_grid_selected_details_20260507.csv
```

网格范围：

```text
use_risk=false/true
direction=earlier/later/any
max_abs_charge_delta=1/2/4/8
recent_centered_min=none/0/0.05/0.10/0.20
hist_centered_min=none/0/0.05/0.10/0.20
```

关键结果：

```text
formal risk010 configs with days>0: NONE
```

一阶段不带 risk 的最好结果仍只有 1 天小正样本：

```text
stage1_earlier_abs1: days=1, false_positive_days=0, total_delta_profit=+63.627237
```

如果放宽到 `charge_later_abs2`，马上出现误报：

```text
worst_delta_profit=-539.100852
```

结论：

```text
charge-only 有一点弱信号，但正式 risk010 口径下没有候选。
不能进入测试期 manifest。
不能生成 changed_days=1 submission。
```

## 2026-05-07 更新：source baseline 漂移原因分析

### 做了什么

新增只读诊断输出：

```text
outputs/source_baseline_drift_window_shape_summary_20260507.csv
outputs/source_baseline_drift_window_rankings_20260507.csv
outputs/source_baseline_drift_slot_prices_focus_20260507.csv
outputs/source_baseline_drift_training_window_summary_20260507.csv
```

重点比较：

```text
rolling fold_04 source predict_meta
vs
full train recent-oof train_meta
```

关注日期：

```text
2025-12-31
2025-12-30
2025-12-18
2025-12-09
2025-12-02
```

### 关键代码发现

`train_replacement_classifier.py` 中这段输出命名容易误导：

```text
*_safe5117_source_full_meta.csv
*_safe5117_source_full_predictions.csv
```

实际保存的是：

```text
full_source_baseline.train_meta
full_source_baseline.train_predictions
```

在 `source_train_baseline_mode=recent-oof` 下，`train_meta` 不是真正的 full predict baseline，而是最后 59 天的 OOF baseline。

也就是说，之前口头说的“full refit baseline 漂移”应更准确地写成：

```text
rolling predict_meta 与 full-train recent-oof train_meta 的 baseline 漂移
```

这很重要，因为两个 baseline 的训练截止点和预测口径不完全一致。

### 训练窗口差异

诊断文件：

```text
outputs/source_baseline_drift_training_window_summary_20260507.csv
```

结果：

```text
rolling_fold04_predict_meta:
work_train=2025-01-01..2025-11-30
source_train=2025-01-01..2025-10-02
source_val=2025-10-03..2025-11-30
baseline_target=2025-12-01..2025-12-31

full_train_recent_oof_meta:
work_train=2025-01-01..2025-12-31
source_train=2025-01-01..2025-11-02
source_val=2025-11-03..2025-12-31
baseline_target=2025-11-03..2025-12-31
```

通俗解释：

```text
rolling fold_04 是用 11 月底以前的信息去预测 12 月；
full recent-oof 是用更靠后的训练截止点，并把 11-12 月作为 OOF baseline。
两者不是同一个 baseline 生成问题。
```

### 2025-12-31 漂移原因

诊断文件：

```text
outputs/source_baseline_drift_window_shape_summary_20260507.csv
outputs/source_baseline_drift_window_rankings_20260507.csv
```

核心对比：

```text
2025-12-31 rolling baseline = 48/66, pred_best_profit=958.769397
2025-12-31 full-oof baseline = 47/80, pred_best_profit=827.451317
max_abs_delta=14
```

rolling 版本的 top window 集中在 `discharge_start=66`：

```text
rolling rank1: charge=50, discharge=66
rolling rank2: charge=48, discharge=66
rolling rank3: charge=49, discharge=66
```

full-oof 版本的 top window 大量并列在 `discharge_start=80..84`：

```text
full-oof rank1: charge=50, discharge=84
full-oof rank5: charge=49, discharge=80
full-oof rank19: charge=47, discharge=80
```

所以 2025-12-31 的漂移不是简单的整体价格抬升，而是日内形状被重排：

```text
rolling_top8=66/67/68/69/70/71/72/73
full_oof_top8=66/67/68/69/70/71/84/85
```

换句话说，full-oof 口径把原本集中在 66-73 的晚间峰，拉成了 66-71 与 84-85 混合的平台；优化器在多个并列窗口里选择了更晚的 discharge window，导致 baseline 从 48/66 漂到 47/80。

### 漂移结论

当前 source-model baseline 不能直接作为“稳定保底锚点”使用，原因有三点：

```text
1. rolling predict_meta 与 full recent-oof train_meta 训练截止点不同；
2. 同一天的日内预测形状会重排，不只是整体价格偏移；
3. baseline 优化器在平台/并列收益窗口中容易跳到很远的等价窗口。
```

因此后续不能把 `safe5117-source-model` baseline 当作真实 5117 保底窗口的替代品。它可以作为稳定性参考，但训练和提交仍应锚定真实保底文件：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

## 当前决策

本轮不进入 P1，不生成 manifest，不生成提交文件。

守门状态：

```text
formal_rolling_proposed_days=1
formal_rolling_false_positive_days=0
formal_rolling_positive_selected_days=1
charge_only_stage1_days=1
charge_only_stage1_false_positive_days=0
charge_only_stage1_total_delta_profit=+63.627237
charge_only_risk010_days=0
charge_only_formal_grid_risk010_days_gt0=0
submit_allowed=false
recommended_submission=output.csv
```

为什么不提交：

```text
1. 唯一 formal rolling 强正样本不是 charge-only；
2. charge-only 一阶段只有 1 个很小正样本，且被二阶段 risk gate 拦住；
3. 测试期仍没有 blocked=False manifest；
4. 没有 changed_days=1 的候选 submission；
5. baseline 漂移问题仍未解决。
```

## 下一步

建议下一轮继续做两件事：

```text
1. 把 source-model baseline 只作为稳定性参考，不再作为训练 delta label 的唯一锚点。
2. 训练/回放时始终使用真实 5117 文件动作作为 baseline anchor，再把 source-model drift 特征作为风险特征。
```

具体可执行方向：

```text
1. 重新构造 safe5117-real-anchor 的历史同源 baseline meta。
2. 对平台并列窗口增加 tie-break 稳定性特征，例如 top window 起点跨度、topN discharge_start 方差、同收益平台宽度。
3. 对 charge-only 分支继续扩大历史 replay 覆盖，但保持 risk010，不降阈值。
4. 只有 formal rolling 同时满足 proposed_days>0、false_positive_days=0、无大额误报，并且测试期生成 blocked=False 单日 manifest 后，才进入 changed_days=1 守门流程。
```

必须继续保留的文件：

```text
output.csv
outputs/output_nwp_unconstrained_online5117.csv
outputs/charge_only_formal_rule_rolling_day_metrics_20260507.csv
outputs/charge_only_formal_rule_rolling_summary_20260507.csv
outputs/charge_only_formal_rule_grid_20260507.csv
outputs/source_baseline_drift_window_shape_summary_20260507.csv
outputs/source_baseline_drift_window_rankings_20260507.csv
outputs/source_baseline_drift_training_window_summary_20260507.csv
```

禁止误提交的文件：

```text
outputs/charge_only_formal_rule_grid_20260507.csv
outputs/charge_only_formal_rule_grid_selected_details_20260507.csv
outputs/source_baseline_drift_*.csv
outputs/test_windows_replacement_classifier_*.csv
```

这些都是诊断或打分输出，不是 submission。
