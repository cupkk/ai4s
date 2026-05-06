# 线上反馈记录 2026-05-03

## 今日策略

今日目标是稳健小步冲分，不提交整套新模型。当前最高可信线上分仍为：

```text
5117.832037755039
```

保底文件：

```text
outputs/output_nwp_unconstrained_online5117.csv
SHA256: AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

根目录 `output.csv` 当前保持为保底文件，不因本地验证自动覆盖。

## 今日允许提交池

| 顺序 | 文件 | 条件 | 预期改动 |
|---:|---|---|---|
| 回退 | `outputs/output_nwp_unconstrained_online5117.csv` | 当前最高线上保底 | 与保底完全一致 |

`outputs/output_safe5117_skip_t500.csv` 已线上验证低于保底，因此今日不再允许继续提交 `t1000/t1500` 等扩大跳过天数的候选。

## 今日禁止提交

```text
outputs/output_nwp_c0_55_d72_88.csv
outputs/output_blend_fine_w025_t1000.csv
outputs/output_nwp_unconstrained_t2000.csv
outputs/output_residual_nwp.csv
outputs/output_window_ranker_c055_d7288.csv
outputs/output_safe5117_skip_t500.csv
outputs/output_safe5117_skip_t1000.csv
outputs/output_safe5117_skip_t1500.csv
changed_days > 5 的任何候选
```

原因：这些文件要么已经线上低于保底，要么本地验证弱于主线，要么相对保底改动过大。

## 提交前守门命令

每个候选提交前运行：

```powershell
python -m src.guard_submission_candidate `
  --candidate <候选文件> `
  --candidate-name <候选名> `
  --diff-output outputs/action_diff_safe5117_vs_<候选名>.csv `
  --summary-output outputs/guard_summary_<候选名>.csv
```

验收标准：

```text
decision=PASS
rows=5664
days=59
check_errors=0
check_warnings=0
changed_days 符合今日允许提交池
```

## 反馈填写模板

### 第 N 次提交

```text
提交时间：
提交文件：
SHA256：
changed_days：
线上分数：
结论：
下一步动作：
```

## 当前待提交建议

当前唯一推荐：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

通俗解释：`t500` 已经证明 `2026-01-11` 不能跳过；保底文件中这一天实际大概率有正贡献。继续提交 `t1000/t1500` 会同时跳过更多日期，风险更高，不符合今日稳健冲分原则。

## 收益导向重训闭环更新

已实现“重新训练收益导向模型，但只允许生成单日替换候选”的工程闭环：

```text
src/train_window_ranker.py
src/make_single_day_candidates.py
src/rolling_validate_window_ranker.py
src/guard_submission_candidate.py
scripts/guard_submission_candidate.ps1
```

新的窗口模型直接枚举每天所有合法 `(charge_start, discharge_start)`，用真实窗口收益做标签，并输出：

```text
avg_profit
capture_ratio
regret
top1_window_hit
top3_window_hit
top1_minus_top2_margin
score_std
```

正式 regression 版本在 2025-01/02 验证：

```text
avg_profit=10609.214886
capture_ratio=0.636997
regret=6045.839414
top1_window_hit=0
top3_window_hit=0
top1_minus_top2_margin=0
```

短轮数 `lambdarank` 烟测：

```text
avg_profit=11410.198569
capture_ratio=0.685089
regret=5244.855730
top1_window_hit=0
top3_window_hit=0
top1_minus_top2_margin=0
```

rolling validation 结果不稳定：

```text
2025-04 avg_profit=-454.620718, capture_ratio=-0.037848
2025-07 avg_profit=642.862610, capture_ratio=0.065094
2025-10 avg_profit=3707.872398, capture_ratio=0.266332
2025-12 avg_profit=-63.427119, capture_ratio=-0.006258
```

结论：收益导向闭环已经搭好，但模型还不能解锁线上提交。生成的单日候选目前都是研究候选，manifest 默认 `blocked=True`，守门脚本会拒绝。

当前研究候选：

```text
outputs/output_df_single_20260125_df_reg.csv
outputs/output_df_single_20260205_df_reg.csv
outputs/output_df_single_20260115_df_reg.csv
outputs/single_day_candidate_manifest_df_reg.csv
```

示例守门结果：

```text
outputs/output_df_single_20260125_df_reg.csv
decision=FAIL
changed_days=1
原因：manifest marks candidate as blocked
```

通俗解释：代码现在已经能做到“只改 1 天再提交”，但模型证据还没过关。与其现在冒险提交，不如先修模型，让它在 4/7/10/12 月 rolling validation 都不崩，再把 manifest 重新生成为可提交版本。

## 相对保底收益差训练更新

已按最新策略把训练目标改为“候选窗口相对 5117 保底窗口的单日收益差”，更贴近真实线上策略：

```text
true_delta_profit = true_window_profit - baseline_true_window_profit
```

本轮关键输出：

```text
outputs/window_ranker_delta_reg_metadata.json
outputs/val_window_ranker_delta_reg_day_metrics.csv
outputs/test_windows_window_ranker_delta_reg.csv
outputs/window_ranker_delta_reg_rolling_summary.csv
```

验证结果：

```text
2025-01/02 avg_delta_profit=-9973.036823
positive_delta_rate=0.071429

2025-04 avg_delta_profit=-7462.127327, positive_delta_rate=0.033333
2025-07 avg_delta_profit=-3150.002230, positive_delta_rate=0.322581
2025-10 avg_delta_profit=-6495.619626, positive_delta_rate=0.161290
2025-12 avg_delta_profit=-5113.229856, positive_delta_rate=0.161290
```

测试期 59 天里，模型预测相对保底为正的日期数：

```text
0
```

候选生成器结果：

```text
ValueError: no eligible single-day replacement candidates
```

结论：这不是脚本失败，而是守门逻辑生效。当前 delta 模型没有给出任何“替换 1 天大概率赚钱”的信号，因此今日不应提交新模型候选，也不应提交 `outputs/output_window_ranker_delta_reg_full.csv`。下一步应训练更保守的 `true_delta_profit > 0` 替换分类器，并把候选限制在保底窗口附近小位移。

## 保守替换分类器更新

已实现“只在保底窗口附近小幅移动，训练 `true_delta_profit > 0` 的替换分类器/校准器”的闭环：

```text
src/replacement_classifier.py
src/train_replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/make_single_day_candidates.py
```

严格 rolling validation：

```text
max_shift=8
proba_threshold=0.75
min_expected_delta=0

2025-04 proposed_days=0
2025-07 proposed_days=0
2025-10 proposed_days=0
2025-12 proposed_days=0
```

宽松诊断：

```text
max_shift=8
proba_threshold=0.40
min_expected_delta=-100000

2025-07 proposed_days=31, false_positive_rate=0.354839, total_delta=15185.479929
2025-10 proposed_days=9, false_positive_rate=0.555556, total_delta=-7768.955597
```

结论：严格阈值下没有误报，但也没有候选；放宽阈值后误报率明显偏高。因此当前不生成 `blocked=False` manifest，不提交替换分类器候选。

测试期输出：

```text
outputs/test_windows_replacement_classifier_ms8_p075.csv
```

该文件只是窗口打分表，不是提交文件。候选生成器复核结果：

```text
ValueError: no eligible single-day replacement candidates
```

通俗解释：新闭环已经能自动做到“没有足够把握就不出手”。这次没有新候选，是因为模型还不能低误报地指出哪一天该替换保底窗口，而不是因为流程缺失。

## 已收到反馈

### 第 1 次提交

```text
提交时间：2026-05-03 10:45:40
提交文件：outputs/output_safe5117_skip_t500.csv
SHA256：76180A22481C3718CF7E31C068EE94C092ADE1A79038AF6EB81CBF56FCED7218
changed_days：1
线上分数：4987.610162489461
结论：低于 5117.832037755039；跳过 2026-01-11 失败，说明保底文件该日交易实际有正贡献或至少不应被跳过。
下一步动作：停止提交 t1000/t1500 等扩大 skip threshold 的候选，回到保底文件。
```


## 5117 ?? baseline meta ????

?????????????????????????

????

```text
src/safe5117_baseline.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

?????????????????? proxy baseline?????????? proxy baseline ????????????? 5117 ?????????????????

????

```text
--baseline-mode safe5117-source-model
```

????????? 5117-source ?? baseline meta???? `true_delta_profit > 0` ?????????

???

```text
strict/?????proposed_days=0
??? 0.30 stop-loss ???4/10/12 ?????????? 0.645161 ? 0.833333
??? eligible_rank_rows=0
??????ValueError: no eligible single-day replacement candidates
```

???????? `blocked=False` manifest????????????????????? rolling validation ????????????????
