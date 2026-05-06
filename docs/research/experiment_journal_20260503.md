# AI4S 研究进度日志 2026-05-03

## 总体研究进度

项目目标：优化 AI4S 电价预测与储能策略比赛提交文件。线上评分只看 `output.csv` 中 `power` 带来的平均日收益，即每天 `sum(真实电价 * power)` 后再取 59 天平均值。预测电价列只影响我们如何选择充放电窗口，不直接计分。

当前最高可信线上分：

```text
5117.832037755039
```

当前保底文件：

```text
output.csv
outputs/output_nwp_unconstrained_online5117.csv
SHA256: AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

已证伪方向：

- `outputs/output_nwp_c0_55_d72_88.csv` 线上 `3798.629342284567`，整套窗口漂移过大。
- `outputs/output_blend_fine_w025_t1000.csv` 线上 `4703.505815153465`，融合过拟合。
- `outputs/output_nwp_unconstrained_t2000.csv` 线上 `4903.504068225546`，阈值过度防守。
- `outputs/output_safe5117_skip_t500.csv` 线上 `4987.610162489461`，跳过 `2026-01-11` 是错误方向。

当前主线：继续以 5117 保底文件为锚点，不再提交 skip threshold 系列；新模型只能提出“单日替换候选”，不能直接替换整套 59 天策略。

## 2026-05-03 工作记录

### 1. 接手确认

已确认：

```text
git status:
 M docs/research/candidate_status.md
?? README.md
?? docs/research/handoff_2026-05-02.md
?? docs/research/online_feedback_2026-05-02.md
?? docs/research/online_feedback_2026-05-03.md
?? scripts/guard_submission_candidate.ps1
?? src/analyze_submission_diff.py
?? src/guard_submission_candidate.py
```

`output.csv` 与 5117 保底文件 hash 一致：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

合法性检查：

```text
python -m src.check_submission --submission output.csv
rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

### 2. 收益导向窗口模型增强

修改文件：

```text
src/train_window_ranker.py
```

新增能力：

- 支持 `--objective-mode regression|lambdarank`。
- 输出每个窗口的多 seed 预测列和 `pred_window_profit_std`。
- 输出每天 `top1_minus_top2_margin`。
- 输出验证单日指标：`avg_profit`、`capture_ratio`、`regret`、`top1_window_hit`、`top3_window_hit`。
- 保存全量窗口排序文件和单日指标文件，便于后续挑单日替换，而不是只看整套 submission。

正式 regression 运行结果：

```text
window_ranker_validation=
avg_profit=10609.214886
capture_ratio=0.636997
regret=6045.839414
top1_window_hit=0.000000
top3_window_hit=0.000000
top1_minus_top2_margin=0.000000
```

输出文件：

```text
outputs/window_ranker_df_reg_metadata.json
outputs/val_windows_window_ranker_df_reg.csv
outputs/val_ranked_windows_window_ranker_df_reg.csv
outputs/val_window_ranker_df_reg_day_metrics.csv
outputs/test_windows_window_ranker_df_reg.csv
outputs/output_window_ranker_df_reg_full.csv
outputs/window_ranker_df_reg_meta.csv
```

短轮数 `lambdarank` 烟测结果：

```text
avg_profit=11410.198569
capture_ratio=0.685089
regret=5244.855730
top1_window_hit=0.000000
top3_window_hit=0.000000
top1_minus_top2_margin=0.000000
```

结论：`lambdarank` 目标代码路径可用，值得继续研究；但 exact hit 和 margin 仍为 0，说明模型对窗口细粒度排序还不够可靠。

### 3. 新增单日候选生成器

新增文件：

```text
src/make_single_day_candidates.py
```

功能：

- 输入 5117 保底文件和模型推荐窗口。
- 每个输出候选只改 1 天。
- 禁止 `2026-01-11`。
- 禁止 no-trade 替换。
- 自动生成 manifest，记录 candidate hash、日期、原窗口、新窗口、模型分数、不确定性、margin、理由。
- 默认 `blocked=True`，即研究候选不能被守门脚本放行；只有明确加 `--allow-submission` 才能生成可提交 manifest。

本轮生成的研究候选：

```text
outputs/output_df_single_20260125_df_reg.csv
outputs/output_df_single_20260205_df_reg.csv
outputs/output_df_single_20260115_df_reg.csv
outputs/single_day_candidate_manifest_df_reg.csv
```

它们都只改 1 天，但目前禁止提交。

### 4. 守门脚本增强

修改文件：

```text
src/guard_submission_candidate.py
scripts/guard_submission_candidate.ps1
```

新增规则：

- 默认仍只放行 `outputs/output_nwp_unconstrained_online5117.csv`。
- 静态黑名单优先生效，继续阻止 `t500/t1000/t1500`、`c0_55_d72_88`、blend、t2000 等候选。
- 动态 manifest 只允许单日候选。
- manifest 必须匹配 candidate path 和 SHA256。
- `changed_days` 必须等于 1。
- changed date 必须与 manifest date 一致。
- 不允许 `2026-01-11`。
- 不允许改成 no-trade。
- `blocked=True` 必须 FAIL。

验证结果：

```text
python -m src.guard_submission_candidate --candidate outputs/output_nwp_unconstrained_online5117.csv
decision=PASS
changed_days=0
```

```text
python -m src.guard_submission_candidate --candidate outputs/output_safe5117_skip_t1000.csv
decision=FAIL
changed_days=3
原因：blocked because t500 failed
```

```text
python -m src.guard_submission_candidate --candidate outputs/output_df_single_20260125_df_reg.csv --manifest outputs/single_day_candidate_manifest_df_reg.csv
decision=FAIL
changed_days=1
原因：manifest marks candidate as blocked
```

注意：不要把候选生成和 guard 检查并行执行。曾经并行运行时，guard 可能读到旧 manifest，导致短暂误判。正确流程是先生成 manifest，再顺序运行 guard。

### 5. 新增窗口模型 rolling validation

新增文件：

```text
src/rolling_validate_window_ranker.py
```

命令：

```powershell
python -m src.rolling_validate_window_ranker `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --seeds 42 `
  --objective-mode regression `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --output outputs/window_ranker_df_reg_rolling_day_metrics.csv `
  --aggregate-output outputs/window_ranker_df_reg_rolling_summary.csv `
  --ranked-output-dir outputs/window_ranker_df_reg_rolling_ranked
```

结果：

```text
2025-04 avg_profit=-454.620718, capture_ratio=-0.037848, negative_selected_days=19
2025-07 avg_profit=642.862610, capture_ratio=0.065094, negative_selected_days=15
2025-10 avg_profit=3707.872398, capture_ratio=0.266332, negative_selected_days=9
2025-12 avg_profit=-63.427119, capture_ratio=-0.006258, negative_selected_days=16
```

结论：收益导向窗口模型在 2025-01/02 验证看起来改善，但滚动验证明显不稳，当前不能解锁线上提交。

### 6. 当前决定

不提交任何新生成的单日候选。

原因：模型还没有稳定证明自己能挑出“相对 5117 保底更好的单日窗口”。现在贸然提交 `2026-01-25` 或其他单日替换，本质上还是用不稳定模型赌线上反馈，不符合当前只剩有限提交次数的冲分策略。

当前唯一可提交/回退文件仍然是：

```text
outputs/output_nwp_unconstrained_online5117.csv
output.csv
```

## 下一步任务

1. 不再扩展 skip threshold，不再提交 `t1000/t1500`。
2. 继续改进窗口模型，但验收标准从 2025-01/02 单点收益改成 rolling validation：
   - 4、7、10、12 月不能出现平均收益为负。
   - `negative_selected_days` 要明显下降。
   - `top1_minus_top2_margin` 不能长期为 0。
3. 优先修模型筛选能力：
   - `baseline-delta` 标签已经完成，下一步不是再改标签，而是训练“是否值得替换”的保守分类器/校准器。
   - 用窗口内价格历史 spread 的排名特征，而不仅是均值差。
   - 限制候选只在保底窗口附近小幅移动，例如充放电起点偏移不超过 4-8 个 15 分钟格。
   - 只在 rolling validation 的单日替换平均收益差为正、误报率低时，才允许生成可提交 manifest。
4. 单日候选只有在 rolling validation 通过后，才允许用：

```powershell
python -m src.make_single_day_candidates ... --allow-submission
```

重新生成可提交 manifest。

5. 每个线上候选提交前必须顺序运行：

```powershell
python -m src.check_submission --submission <candidate>
python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate <candidate> --reference-name safe5117 --candidate-name <tag>
python -m src.guard_submission_candidate --candidate <candidate> --manifest <manifest>
```

### 7. 已改为相对保底窗口的单日收益差训练

本轮按最新策略把窗口模型训练目标从“绝对窗口收益”改成了“相对保底窗口的单日收益差”：

```text
true_delta_profit = true_window_profit - baseline_true_window_profit
```

修改文件：

```text
src/train_window_ranker.py
src/rolling_validate_window_ranker.py
src/make_single_day_candidates.py
```

关键实现：

- `src/train_window_ranker.py` 新增 `--label-mode absolute|baseline-delta`。
- `baseline-delta` 模式会给每个候选窗口附加保底窗口上下文：保底充电起点、保底放电起点、与保底窗口的起点偏移、gap 偏移、历史价差偏移。
- 训练标签使用 `true_delta_profit`，而不是 `true_window_profit`。
- 2026 测试期必须显式传入 `--test-baseline-submission outputs/output_nwp_unconstrained_online5117.csv`，避免候选生成脱离 5117 保底锚点。
- `src/make_single_day_candidates.py` 默认要求 `--min-pred-score 0.0`，因此 baseline-delta 模型只有在预测“替换后相对保底有正收益”时才会生成候选。

主训练命令：

```powershell
python -m src.train_window_ranker `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --test-feature to_sais_new\to_sais_new\test\test_in_feature_ori.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --objective-mode regression `
  --label-mode baseline-delta `
  --test-baseline-submission outputs/output_nwp_unconstrained_online5117.csv `
  --model-output outputs/window_ranker_delta_reg_model.txt `
  --metadata-output outputs/window_ranker_delta_reg_metadata.json `
  --val-window-output outputs/val_windows_window_ranker_delta_reg.csv `
  --val-ranked-window-output outputs/val_ranked_windows_window_ranker_delta_reg.csv `
  --val-day-metrics-output outputs/val_window_ranker_delta_reg_day_metrics.csv `
  --test-window-output outputs/test_windows_window_ranker_delta_reg.csv `
  --submission-output outputs/output_window_ranker_delta_reg_full.csv `
  --meta-output outputs/window_ranker_delta_reg_meta.csv
```

2025-01/02 验证结果：

```text
avg_profit=308.046665
avg_delta_profit=-9973.036823
positive_delta_rate=0.071429
capture_ratio=0.018496
```

rolling validation：

```text
2025-04 avg_delta_profit=-7462.127327, positive_delta_rate=0.033333
2025-07 avg_delta_profit=-3150.002230, positive_delta_rate=0.322581
2025-10 avg_delta_profit=-6495.619626, positive_delta_rate=0.161290
2025-12 avg_delta_profit=-5113.229856, positive_delta_rate=0.161290
```

测试期候选检查：

```text
59 天中预测相对保底正收益天数 = 0
最高预测 delta = 2026-02-15, -4392.883070
```

候选生成器结果：

```text
ValueError: no eligible single-day replacement candidates
```

复核命令：

```powershell
python -m src.make_single_day_candidates `
  --baseline outputs/output_nwp_unconstrained_online5117.csv `
  --ranked-windows outputs/test_windows_window_ranker_delta_reg.csv `
  --manifest-output outputs/single_day_candidate_manifest_delta_reg.csv `
  --tag delta_reg `
  --max-candidates 3 `
  --min-pred-score 0
```

结果仍然是：

```text
ValueError: no eligible single-day replacement candidates
```

结论：这次改法方向是正确的，因为它和真实提交策略一致：只考虑“替换 1 天是否比 5117 保底更好”。但当前模型学到的是强烈负信号，说明它还不能可靠识别可替换日期。当前不能生成线上候选，不能提交 `outputs/output_window_ranker_delta_reg_full.csv`，也不能把任何 delta 模型输出覆盖 `output.csv`。

下一步应改成更保守的二阶段方法：

1. 先固定候选池为“保底窗口附近的小位移”，不再自由选择全日窗口。
2. 对每个候选训练 `true_delta_profit > 0` 的分类器或校准器。
3. 只输出 predicted delta 为正、跨 seed 分歧小、rolling validation 误报率低的单日候选。
4. 候选仍必须 `changed_days=1`、manifest 合法、guard `PASS` 后才能考虑提交。

### 8. 收尾安全检查

本轮未覆盖根目录 `output.csv`。收尾检查：

```text
python -m compileall src
通过

python -m src.check_submission --submission output.csv
rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate output.csv --reference-name safe5117 --candidate-name output_current
changed_days=0

python -m src.guard_submission_candidate --candidate outputs/output_nwp_unconstrained_online5117.csv --candidate-name safe5117_fallback
decision=PASS
changed_days=0

output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

### 9. 保守替换分类器闭环

按最新任务实现“只在保底窗口附近小幅移动，训练 `true_delta_profit > 0` 的替换分类器/校准器，rolling validation 误报率低后再生成 `blocked=False` 单日 manifest”。

新增文件：

```text
src/replacement_classifier.py
src/train_replacement_classifier.py
src/rolling_validate_replacement_classifier.py
```

小改文件：

```text
src/make_single_day_candidates.py
```

实现要点：

- 候选池只保留接近 baseline 的窗口，默认 `max_shift=8`，即充电起点和放电起点相对保底各最多移动 8 个 15 分钟格。
- 训练标签是二分类：`positive_delta_label = true_delta_profit > 0`。
- 模型输出 `pred_positive_proba`。
- 用训练集概率桶做简单校准，得到 `pred_expected_delta`。
- 单日候选生成器现在能读取 classifier 输出，并在 manifest 里记录 `pred_positive_proba`、`pred_delta_profit`、`score_std`、窗口起点和理由。
- 测试期如果 rolling validation 未通过，训练脚本会把所有测试窗口 `pred_rank` 置为非候选值，避免误生成可提交 manifest。

正式低风险 rolling validation：

```powershell
python -m src.rolling_validate_replacement_classifier `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --seeds 42 `
  --max-shift 8 `
  --proba-threshold 0.75 `
  --min-expected-delta 0 `
  --num-boost-round 500 `
  --early-stopping-rounds 50 `
  --output outputs/replacement_classifier_ms8_p075_rolling_day_metrics.csv `
  --aggregate-output outputs/replacement_classifier_ms8_p075_rolling_summary.csv `
  --scored-output-dir outputs/replacement_classifier_ms8_p075_rolling_scored
```

结果：

```text
2025-04 proposed_days=0, false_positive_rate=0
2025-07 proposed_days=0, false_positive_rate=0
2025-10 proposed_days=0, false_positive_rate=0
2025-12 proposed_days=0, false_positive_rate=0
```

解释：严格阈值下没有误报，但也没有任何可冲分候选，因此不能生成 `blocked=False` manifest。

诊断性宽松运行：

```text
max_shift=8, proba_threshold=0.40, min_expected_delta=-100000
2025-07 proposed_days=31, false_positive_rate=0.354839, total_delta=15185.479929
2025-10 proposed_days=9, false_positive_rate=0.555556, total_delta=-7768.955597
```

解释：只要放宽阈值，模型会产生较高误报；10 月折直接负收益。这说明当前分类器还不能满足“误报率低后放行”的条件。

正式训练到 2026 测试期：

```powershell
python -m src.train_replacement_classifier `
  --train-feature to_sais_new\to_sais_new\train\mengxi_boundary_anon_filtered.csv `
  --train-label to_sais_new\to_sais_new\train\mengxi_node_price_selected.csv `
  --test-feature to_sais_new\to_sais_new\test\test_in_feature_ori.csv `
  --nwp-dir to_sais_new\to_sais_new\all_nc `
  --val-start-date 2025-01-01 `
  --val-end-date 2025-02-28 `
  --seeds 42,2024,2026 `
  --max-shift 8 `
  --proba-threshold 0.75 `
  --min-expected-delta 0 `
  --test-baseline-submission outputs/output_nwp_unconstrained_online5117.csv `
  --num-boost-round 800 `
  --early-stopping-rounds 60 `
  --model-output outputs/replacement_classifier_ms8_p075_model.txt `
  --metadata-output outputs/replacement_classifier_ms8_p075_metadata.json `
  --val-window-output outputs/val_windows_replacement_classifier_ms8_p075.csv `
  --val-day-metrics-output outputs/val_replacement_classifier_ms8_p075_day_metrics.csv `
  --test-window-output outputs/test_windows_replacement_classifier_ms8_p075.csv
```

结果：

```text
replacement_classifier_validation:
proposed_days=0
false_positive_rate=0

outputs/test_windows_replacement_classifier_ms8_p075.csv:
pred_rank=999999 for all rows
```

候选生成器复核：

```powershell
python -m src.make_single_day_candidates `
  --baseline outputs/output_nwp_unconstrained_online5117.csv `
  --ranked-windows outputs/test_windows_replacement_classifier_ms8_p075.csv `
  --manifest-output outputs/single_day_candidate_manifest_replacement_ms8_p075.csv `
  --tag repl_ms8_p075 `
  --max-candidates 3 `
  --min-pred-score 0 `
  --allow-submission
```

结果：

```text
ValueError: no eligible single-day replacement candidates
```

结论：保守替换分类器闭环已经搭好，但当前模型没有达到放行标准。今天不生成 `blocked=False` manifest，不提交任何 replacement classifier 候选。

下一步建议：

1. 先不要继续调低阈值；宽松诊断已经证明误报率偏高。
2. 下一轮应改进特征或 baseline 历史 meta，而不是强行提交：
   - 用旧 5117 策略在历史期生成 baseline meta，替代当前历史价差 proxy baseline。
   - 加入“候选窗口相对保底窗口的 slot 排名变化”和“候选日与历史相似日的 delta 分布”。
   - 改成只预测每天 top K 个 near-baseline 候选，而不是所有近邻窗口一起分类。
3. 只有 rolling validation 出现 `proposed_days>0` 且 `false_positive_rate` 很低、`total_delta_profit>0`，才允许重新运行 `make_single_day_candidates --allow-submission`。


### 8. ??? 5117 ?? baseline meta ????????

?????????????????????? `baseline_true_window_profit` ?? `spread_price_hist_recent_28d_slot_mean` ? proxy baseline???? 5117 ?????????????????? `true_delta_profit` ?????????????? proxy baseline ????????? 5117 ??????

???????

```text
src/safe5117_baseline.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

?????

- ?? `safe5117-source-model` baseline ???
- ??? baseline ?????????????????
- ?????????????????????????? `generate_strategy` ?????? baseline ?? meta?
- baseline meta ??????????? 5117 ?? `output.csv` ? 59 ?????
- replacement classifier ???????

```text
positive_delta_label = true_delta_profit > 0
true_delta_profit = true_window_profit - baseline_true_window_profit
```

?????

```text
outputs/output_blend_w100.csv == outputs/output_nwp_unconstrained_online5117.csv
SHA256: AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

??? `blend_w100` ??? 5117 ?????????????`outputs/output_nwp_constrained.csv` ?? 5117 ??????? 35 ????????????? baseline ??????

rolling validation ???

```text
outputs/replacement_classifier_safe5117source_oof59_forced_ms8_p040_loose_rolling_summary.csv

proba_threshold=0.40:
2025-04 proposed_days=0
2025-07 proposed_days=0
2025-10 proposed_days=0
2025-12 proposed_days=0
```

??????? stop-loss ???

```text
outputs/replacement_classifier_safe5117source_oof59_forced_ms8_p030_loose_rolling_summary.csv

2025-04 proposed_days=30, false_positive_rate=0.833333, total_delta_profit=-168919.415809
2025-07 proposed_days=31, false_positive_rate=0.645161, total_delta_profit=13689.336359
2025-10 proposed_days=31, false_positive_rate=0.645161, total_delta_profit=-91295.890595
2025-12 proposed_days=31, false_positive_rate=0.645161, total_delta_profit=-21717.003481
```

????? baseline meta ?????????????????????????????? 0.40 ??????? 0.30 ??????? 4/10/12 ????????????? `blocked=False` manifest????? replacement classifier ???

????????

```text
outputs/test_windows_replacement_classifier_safe5117source_oof59_forced_ms8_p075.csv
rows=14702
days=59
pred_rank_min=999999
eligible_rank_rows=0
```

????????

```text
python -m src.make_single_day_candidates ...
ValueError: no eligible single-day replacement candidates
```

????????????

```text
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
python -m src.check_submission --submission output.csv
rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

????????????????????????????????????????? near-baseline ????? top-K ????????????????????????????????? baseline ??? slot ?????? delta ???????????? margin ???
