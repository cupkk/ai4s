# 提交候选状态表

更新日期：2026-05-26

这份文档只回答一个问题：现在应该提交哪个文件，哪些文件只是实验候选，哪些文件不能再当主线。

## 当前结论

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `output.csv` | 当前推荐提交/回退锚点 | `5135.148567685195` | 已回退为 `outputs/output_stochastic_conservative_online5135_20260526.csv` 的字节级副本，SHA256=`7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C`。 |
| `outputs/output_stochastic_conservative_online5135_20260526.csv` | 当前线上最佳锚点 | `5135.148567685195` | 第四轮保守 all-seed 候选线上提升成功，必须保留；后续所有候选都必须相对它做差异审计。 |
| `outputs/output_offline_policy_shape_safe_online5135_20260526.csv` | 已排除 | `5087.6977470609945` | 保守离线 RL / 策略改进候选，`2026-02-02: 51/71 -> 50/73`，线上相对 5135 大幅回撤，已加入 guard 黑名单。 |
| `outputs/output_stochastic_seedagree_online5135_20260526.csv` | 已排除 | `5129.413866405826` | 第五轮 seed-agreement 候选，`2026-01-05: 53/67 -> 53/69`，线上低于 5135，已加入 guard 黑名单。 |
| `outputs/output_stochastic_chain2_online5124_20260525.csv` | 历史线上最佳锚点 | `5124.643279527319` | 第二轮随机场景候选，已被第四轮 `5135.148567685195` 超过，仍需保留作为回退链路。 |
| `outputs/output_stochastic_conservative_online5124_20260525.csv` | 已验证提升候选 | `5135.148567685195` | 第四轮保守 all-seed 候选源文件：`2026-01-18: 56/80 -> 55/76`。已复制固化为 `outputs/output_stochastic_conservative_online5135_20260526.csv`。 |
| `outputs/output_stochastic_pool_top1_online5124_20260525.csv` | 已排除 | `5113.038426444253` | 第三轮激进 pool top1，`2026-01-27: 52/69 -> 49/73`，双 seed、`risk_lambda=0`、低 margin，线上大幅掉分，已加入 guard 黑名单。 |
| `outputs/output_stochastic_seed_risk025_online5118_20260525.csv` | 已验证提升锚点 | `5118.064870304419` | 第一轮随机场景单日候选，`2026-01-23: 52/70 -> 51/72`。 |
| `outputs/output_nwp_unconstrained_online5117.csv` | 历史线上保底 | `5117.832037755039` | 早期安全锚点，必须保留；对应 2026-04-29 左右生成的老版本 output。 |
| `outputs/output_safe5117_skip_t500.csv` | 已排除 | `4987.610162489461` | 只跳过 2026-01-11 一天仍低于保底；说明该日不应跳过。 |
| `outputs/output_safe5117_skip_t1000.csv` | 已排除 | 未提交 | 因 `t500` 已失败，不能继续扩大 skip threshold。 |
| `outputs/output_safe5117_skip_t1500.csv` | 已排除 | 未提交 | 因 `t500` 已失败，不能继续扩大 skip threshold。 |
| `outputs/output_df_single_20260125_df_reg.csv` | 研究候选，禁止直接提交 | 未提交 | 收益导向窗口模型生成的单日替换候选，只改 2026-01-25；但 rolling validation 显示模型跨月不稳，manifest 已标记 `blocked=True`，守门脚本会拒绝。 |
| `outputs/output_window_ranker_delta_reg_full.csv` | 研究输出，禁止提交 | 未提交 | 使用相对保底收益差训练出的整套 59 天窗口输出；验证 `avg_delta_profit=-9973.036823`，且不是单日替换候选，不能提交。 |
| `outputs/output_offline_policy_online5135_20260526.csv` | 研究候选，禁止提交 | 未提交 | 初版离线策略改进候选 `2026-02-05: 56/88 -> 58/86` 虽通过基本 guard，但历史 holdout 同形态“充电后移、放电前移、间隔缩短”胜率低且均值为负，已被形态安全门排除。 |
| `outputs/offline_policy_multiprice_candidate_pool_online5135_20260526.csv` | 诊断候选池，禁止提交 | 不适用 | P1 多口径守门输出；前 80 个离线高分候选中 `passes_policy_gate=0`，所有候选的 `submission_price_delta` 均为负，因此没有生成新 submission。 |
| `outputs/offline_policy_tailrisk_candidate_pool_online5135_20260526.csv` | 诊断候选池，禁止提交 | 不适用 | P3 多口径 + tail-risk 守门输出；前 120 个离线高分候选中 `passes_policy_gate=0`、`submission_price_delta>=0 rows=0`、`shape_pass_count=0`，不生成新 submission。 |
| `outputs/test_windows_replacement_classifier_ms8_p075.csv` | 研究打分输出，禁止提交 | 未提交 | 保守替换分类器输出；严格 rolling validation 下 0 提案，测试期所有 `pred_rank=999999`，不会生成可提交单日候选。 |
| `outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505.csv` | 研究打分输出，禁止提交 | 未提交 | 更严格 `min-risk-expected-delta=0.10` 下测试期仍无可提交候选：`pred_rank=1 days=0`，`risk_expected_delta>=0.10 days=0`。 |
| `outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506.csv` | 研究打分输出，禁止提交 | 未提交 | 新增 baseline 稳定性 gate 后测试期出现 1 个研究信号：`2026-01-03` 从 `50/68` 微移到 `51/70`；但同参数 formal rolling 为 0 提案，缺少跨 fold 放行证据，不能生成 `blocked=False` manifest。 |
| `outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_summary_20260506.csv` | 诊断输出，禁止提交 | 不适用 | 只用于解释 safe5117-source-model baseline 漂移；不是 submission 文件。 |

## 已排除候选

| 文件 | 线上分数或本地结果 | 排除原因 |
|---|---:|---|
| `outputs/output_stochastic_seedagree_online5135_20260526.csv` | `5129.413866405826` | 第五轮线上失败；虽然三条 seed 回算均为正，但 `2026-01-05: 53/67 -> 53/69` 实际伤害了 5135 锚点，禁止重复提交。 |
| `outputs/output_offline_policy_shape_safe_online5135_20260526.csv` | `5087.6977470609945` | 第六轮线上失败；虽然离线 lower bound 与三 seed 均为正，但 `2026-02-02: 51/71 -> 50/73` 在真实线上评分中大幅伤害 5135 锚点，说明离线策略奖励口径和线上利润错配。 |
| `outputs/output_stochastic_pool_top1_online5124_20260525.csv` | `5113.038426444253` | 第三轮线上失败；双 seed、`risk_lambda=0`、`top1_top2_margin=0.028588`，`2026-01-27: 52/69 -> 49/73` 是场景过拟合动作，禁止重复提交。 |
| `outputs/output_nwp_c0_55_d72_88.csv` | `3798.629342284567` | 2026 测试期明显过拟合；不要再提交。 |
| `outputs/output_blend_fine_w025_t1000.csv` | `4703.505815153465` | 已线上验证低于保底。 |
| `outputs/output_nwp_unconstrained_t2000.csv` | `4903.504068225546` | 已线上验证低于保底。 |
| `outputs/output_safe5117_skip_t500.csv` | `4987.610162489461` | 已线上验证低于保底；不要再提交 skip 系列。 |
| `outputs/output_residual_nwp.csv` | 本地验证 `avg_profit=9906.3860` | 本地收益低于主线验证，不提交。 |
| `outputs/output_window_ranker_c055_d7288.csv` | 本地验证 `avg_profit=7712.8071` | 直接窗口收益模型当前未跑赢主线，不提交。 |
| `outputs/output_df_single_20260125_df_reg.csv` | 只改 1 天，guard 因 manifest blocked 拒绝 | 新收益导向窗口模型的 rolling validation 在 2025-04/07/12 月平均收益为负或接近 0，不能占用线上提交次数。 |
| `outputs/output_df_single_20260205_df_reg.csv` | 只改 1 天，manifest blocked | 同一批研究候选，模型 margin 为 0 且 score_std 较高，不提交。 |
| `outputs/output_df_single_20260115_df_reg.csv` | 只改 1 天，manifest blocked | 同一批研究候选，模型 margin 为 0 且 score_std 较高，不提交。 |
| `outputs/output_window_ranker_delta_reg_full.csv` | `avg_delta_profit=-9973.036823`，测试期预测正 delta 天数为 0 | 该文件是整套模型输出，不符合“只替换 1 天”的提交策略；当前 delta 模型也没有给出任何预测正收益单日。 |
| `outputs/output_offline_policy_online5135_20260526.csv` | holdout 形态回放不通过 | `2026-02-05: 56/88 -> 58/86` 属于“充电后移且放电前移”的间隔压缩动作；2025-02 同形态均值为负，不能提交。 |
| `outputs/test_windows_replacement_classifier_ms8_p075.csv` | 严格阈值下 0 提案；宽松阈值误报高 | 它不是 submission 文件，只是窗口打分表；当前无法生成 `blocked=False` manifest。 |

## 线上反馈修正

`outputs/output_nwp_c0_55_d72_88.csv` 曾在本地验证集中排名第一，但线上只有：

```text
3798.629342284567
```

这说明 2025 年 1-2 月验证窗口不能直接代表 2026 年 1-2 月测试期，尤其是“强制晚间 72-88 放电”的约束发生了明显过拟合。

动作差异文件：

```text
outputs/action_diff_safe5117_vs_bad3798.csv
outputs/action_diff_safe5117_vs_bad3798_summary.csv
```

关键结论：

```text
59 天中 0 天动作完全相同
平均充电起点偏移 7.37 个 15 分钟格
平均放电起点偏移 7.34 个 15 分钟格
```

## 当前最稳的下一步

1. 当前不要提交新的实验候选；`output.csv` 已回退到 `5135.148567685195` 锚点。
2. P1 多口径 reward/evaluation guard 已生效：候选必须同时通过预测场景收益、提交文件价格口径收益、历史形态回放、线上失败模式黑名单。
3. 不再提交 `2026-02-02: 51/71 -> 50/73`、`2026-01-27: 52/69 -> 49/73`、`2026-01-05: 53/67 -> 53/69`，也不提交初版离线策略 `2026-02-05: 56/88 -> 58/86`。
4. P3 tail-risk 重跑仍未找到新候选：`offline_policy_candidate=none`。后续若继续使用离线 RL，只允许先产出诊断表，不允许自动覆盖 `output.csv`。

## 2026-05-06 更新：严格 min-risk=0.10 与 baseline 漂移诊断

本轮已按更严格参数复核收益导向替换分类器：

```text
baseline_mode=safe5117-source-model
daily_top_k=10
max_shift=8
proba_threshold=0.40
risk_objective=rule
min-risk-expected-delta=0.10
```

正式 rolling 结果：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_rolling_summary_20260505.csv

fold=2025-10 proposed_days=1 false_positive_days=0 total_delta_profit=14291.49180180204
other folds proposed_days=0
overall false_positive_days=0
```

全量训练尾部验证：

```text
outputs/val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505_day_metrics.csv

proposed_days=1
date=2025-12-24
selected_true_delta_profit=535.8367742571918
false_positive_days=0
```

测试期打分：

```text
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_20260505.csv

rows=590
days=59
pred_rank=1 rows=0
risk_expected_delta>=0.10 rows=0
max_risk_expected_delta=0.0350848884796989
```

因此当前没有任何 `blocked=False` 单日 manifest，也没有新 submission 候选。

额外新增 baseline 漂移诊断脚本：

```text
src/analyze_baseline_drift.py
outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_20260506.csv
outputs/safe5117_source_baseline_drift_fold04_vs_full_tail_minrisk010_summary_20260506.csv
```

关键发现：

```text
fold04_dec_vs_full_tail overlap_days=31
changed_days=28
changed_rate=0.9032258064516129
mean_abs_charge_delta=2.6129032258064515
mean_abs_discharge_delta=5.354838709677419
large_drift_days_ge_8=8
```

通俗解释：12 月 formal rolling 和 full-tail validation 对同一天生成的“同源保底窗口”大多数并不一致，个别日期甚至漂移超过 8 个 15 分钟格。当前模型如果继续放宽阈值，很容易把“锚点变了”误判为“替换窗口更好”。所以正式结论是：

```text
submit_allowed=false
recommended_submission=output.csv
reason=min-risk=0.10 测试期仍为 0 候选，且 safe5117-source baseline 漂移仍需控制
```

## 2026-05-06 更新：baseline 稳定性 gate 已接入，但仍不放行提交

已新增并接入 baseline 稳定性 gate：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

当前 gate 逻辑：

```text
--use-baseline-stability-gate
--baseline-stability-max-abs-delta 2
```

历史 rolling / validation 阶段用 source-model baseline 与同日 proxy baseline 的差距判断锚点稳定；测试阶段用 5117 提交动作与 source-model 测试期 baseline 的差距判断锚点稳定。只有稳定性、结构规则、`risk_expected_delta>=0.10` 同时通过，才允许进入最终日内候选。

正式 rolling 结果：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v2_rolling_summary_20260506.csv

fold=2025-04 proposed_days=0
fold=2025-07 proposed_days=0
fold=2025-10 proposed_days=0
fold=2025-12 proposed_days=0
false_positive_days=0
```

全量训练尾部验证出现 1 个正向研究信号：

```text
outputs/val_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506_day_metrics.csv

date=2025-12-10
baseline=50/69
candidate=51/73
true_delta_profit=1036.2522658178368
risk_expected_delta=0.11418190279787432
```

测试期打分出现 1 个研究信号：

```text
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability2_v3_20260506.csv

date=2026-01-03
baseline=50/68
source_reference=50/68
candidate=51/70
baseline_stability_max_abs_delta=0
risk_expected_delta=0.104161
pred_rank=1
```

但这个信号仍不允许提交，原因是：

```text
formal rolling proposed_days=0
full-tail proposed_days=1
test proposed_days=1
```

通俗解释：测试期虽然出现一个很小幅度、锚点稳定的单日替换信号，但正式 rolling 没有任何同类候选被放行，说明当前 gate 可能过严，也可能只是 full/test 的偶然信号。没有跨 fold 证据前，不生成 `blocked=False` manifest，不生成提交文件，不覆盖 `output.csv`。

当前结论：

```text
submit_allowed=false
recommended_submission=output.csv
reason=baseline 稳定性 gate 已生效，但 formal rolling 无放行样本，测试期 2026-01-03 仅作研究信号
```

### baseline 稳定性阈值网格回放

已固定以下参数不变：

```text
min-risk-expected-delta=0.10
proba_threshold=0.40
risk_objective=rule
daily_top_k=10
max_shift=8
```

只回放：

```text
baseline_stability_max_abs_delta = 0, 1, 2, 4, 8
```

汇总输出：

```text
outputs/replacement_classifier_baseline_stability_grid_minrisk010_summary_20260506.csv
outputs/replacement_classifier_baseline_stability_grid_minrisk010_proposals_20260506.csv
```

结果：

| baseline_stability_max_abs_delta | proposed_days | positive_selected_days | false_positive_days | total_delta_profit | 结论 |
|---:|---:|---:|---:|---:|---|
| 0 | 0 | 0 | 0 | 0.0 | 太严，恢复不了正样本 |
| 1 | 0 | 0 | 0 | 0.0 | 太严，恢复不了正样本 |
| 2 | 0 | 0 | 0 | 0.0 | 太严，测试期研究信号仍不能提交 |
| 4 | 2 | 1 | 1 | `-2354.2077596053086` | 明确失败，12 月放出大误报 |
| 8 | 0 | 0 | 0 | 0.0 | 非单调，rule gate 重校准后无提案 |

`max_abs_delta=4` 的提案细节：

```text
2025-12-30 baseline=48/68 candidate=49/71 true_delta_profit=-3626.752494527096
2025-12-31 baseline=48/66 candidate=49/71 true_delta_profit=+1272.5447349217875
```

通俗解释：稳定性阈值开到 4 后，确实恢复了候选，但同时放出了一个大亏损日，合计收益为负。这说明不能为了恢复正样本而简单放宽 baseline stability gate。

当前结论保持：

```text
submit_allowed=false
recommended_submission=output.csv
reason=稳定性网格未找到低误报且恢复正样本的阈值；max_abs_delta=4 已被 formal rolling 证伪
```

## 收益导向窗口模型闭环状态

已新增并验证：

```text
src/train_window_ranker.py
src/make_single_day_candidates.py
src/rolling_validate_window_ranker.py
src/guard_submission_candidate.py
```

关键结果：

```text
window_ranker_df_reg 2025-01/02 验证:
avg_profit=10609.214886
capture_ratio=0.636997
regret=6045.839414
top1_window_hit=0
top3_window_hit=0
top1_minus_top2_margin=0
```

rolling validation 显示泛化不稳：

```text
2025-04: avg_profit=-454.620718, capture_ratio=-0.037848
2025-07: avg_profit=642.862610, capture_ratio=0.065094
2025-10: avg_profit=3707.872398, capture_ratio=0.266332
2025-12: avg_profit=-63.427119, capture_ratio=-0.006258
```

通俗解释：模型在 2025 年 1-2 月看起来比旧窗口模型强，但换到 4、7、12 月就不稳，说明它还没有学到稳定的“哪一天该替换保底窗口”。因此当前不能把它生成的单日候选拿去线上赌分。

## 相对保底收益差模型状态

已完成标签切换：

```text
label_mode=baseline-delta
true_delta_profit = true_window_profit - baseline_true_window_profit
```

相关输出：

```text
outputs/window_ranker_delta_reg_metadata.json
outputs/val_window_ranker_delta_reg_day_metrics.csv
outputs/test_windows_window_ranker_delta_reg.csv
outputs/window_ranker_delta_reg_rolling_summary.csv
```

结果：

```text
2025-01/02 avg_profit=308.046665
2025-01/02 avg_delta_profit=-9973.036823
2025-01/02 positive_delta_rate=0.071429

2025-04 avg_delta_profit=-7462.127327
2025-07 avg_delta_profit=-3150.002230
2025-10 avg_delta_profit=-6495.619626
2025-12 avg_delta_profit=-5113.229856

2026 测试期预测正 delta 天数=0
```

结论：这条主线的工程实现是对的，但模型结论是“不要替换保底窗口”。候选生成器用 `--min-pred-score 0.0` 时会报 `no eligible single-day replacement candidates`，这是正确的安全行为。

下一步不应提交 delta 整套输出，而应训练更保守的替换分类器：只在保底窗口附近小幅移动，并预测 `true_delta_profit > 0` 的概率。只有 rolling validation 证明误报率低，才允许重新生成 `blocked=False` 的单日 manifest。

## 保守替换分类器状态

已实现：

```text
src/replacement_classifier.py
src/train_replacement_classifier.py
src/rolling_validate_replacement_classifier.py
```

核心规则：

```text
候选窗口只允许在保底窗口附近移动，默认 max_shift=8
训练标签 positive_delta_label = true_delta_profit > 0
只有 rolling validation 低误报后，才允许 blocked=False manifest
```

严格验证：

```text
outputs/replacement_classifier_ms8_p075_rolling_summary.csv

2025-04 proposed_days=0
2025-07 proposed_days=0
2025-10 proposed_days=0
2025-12 proposed_days=0
```

宽松诊断说明不能放行：

```text
outputs/replacement_classifier_ms8_p040_loose_rolling_summary.csv

2025-07 proposed_days=31, false_positive_rate=0.354839, total_delta=15185.479929
2025-10 proposed_days=9, false_positive_rate=0.555556, total_delta=-7768.955597
```

测试期：

```text
outputs/test_windows_replacement_classifier_ms8_p075.csv
pred_rank 全部为 999999
```

候选生成器复核：

```text
ValueError: no eligible single-day replacement candidates
```

结论：保守替换分类器闭环已经存在，但当前没有可提交候选。不能为了冲分而降低阈值，因为宽松诊断已经显示误报率高。

当前 pipeline 已改为默认恢复线上保底文件。只有显式运行：

```powershell
.\scripts\run_pipeline.ps1 -UseLocalBest
```

才会用本地验证最高候选覆盖 `output.csv`。

## 提交前检查

```powershell
python -m src.check_submission --submission output.csv
```

今日候选提交前必须运行守门脚本：

```powershell
python -m src.guard_submission_candidate `
  --candidate <候选文件> `
  --candidate-name <候选名> `
  --diff-output outputs/action_diff_safe5117_vs_<候选名>.csv `
  --summary-output outputs/guard_summary_<候选名>.csv
```

如果是模型生成的单日替换候选，还必须额外传入 manifest：

```powershell
python -m src.guard_submission_candidate `
  --candidate <单日候选文件> `
  --candidate-name <候选名> `
  --manifest outputs/single_day_candidate_manifest.csv
```

只有输出 `decision=PASS` 才能提交。`t500/t1000/t1500` 当前会被守门脚本拒绝；研究候选即使只改 1 天，只要 manifest 为 `blocked=True` 也会被拒绝。

当前 `output.csv` 检查结果：

```text
rows=5664
days=59
traded_days=59
errors=0
warnings=0
```


## 5117 ?? baseline meta ?????

??????????? proxy baseline?

```text
src/safe5117_baseline.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
```

????

```text
--baseline-mode safe5117-source-model
--source-train-baseline-mode recent-oof
--source-threshold -1000000000000000000
```

??????????????? baseline ?? meta???????? baseline ?????????????? `spread_price_hist_recent_28d_slot_mean` ?? baseline proxy?????? `outputs/output_nwp_unconstrained_online5117.csv` ??????

?????

```text
outputs/output_blend_w100.csv ? 5117 ?????????SHA256 ????
outputs/output_nwp_constrained.csv ??? changed_days=35????????
```

rolling validation?

```text
outputs/replacement_classifier_safe5117source_oof59_forced_ms8_p040_loose_rolling_summary.csv
proba_threshold=0.40: ?? fold proposed_days ??? 0

outputs/replacement_classifier_safe5117source_oof59_forced_ms8_p030_loose_rolling_summary.csv
proba_threshold=0.30:
2025-04 false_positive_rate=0.833333, total_delta_profit=-168919.415809
2025-07 false_positive_rate=0.645161, total_delta_profit=13689.336359
2025-10 false_positive_rate=0.645161, total_delta_profit=-91295.890595
2025-12 false_positive_rate=0.645161, total_delta_profit=-21717.003481
```

??????

```text
outputs/test_windows_replacement_classifier_safe5117source_oof59_forced_ms8_p075.csv
pred_rank_min=999999
eligible_rank_rows=0
```

????????

```text
ValueError: no eligible single-day replacement candidates
```

????? baseline meta ????????????????????????????????? `replacement_classifier_safe5117source` ??????????????

## 2026-05-04 top-K 替换分类器状态

本轮按最新要求实现了“日级 top-K 替换候选筛选 + 更强校准特征”，但目前仍不放行线上候选。

已更新文件：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
docs/research/experiment_journal_20260504.md
```

新增逻辑：

```text
near-baseline 过滤
-> 相对保底窗口的校准特征
-> 每天只保留 top-K 个候选
-> 再训练 true_delta_profit > 0 分类器
```

严格验证结果：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p060_rolling_summary.csv

2025-04 proposed_days=0
2025-07 proposed_days=0
2025-10 proposed_days=0
2025-12 proposed_days=0
```

宽松诊断结果：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_diag_rolling_summary.csv

2025-04 proposed_days=30, false_positive_rate=0.433333, total_delta_profit=20059.409832
2025-07 proposed_days=31, false_positive_rate=0.290323, total_delta_profit=24572.005315
2025-10 proposed_days=31, false_positive_rate=0.516129, total_delta_profit=-5487.903253
2025-12 proposed_days=31, false_positive_rate=0.580645, total_delta_profit=-2508.036149
```

`daily_top_k=5` 诊断仍未解决 10 月、12 月误报问题。

阈值网格产物：

```text
outputs/replacement_classifier_topk10_safe5117source_threshold_grid.csv
outputs/replacement_classifier_topk5_safe5117source_threshold_grid.csv
```

当前结论：

```text
status=blocked
reason=rolling validation 仍存在负收益 fold 和较高 false positive rate
submit_allowed=false
```

下一步应分析 10 月、12 月误报样本，增强季节/月份校准和二阶段风险校准；不要继续降低阈值硬生成单日候选。

## 2026-05-04 二阶段风险 gate 状态

已按要求分析：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_diag_rolling_scored
```

中的 2025-10 和 2025-12 误报样本，并实现二阶段风险校准接口。

新增/更新：

```text
src/replacement_classifier.py
src/rolling_validate_replacement_classifier.py
src/train_replacement_classifier.py
docs/research/experiment_journal_20260504.md
```

关键发现：

```text
stage1 日级候选=123
true positive=67
false positive=56
```

原一阶段概率和 expected_delta 对 10/12 月日内正负例区分很弱。更有效的风险信号是：

```text
delta_vs_baseline_spread_net_load
delta_vs_baseline_spread_hist_slot_mean_daily_centered
```

也就是“候选相对保底窗口的净负荷/供需改善足够强，同时历史 slot 支持不弱”。

当前最稳的规则型二阶段 gate：

```text
delta_vs_baseline_spread_net_load >= 训练期 stage1 的 0.85 分位
delta_vs_baseline_spread_hist_slot_mean_daily_centered >= 训练期 stage1 的 0.80 分位
```

基于已保存 scored folds 的无泄漏回放：

```text
outputs/replacement_classifier_topk10_rule_gate_replay_summary.csv

fold=1 proposed_days=13, false_positive_days=1, false_positive_rate=0.076923, total_delta_profit=45353.953321
fold=2 proposed_days=2, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=4540.805471
fold=3 proposed_days=1, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=14291.491802
fold=4 proposed_days=0, false_positive_days=0, false_positive_rate=0.000000, total_delta_profit=0.000000

aggregate proposed_days=16
aggregate false_positive_days=1
aggregate total_delta_profit=64186.250594
```

但正式 rolling 命令因当前环境资源问题超时，未生成完整 summary；此前 NWP 缓存也因为缺少 `2025-01-01` 覆盖而触发重建 nc 文件并内存不足。

当前结论：

```text
status=research_only
submit_allowed=false
reason=rule gate replay 信号好，但正式 rolling 未完成，不能生成 blocked=False manifest
```

下一步：

1. 补全或修复 NWP 特征缓存覆盖。
2. 清理确认无效的残留 Python 进程，降低内存压力。
3. 重新跑正式 `--risk-objective rule` rolling。
4. 正式 rolling 达标前，不生成可提交候选，不运行 `make_single_day_candidates --allow-submission`。

## 2026-05-04 更新：缓存已补齐，但正式 rolling 仍未达标

这次已经把 `outputs/nwp_features_train.csv` 补齐到 `2025-01-01 00:00:00`，并补跑了正式 `risk-objective rule` rolling。正式 rolling 不再触发 `.nc` 重建，说明内存问题已经解决，但结果还不能放行。

正式 rolling 汇总：

```text
proposed_days_sum=8
positive_selected_days_sum=4
false_positive_days_sum=4
total_delta_profit_sum=19540.0802914541
worst_selected_delta_min=-5768.976570827298
false_positive_rate_max=0.75
false_positive_rate_mean=0.3125
```

结论保持不变：

```text
status=research_only
submit_allowed=false
reason=formal rolling 仍有较高误报与负向最差替换，不能生成 blocked=False manifest
```

当前需要保留的关键产物：

```text
outputs/nwp_features_train_original_from_20250102.csv
outputs/nwp_features_train_with_20250101.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_day_metrics_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_scored_cachefix_20260504/
```

下一步仍然不是出候选，而是继续做 2025-10/12 这类误报的二阶段校准分析；在正式 rolling 达标前，不提交任何新候选。

## 2026-05-04 更新：结构性二阶段 gate 已正式达标，但尚未放行测试期候选

已继续分析 2025-10 / 2025-12 误报 fold，并在规则型二阶段 gate 中加入结构性约束：

```text
require_both_windows_moved=true
block_both_windows_earlier=true
risk_rule_structural_pass
```

最新正式 rolling 已经复跑通过，且与无泄漏 replay 结果完全一致：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_structural_20260504.csv

fold=1 2025-04 proposed_days=1 false_positive_days=0 total_delta_profit=10535.642825
fold=2 2025-07 proposed_days=1 false_positive_days=0 total_delta_profit=2628.647939
fold=3 2025-10 proposed_days=1 false_positive_days=0 total_delta_profit=14291.491802
fold=4 2025-12 proposed_days=1 false_positive_days=0 total_delta_profit=210.676144

aggregate proposed_days=4
aggregate false_positive_days=0
aggregate total_delta_profit=27666.458709
aggregate worst_selected_delta=210.676144
```

当前状态更新为：

```text
status=validated_research_line
submit_allowed=false
reason=rolling validation 已达标，但尚未生成测试期 scored windows、manifest 和 guard PASS 候选
```

解释：模型验证这一步现在可以继续往前走了，但还没有任何新的 `blocked=False` 单日提交文件。下一步只能进入“测试期 scored windows 生成 -> 单日 manifest -> action diff -> guard”流程。`output.csv` 继续保持 5117 保底，不允许训练脚本自动覆盖。

当前可继续推进但不能直接提交的主线产物：

```text
src/replacement_classifier.py
src/train_replacement_classifier.py
src/rolling_validate_replacement_classifier.py
outputs/replacement_classifier_rulegate_structural_replay_summary_cachefix_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_summary_structural_20260504.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_rolling_scored_structural_20260504/
```

下一步若生成测试期候选，必须先满足：

```text
changed_days=1
不涉及 2026-01-11 skip
python -m src.check_submission --submission <candidate> 通过
python -m src.analyze_submission_diff --reference outputs/output_nwp_unconstrained_online5117.csv --candidate <candidate> 已记录
python -m src.guard_submission_candidate --candidate <candidate> --manifest <manifest> 输出 decision=PASS
```

## 2026-05-04 更新：测试期无可提交单日候选

已用结构性 gate 主线训练全量模型并输出测试期 scored windows：

```text
outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_structural_20260504.csv
```

测试期结果：

```text
rows=590
unique_days=59
pred_rank=1 days=0
```

结论更新为：

```text
status=no_test_candidate
submit_allowed=false
recommended_submission=output.csv
reason=正式 gate 在测试期没有放行任何单日替换候选
```

重要补充：全量训练脚本的尾部验证仍显示 2025-11/12 存在误报风险：

```text
proposed_days=8
false_positive_days=5
false_positive_rate=0.625
total_delta_profit=-574.604850
worst_selected_delta=-836.087342
```

因此不能因为 rolling structural 通过就直接放行候选。下一轮应优先重跑更严格的风险阈值：

```text
min-risk-expected-delta=0.10
```

当前所有新模型输出仍是研究产物，不是可提交文件。根目录 `output.csv` 继续是唯一建议提交入口，且 SHA256 仍应保持：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

## 2026-05-06 更新：`max_abs_delta=4` 的 12 月末误报拆解后状态

本轮没有生成新的提交文件，也没有修改 `output.csv`。

新增诊断文件：

```text
outputs/dec30_dec31_4971_profit_decomposition_20260506.csv
outputs/dec30_dec31_4971_slot_prices_20260506.csv
outputs/dec30_dec31_4971_shape_features_20260506.csv
```

核心结论：

```text
2025-12-30 baseline=48/68 candidate=49/71 delta=-3626.752495
2025-12-31 baseline=48/66 candidate=49/71 delta=+1272.544735
```

两天同样推荐 `49/71`，但收益相反。原因是：

1. `2025-12-30` 放电窗口后移错过原窗口里的 17:15 尖峰，放电贡献 `-3881.261442`，充电小幅改善无法弥补。
2. `2025-12-31` 后段是稳定高价平台，放电后移贡献 `+1272.544735`。
3. 当前二阶段 rule gate 对历史/净负荷边际信号过于乐观，没有识别“移出放电窗口含尖峰”的风险。
4. 两天一阶段模型本身都不强：`pred_expected_delta=-436.408217`，真正放行来自规则分数，因此下一步应修二阶段风险校准。

当前状态：

```text
status=blocked_by_formal_false_positive
submit_allowed=false
recommended_submission=output.csv
reason=max_abs_delta=4 放出 12 月末 49/71，但组合收益为负；正式 rolling 尚未证明测试期可安全提交
```

什么时候提交：

```text
只有当新的二阶段形态风险 gate 重跑 formal rolling 后，同时满足：
1. false_positive_days=0 或没有 material large-loss false positive
2. proposed_days>0
3. 测试期生成 blocked=False 的单日 manifest
4. candidate 相对 outputs/output_nwp_unconstrained_online5117.csv changed_days=1
5. check_submission / analyze_submission_diff / guard_submission_candidate 全部 PASS
才允许提交新候选。
否则继续提交或保留根目录 output.csv。
```

## 2026-05-06 更新：二阶段形态 gate 重跑 formal rolling 后仍不可提交

已把“移出放电窗口尖峰风险 / 新窗口平台强度”加入二阶段校准，并重跑正式 rolling。相关代码和产物：

```text
src/train_window_ranker.py
src/replacement_classifier.py
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_rolling_day_metrics_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_rolling_scored_20260506/
```

第一版 shape gate 仍失败：

```text
aggregate proposed_days=2
aggregate positive_selected_days=1
aggregate false_positive_days=1
aggregate total_delta_profit=-2036.07157587486
status=failed
reason=12 月 fold 仍有大额误报
```

加入 `discharge_shape_risk_balance >= 0` 后，误报被压住，但正式 rolling 没有任何提案：

```text
aggregate proposed_days=0
aggregate positive_selected_days=0
aggregate false_positive_days=0
aggregate total_delta_profit=0
status=blocked_zero_proposals
```

当前提交状态：

```text
submit_allowed=false
recommended_submission=output.csv
baseline_sha256=AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
candidate_manifest_generated=false
candidate_changed_days=n/a
reason=虽然没有误报，但 proposed_days=0，不满足用户要求的 proposed_days>0；不能生成 blocked=False 单日 manifest
```

保底文件状态已复核：

```text
output.csv == outputs/output_nwp_unconstrained_online5117.csv
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

后续方向：

```text
不要提交新候选。
不要为了出候选降低 min-risk-expected-delta=0.10。
下一步分析 shape-balance 挡掉的真实正样本，改进二阶段校准后再重跑 formal rolling。
只有同时满足 proposed_days>0、无大额误报、测试期 blocked=False manifest、changed_days=1、三项守门检查 PASS，才允许进入线上提交。
```

## 2026-05-06 更新：shape-balance 真实正样本诊断与 hard-pass 复跑

已按要求拆解 shape-balance 挡掉的真实正样本，没有降低 `min-risk-expected-delta=0.10`。

新增诊断文件：

```text
outputs/shape_balance_blocked_true_positive_summary_20260506.csv
outputs/shape_balance_blocked_true_positive_diagnostics_20260506.csv
outputs/shape_balance_near_miss_positive_windows_20260506.csv
outputs/shape_balance_blocked_true_positive_blocker_counts_20260506.csv
outputs/shape_balance_blocked_and_false_positive_source_terms_20260506.csv
outputs/shape_balance_vs_hardpass_stage1_day_drift_20260506.csv
outputs/shape_balance_vs_hardpass_s42_stage1_day_drift_20260506.csv
```

诊断结论：

```text
shape-balance 挡掉的日级真实正样本只有 3 个：
2025-07-25 true_delta_profit=+2.729608 blocker=shape_balance_too_low
2025-12-27 true_delta_profit=+1510.593316 blocker=shape_balance_too_low
2025-12-31 true_delta_profit=+1081.663025 blocker=min_risk_expected_delta_buffer, risk_expected_delta=0.092418
```

含义：

```text
2025-07-25 正收益太小，放行价值低。
2025-12-27 平台强度不足，shape_balance=-0.174340，仍不安全。
2025-12-31 是 near miss，但只靠它不能证明测试期安全。
```

已尝试更细的 hard-pass 版本：`shape_balance >= 0` 仍是硬门槛，但不再参与 `risk_expected_delta` 最小值打分；`min-risk-expected-delta=0.10` 保持不变。

复跑产物：

```text
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_hardpass_rolling_summary_20260506.csv
outputs/replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_shape_balance_hardpass_s42_rolling_summary_20260506.csv
```

复跑结果：

```text
hardpass multi-seed aggregate proposed_days=0 false_positive_days=0 total_delta_profit=0
hardpass seeds=42 aggregate proposed_days=0 false_positive_days=0 total_delta_profit=0
```

额外风险发现：

```text
旧 shape-balance 产物里 2025-12-27 / 2025-12-31 是日级真实正样本；
当前代码同样用 seeds=42 复跑后，一阶段日级候选集合漂移，12 月候选没有复现。
这说明当前问题不只是二阶段阈值，而是 formal rolling 重跑稳定性不足。
```

当前提交状态：

```text
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
reason=hard-pass formal rolling 仍 proposed_days=0，且一阶段候选集合复跑漂移；不满足 proposed_days>0
```

下一步：

```text
先固定 scored windows 做纯规则 replay，验证二阶段规则能否稳定区分 2025-12-30 与 2025-12-31。
再做一阶段候选稳定性 gate，例如只接受跨 seed / 跨 rerun 都出现的日级候选。
formal rolling 同时满足 proposed_days>0 且无大额误报前，不进入测试期 manifest。
```

## 2026-05-06 更新：固定 scored replay 与 P1 进入条件

新增只读规则回放工具：

```text
src/replay_scored_replacement_rules.py
```

新增产物：

```text
outputs/scored_rule_replay_detail_20260506.csv
outputs/scored_rule_replay_summary_20260506.csv
outputs/scored_rule_replay_stage1_20260506.csv
outputs/scored_rule_replay_stage1_stability_20260506.csv
outputs/scored_rule_replay_stage1_stability_summary_20260506.csv
```

固定 scored windows replay 结果：

```text
shape:
proposed_days=2
positive_selected_days=1
false_positive_days=1
total_delta_profit=-2036.071576
worst_selected_delta=-3308.616311

shape_balance:
proposed_days=0
false_positive_days=0
total_delta_profit=0

hardpass_s42:
proposed_days=0
false_positive_days=0
total_delta_profit=0
```

一阶段候选稳定性 gate：

```text
min_source_count=2
stable_days=8
stable_positive_days=3
stable_false_positive_days=5
stable_total_delta_profit=-6950.912939
stable_worst_delta_profit=-3594.938876
decision=BLOCK
reason=stable stage1 set includes false positives
```

当前 P1 判断：

```text
p1_allowed=false
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
reason=固定 replay 未恢复低误报正样本；稳定性 gate 仍含大额误报
```

什么时候进入 P1：

```text
只有同时满足以下条件，才进入 P1：
1. 固定 scored replay 中 proposed_days>0
2. 无大额误报，尤其不能放出 -3000 级别亏损日
3. 一阶段稳定性 gate 后仍有正样本，且 false_positive_days=0
4. formal rolling 重新训练后仍复现上述结果
```

P1 的具体动作不是现在做。达标后再执行：

```text
生成测试期 scored windows
生成 blocked=False 单日 manifest
生成 changed_days=1 候选
运行 check_submission
运行 analyze_submission_diff
运行 guard_submission_candidate
```

## 2026-05-06 更新：P0 保底与格式风险修复完成

基于新的高分路线规划，已完成 P0 级别任务。没有生成新候选，也没有修改根目录 `output.csv` 的内容。

保底状态：

```text
output.csv exists=True
outputs/output_nwp_unconstrained_online5117.csv exists=True
output.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256 = AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

已修复乱码价格列名输出：

```text
src/make_robust_submission.py
src/train_lgb_ranker.py
src/train_quantile_lgb.py
src/train_residual_lgb.py
src/train_trade_classifier.py
src/train_window_ranker.py
```

修复内容：

```text
乱码列名 "鐎圭偞妞傛禒閿嬬壐" -> "实时价格"
```

验证：

```text
Select-String -Path src\*.py,scripts\*.ps1,*.py -Pattern "鐎|偞|妞|閿" -Encoding UTF8
no matches
python -m compileall src
passed
```

提交状态保持：

```text
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
reason=本轮只做格式风险修复；没有 formal rolling proposed_days>0，也没有测试期 blocked=False manifest
```

注意事项：

```text
threshold_by_month、uncertainty、轻窗口约束可以继续作为研究任务。
但由于 skip / no-trade 阈值线上已证伪，不能直接用 threshold 类候选覆盖 output.csv。
任何新候选仍必须相对 outputs/output_nwp_unconstrained_online5117.csv changed_days=1，并通过 manifest + guard。
```

## 2026-05-06 状态补充：P1 暂不启动

按最新要求，已复跑固定 scored windows 的纯规则 replay，并检查一阶段候选稳定性 gate。结论没有变化：当前仍不能进入测试期 manifest 或单日候选生成。

复跑结果：

```text
shape:
proposed_days=2
positive_selected_days=1
false_positive_days=1
total_delta_profit=-2036.071576
worst_selected_delta=-3308.616311

shape_balance:
proposed_days=0
false_positive_days=0
total_delta_profit=0

hardpass_s42:
proposed_days=0
false_positive_days=0
total_delta_profit=0
```

一阶段稳定性 gate：

```text
stable_days=8
stable_positive_days=3
stable_false_positive_days=5
stable_total_delta_profit=-6950.912939
stable_worst_delta_profit=-3594.938876
decision=BLOCK
```

当前状态：

```text
p1_allowed=false
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
reason=固定 replay 与一阶段稳定性 gate 都没有同时满足 proposed_days>0 且无大额误报
```

P1 启动条件：

```text
fixed_replay.proposed_days > 0
fixed_replay.false_positive_days == 0
stage1_stability.stable_positive_days > 0
stage1_stability.stable_false_positive_days == 0
formal_rolling_retrain 复现 proposed_days>0 且 false_positive_days=0
```

## 2026-05-06 更新：postshape full train/test 产物仍禁止提交

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `outputs/test_windows_replacement_classifier_topk10_safe5117source_ms8_p040_rulegate_minrisk010_stability4_postshape_s010_p010_20260506.csv` | 研究打分输出，禁止提交 | 未提交 | 严格参数 full train/test 已跑通，但测试期 `pred_rank=1 rows=0`、`pred_rank=1 days=0`，没有可生成 `blocked=False` manifest 的单日替换。 |
| `outputs/baseline_drift_safe5117source_full_vs_rolling_fold04_20260506.csv` | 诊断输出，禁止提交 | 不适用 | 记录 rolling fold_04 与 full refit 的 source-model baseline 漂移；overlap 31 天中 28 天漂移，12 天 `max_abs_delta>4`。 |
| `outputs/baseline_drift_safe5117source_test_vs_safe5117_20260506.csv` | 诊断输出，禁止提交 | 不适用 | 记录测试期真实 5117 保底窗口与 source-model baseline 漂移；59 天中只有 15 天 `max_abs_delta<=4`，其余 44 天漂移过大。 |
| `outputs/test_postshape_failure_baseline_stable_best_by_day_20260506.csv` | 诊断输出，禁止提交 | 不适用 | 记录测试期 15 个 baseline-stable 日最接近通过 postshape 的候选；没有任何一天同时满足 spike、plateau、balance 与 structural gate。 |
| `outputs/charge_only_replay_summary_earlier_abs1_20260506.csv` | 诊断输出，禁止提交 | 不适用 | 只回放“放电窗口不动、充电提前 1 格”的微移规则；历史 rolling 只有 1 个正样本、0 误报，样本量太薄，不能作为提交依据。 |
| `outputs/test_charge_only_earlier_abs1_preview_20260506.csv` | 测试期预览，禁止提交 | 未提交 | 该规则在测试期预览 4 天，但没有 formal rolling 充足证据，也没有 manifest / changed_days=1 / guard 流程，不能提交。 |

本轮守门结果：

```text
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
p1_allowed=false
```

原因：

```text
1. 测试期没有 pred_rank=1。
2. 没有 blocked=False 单日 manifest。
3. 没有 changed_days=1 submission。
4. 后置形态门把所有 baseline-stable 的测试信号拦住，主要风险仍是“移出放电尖峰 / 新窗口平台不够强”。
```

重要诊断：

```text
2025-12-31 在 formal rolling 中：
rolling baseline = 48/66
候选 49/71 true_delta_profit = +1272.544735

2025-12-31 在 full refit 中：
full baseline = 47/80
与 rolling baseline max_abs_delta = 14
```

这说明当前 source-model baseline 在 rolling 与 full refit 之间仍有明显漂移。后续如果继续用
source-model baseline 作为训练收益差锚点，必须先解决漂移；否则模型学到的“相对保底收益差”
和测试期真实 5117 锚点不一致，误报很难压低。

下一步只允许继续研究：

```text
1. 拆 test 期 baseline_stability_pass 的 15 天。
2. 分析 source baseline 漂移来源。
3. 固定 min-risk-expected-delta=0.10，不通过降阈值硬出候选。
4. formal rolling 未同时满足 proposed_days>0 且无大额误报前，不生成提交文件。
```

测试期 postshape 诊断补充：

```text
baseline-stable days=15
spike_pass days=7
plateau_pass days=2
spike+plateau days=0
spike+plateau+balance days=0
structural+postshape days=0
```

结论：

```text
当前不是“阈值差一点”的状态。
一部分候选只是单边移动充电窗口，结构门应继续拦截；
另一部分双窗口移动候选同时存在尖峰移出风险和平台强度不足，不能为了出候选放宽 postshape。
```

charge-only 分支状态：

```text
direction=charge_earlier
max_abs_charge_delta=1
historical_replay_days=1
historical_false_positive_days=0
historical_total_delta_profit=63.627237
test_preview_days=4
submit_allowed=false
```

结论：

```text
charge-only 是值得继续 formal rolling 的研究方向；
但历史样本只有 1 天，不能直接进入测试期 manifest，更不能提交。
```

## 2026-05-07 更新：charge-only formal rolling 与 source baseline 漂移原因

本轮只做研究诊断，没有生成 submission，没有覆盖 `output.csv`。

保底状态复核：

```text
output.csv SHA256=AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256=AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

新增或更新的研究产物：

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `outputs/charge_only_formal_rule_rolling_day_metrics_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 正式 risk gate 口径 rolling 输出；4 个 fold 中只有 2025-12 fold 提出 1 天，且该样本不是 charge-only。 |
| `outputs/charge_only_formal_rule_grid_20260507.csv` | 研究诊断，禁止提交 | 不适用 | charge-only 小网格 replay；正式 `risk010` 配置下没有任何 `days>0` 的配置。 |
| `outputs/charge_only_formal_stage1_summary_earlier_abs1_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 不带 risk gate 的一阶段 replay 只有 1 个小正样本，`total_delta_profit=+63.627237`，证据太薄。 |
| `outputs/charge_only_formal_risk010_summary_earlier_abs1_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 带正式二阶段风险门后 `days=0`。 |
| `outputs/source_baseline_drift_window_shape_summary_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 解释 2025-12 大漂移日的日内价格形状变化。 |
| `outputs/source_baseline_drift_window_rankings_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 对比 rolling 与 full-oof 口径下的窗口收益排序。 |
| `outputs/source_baseline_drift_training_window_summary_20260507.csv` | 研究诊断，禁止提交 | 不适用 | 记录 rolling predict_meta 与 full recent-oof train_meta 的训练窗口差异。 |

charge-only 结论：

```text
stage1_only:
days=1
positive_days=1
false_positive_days=0
total_delta_profit=+63.627237

risk010:
days=0
positive_days=0
false_positive_days=0
total_delta_profit=0
```

说明：charge-only 有微弱信号，但正式二阶段风险门没有放行任何候选，不能进入测试期 manifest。

source baseline 漂移结论：

```text
2025-12-31 rolling baseline = 48/66
2025-12-31 full recent-oof baseline = 47/80
max_abs_delta=14
```

关键原因不是整体价格偏移，而是日内形状重排：

```text
rolling_top8=66/67/68/69/70/71/72/73
full_oof_top8=66/67/68/69/70/71/84/85
```

同时，`*_safe5117_source_full_meta.csv` 名称容易误导；代码实际保存的是 `full_source_baseline.train_meta`，在 `recent-oof` 模式下这是最后 59 天的 OOF baseline，不是真正的 full predict baseline。

本轮守门结果：

```text
submit_allowed=false
recommended_submission=output.csv
candidate_manifest_generated=false
p1_allowed=false
```

原因：

```text
1. charge-only 正式 risk010 没有候选；
2. 没有 blocked=False 测试期 manifest；
3. 没有 changed_days=1 submission；
4. source-model baseline 仍不能替代真实 5117 保底锚点。
```

下一步只允许继续研究，不允许提交新候选：

```text
1. 把 source-model baseline 改为风险/稳定性参考，而不是 delta label 的唯一锚点。
2. 用真实 5117 文件动作构造历史同源 baseline meta。
3. 增加平台并列窗口稳定性特征，避免 optimizer 在等价平台上漂到很远的放电窗口。
4. formal rolling 未同时满足 proposed_days>0、false_positive_days=0、测试期 blocked=False manifest 前，不进入提交流程。
```
## 2026-05-25 更新：随机场景单日候选进入 `output.csv`（待线上评分）

本轮按“路径 C：随机运筹优化 + 场景生成”做小步提交探针。没有重训大模型，直接使用 `outputs/test_predictions_nwp.csv` 中的三条 seed 预测作为场景：

```text
pred_price_seed42
pred_price_seed2024
pred_price_seed2026
```

新增脚本：

```text
src/stochastic_optimizer.py
```

新增候选与验证产物：

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `output.csv` | 当前待提交表格 | 待提交 | 已同步为本轮随机场景单日候选，SHA256=`14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC`。 |
| `outputs/output_stochastic_single_day_seed_risk025_20260525.csv` | 候选源文件，待线上评分 | 待提交 | 只改 `2026-01-23` 一天：`52/70 -> 51/72`，相对 5117 基线 `changed_days=1`。 |
| `outputs/stochastic_single_day_manifest_seed_risk025_20260525.csv` | guard manifest | 不适用 | manifest 匹配候选 SHA，`blocked=False`，记录 `pred_delta_score=+3968.318035`。 |
| `outputs/action_diff_safe5117_vs_output_current_20260525_summary.csv` | guard 摘要 | 不适用 | `decision=PASS`，`rows=5664`，`days=59`，`changed_days=1`，`check_errors=0`，`check_warnings=0`。 |
| `outputs/output_nwp_unconstrained_online5117.csv` | 必须保留的安全回退基线 | `5117.832037755039` | 如果本轮线上分数低于 5117，立刻恢复该文件到 `output.csv`。 |

最终校验：

```text
python -m unittest tests.test_stochastic_optimizer
OK

python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_nwp_unconstrained_online5117.csv --candidate-name output_current --manifest outputs/stochastic_single_day_manifest_seed_risk025_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-23
changed_actions:
  2026-01-23: charge=52-59;discharge=70-77 -> charge=51-58;discharge=72-79
```

提交后必须补记线上结果：

```text
if score > 5117.832037755039:
  将本候选加入可继续扩展的小步基线池
else:
  Copy-Item -LiteralPath outputs/output_nwp_unconstrained_online5117.csv -Destination output.csv -Force
  将 outputs/output_stochastic_single_day_seed_risk025_20260525.csv 加入禁止重复提交名单
```

### 2026-05-25 12:06:55 线上反馈

```text
submitted=output.csv
sha256=14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC
score=5118.064870304419
baseline_score=5117.832037755039
delta=+0.232832549380
decision=ACCEPT_AS_CURRENT_ONLINE_BEST
```

当前线上最佳锚点已保存为：

```text
outputs/output_stochastic_seed_risk025_online5118_20260525.csv
```

继续迭代约束：

```text
reference=outputs/output_stochastic_seed_risk025_online5118_20260525.csv
reference_score=5118.064870304419
blocked_dates=2026-01-11,2026-01-23
next_candidate_must_change_days_vs_reference=1
```

## 2026-05-25 更新：第二个随机场景单日候选进入 `output.csv`（待线上评分）

本轮继续路径 C，但 reference 已从原始 5117 文件切换为线上刚验证提升的 5118 锚点：

```text
reference=outputs/output_stochastic_seed_risk025_online5118_20260525.csv
reference_score=5118.064870304419
blocked_dates=2026-01-11,2026-01-23
```

新增候选：

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `output.csv` | 当前待提交表格 | 待提交 | 已同步为 chain2 候选，SHA256=`3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9`。 |
| `outputs/output_stochastic_chain2_seed_risk025_20260525.csv` | 候选源文件，待线上评分 | 待提交 | 相对 5118 锚点只改 `2026-01-22` 一天：`51/69 -> 50/68`。 |
| `outputs/stochastic_single_day_manifest_chain2_seed_risk025_20260525.csv` | guard manifest | 不适用 | manifest 匹配候选 SHA，`blocked=False`，记录 `pred_delta_score=+2369.235546`。 |
| `outputs/action_diff_online5118_vs_output_current_20260525_summary.csv` | guard 摘要 | 不适用 | `decision=PASS`，`changed_days=1`，`check_errors=0`，`check_warnings=0`。 |
| `outputs/action_diff_safe5117_vs_output_current_chain2_20260525_summary.csv` | 累计差异审计 | 不适用 | 相对原始 5117 安全基线累计 `changed_days=2`，日期为 `2026-01-22` 和 `2026-01-23`。 |

本轮新增动作：

```text
date=2026-01-22
reference_5118=charge=51-58;discharge=69-76
candidate=charge=50-57;discharge=68-75
pred_delta_score=+2369.235546
expected_delta_profit=+2512.146124
```

最终校验：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_seed_risk025_online5118_20260525.csv --reference-name online5118 --candidate-name output_current --baseline-score 5118.064870304419 --manifest outputs/stochastic_single_day_manifest_chain2_seed_risk025_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-22
changed_actions:
  2026-01-22: charge=51-58;discharge=69-76 -> charge=50-57;discharge=68-75
```

提交后必须补记线上结果：

```text
if score > 5118.064870304419:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_seed_risk025_online5118_20260525.csv -Destination output.csv -Force
  将 outputs/output_stochastic_chain2_seed_risk025_20260525.csv 加入禁止重复提交名单
```

### 2026-05-25 12:45:17 线上反馈

```text
submitted=output.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
score=5124.643279527319
previous_best=5118.064870304419
delta=+6.578409222900
decision=ACCEPT_AS_CURRENT_ONLINE_BEST
```

当前线上最佳锚点已保存为：

```text
outputs/output_stochastic_chain2_online5124_20260525.csv
```

继续迭代约束：

```text
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
blocked_dates=2026-01-11,2026-01-22,2026-01-23
next_candidate_must_change_days_vs_reference=1
```

## 2026-05-25 更新：top-K 随机场景候选池进入 `output.csv`（待线上评分）

本轮执行 A-D 后，不再只取单一 `risk_lambda=0.25/all_seed` 路径，而是批量生成候选池：

```text
src/stochastic_candidate_pool.py
```

候选池配置：

```text
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
risk_lambdas=0,0.1,0.25,0.5
max_abs_start_deltas=1,2,4
scenario_sets=all_seed + seed pairs
blocked_dates=2026-01-11,2026-01-22,2026-01-23
top_k=50
```

新增产物：

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `output.csv` | 当前待提交表格 | 待提交 | 已同步为 pool top1 候选，SHA256=`D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7`。 |
| `outputs/output_stochastic_pool_top1_online5124_20260525.csv` | 候选源文件，待线上评分 | 待提交 | 相对 5124 锚点只改 `2026-01-27` 一天：`52/69 -> 49/73`。 |
| `outputs/stochastic_candidate_pool_online5124_20260525.csv` | top-K 候选池 | 不适用 | 批量扫描风险权重、窗口位移和 seed 场景组合后的候选排行榜。 |
| `outputs/stochastic_candidate_pool_top1_manifest_online5124_20260525.csv` | guard manifest | 不适用 | manifest 匹配候选 SHA，`blocked=False`，记录 pool rank 和场景配置。 |
| `outputs/action_diff_online5124_vs_output_current_pool_top1_20260525_summary.csv` | guard 摘要 | 不适用 | `decision=PASS`，`changed_days=1`，`check_errors=0`，`check_warnings=0`。 |
| `outputs/action_diff_safe5117_vs_output_current_pool_top1_20260525_summary.csv` | 累计差异审计 | 不适用 | 相对原始 5117 安全基线累计 `changed_days=3`。 |

pool top1：

```text
date=2026-01-27
reference_5124=charge=52-59;discharge=69-76
candidate=charge=49-56;discharge=73-80
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.0
pred_delta_score=+1969.657878
top1_top2_margin=0.028588
```

风险备注：

```text
这是更激进的 pool top1。虽然 guard 通过，但 top1_top2_margin 很小，说明窗口排序接近并列。
如果线上失败，回退到 5124 锚点，并优先从 pool 中改选 all_seed 或 risk_lambda>0 的保守候选。
```

最终校验：

```text
python -m unittest tests.test_stochastic_optimizer
Ran 5 tests
OK

python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_chain2_online5124_20260525.csv --reference-name online5124 --candidate-name output_current --baseline-score 5124.643279527319 --manifest outputs/stochastic_candidate_pool_top1_manifest_online5124_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-27
changed_actions:
  2026-01-27: charge=52-59;discharge=69-76 -> charge=49-56;discharge=73-80
```

提交后必须补记线上结果：

```text
if score > 5124.643279527319:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
  将 outputs/output_stochastic_pool_top1_online5124_20260525.csv 加入禁止重复提交名单
```

### 2026-05-25 13:32:15 线上反馈

```text
submitted=output.csv
sha256=D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7
score=5113.038426444253
previous_best=5124.643279527319
delta=-11.604853083066
decision=REJECT_AND_ROLL_BACK
```

已恢复当前推荐提交为 5124 线上最佳：

```text
output.csv=outputs/output_stochastic_chain2_online5124_20260525.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
score=5124.643279527319
```

禁止重复提交：

| 文件 | 状态 | 已知线上分数 | 原因 |
|---|---:|---:|---|
| `outputs/output_stochastic_pool_top1_online5124_20260525.csv` | 禁止重复提交 | `5113.038426444253` | `2026-01-27: 52/69 -> 49/73`，双 seed、`risk_lambda=0`、`top1_top2_margin=0.028588`，线上大幅掉分。 |

后续候选池收紧条件：

```text
blocked_dates=2026-01-11,2026-01-22,2026-01-23,2026-01-27
prefer_scenario_set=all_seed
prefer_risk_lambda>0
avoid_low_top1_top2_margin
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
```
