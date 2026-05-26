# experiment journal 20260526

## 总体研究进展

项目当前主线是围绕已验证线上最佳表格做单日小扰动，而不是重新生成全 59 天策略。2026-05-26 之前的最新已知线上最佳为 `5124.643279527319`，来源于随机场景链式候选；第三轮激进 pool top1 曾降到 `5113.038426444253`，因此后续候选必须加强场景一致性和守门。

## 2026-05-26 实时更新

### 01:02:43 第四轮线上反馈

用户反馈第四轮保守 all-seed 候选线上得分：

```text
score=5135.148567685195
previous_best=5124.643279527319
delta=+10.505288157876
```

结论：`2026-01-18: 56/80 -> 55/76` 是有效正向动作。已将当前 `output.csv` 固化为新的线上最佳锚点：

```text
outputs/output_stochastic_conservative_online5135_20260526.csv
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
score=5135.148567685195
```

### 候选池脚本升级

第三轮失败说明只看候选生成时使用的部分 seed 不够稳。本轮更新 `src/stochastic_candidate_pool.py`，新增逐 seed 回算诊断：

```text
pred_price_seed42_delta
pred_price_seed2024_delta
pred_price_seed2026_delta
all_seed_delta_min
all_seed_delta_mean
all_seed_delta_positive_count
```

新增过滤参数：

```text
--min-all-seed-delta
--min-all-seed-positive-count
```

对应新增单测：

```text
tests/test_stochastic_optimizer.py
```

验证：

```text
python -m unittest tests.test_stochastic_optimizer
Ran 7 tests
OK
```

### 第五轮候选生成

以新的 5135 锚点为 reference，排除已经验证或失败的日期：

```text
reference=outputs/output_stochastic_conservative_online5135_20260526.csv
reference_score=5135.148567685195
blocked_dates=2026-01-11,2026-01-18,2026-01-22,2026-01-23,2026-01-27
```

保守 all-seed 只剩 `2026-01-02`，但信号弱且一条 seed 为负，因此没有采用：

```text
2026-01-02: 52/73 -> 54/71
pred_price_seed42   delta=+2.652756
pred_price_seed2024 delta=-10.194596
pred_price_seed2026 delta=+30.079510
```

最终采用逐 seed 一致性过滤后的候选：

```text
outputs/output_stochastic_seedagree_online5135_20260526.csv
sha256=85079822DAF14909F260B44EDE2EAC6CB128581BD4A5C8620CDED7EFF69FF7C6
date=2026-01-05
reference=charge=53-60;discharge=67-74
candidate=charge=53-60;discharge=69-76
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.5
pred_delta_score=+299.127391
expected_delta_profit=+326.231832
top1_top2_margin=1.649489
all_seed_delta_min=+221.843994
all_seed_delta_mean=+291.435886
all_seed_delta_positive_count=3
```

当前 `output.csv` 已同步为第五轮候选。

### 最终验证

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_conservative_online5135_20260526.csv --reference-name online5135 --candidate-name output_current --baseline-score 5135.148567685195 --manifest outputs/stochastic_seedagree_manifest_online5135_20260526.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-05
changed_days=1
```

### 提交后处理规则

```text
submit=output.csv
if online_score > 5135.148567685195:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
  将 outputs/output_stochastic_seedagree_online5135_20260526.csv 加入禁止重复提交名单
```

## 2026-05-26 02:34:22 第五轮失败与强化学习路线判断

用户反馈第五轮线上得分：

```text
score=5129.413866405826
previous_best=5135.148567685195
delta=-5.734701279369
```

失败候选：

```text
outputs/output_stochastic_seedagree_online5135_20260526.csv
sha256=85079822DAF14909F260B44EDE2EAC6CB128581BD4A5C8620CDED7EFF69FF7C6
date=2026-01-05
action=53/67 -> 53/69
```

已回退：

```text
Copy-Item -LiteralPath outputs/output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
output.csv sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

已更新守门脚本：

```text
BLOCKED_CANDIDATES += outputs/output_stochastic_seedagree_online5135_20260526.csv
```

### 对强化学习路线的判断

不建议直接上在线 PPO/SAC 端到端替换提交，原因是 2026 测试期真价格不可见，线上提交次数极少，无法形成可交互环境。用预测价格当环境训练出的 PPO/SAC 会继续学习预测器偏差，风险不低于当前随机场景候选池。

建议切换为保守离线 RL / 策略改进层：

```text
1. 用 2025 真实价格构建状态-动作-收益数据。
2. 动作空间限制为当前锚点附近的充放电窗口小位移，而不是全量 59 天自由策略。
3. 学习 Q(s,a) / advantage(s,a)，目标是相对 baseline 的 delta profit。
4. 在 2026 上只生成候选池，不直接覆盖 output.csv。
5. 候选必须经过反事实风险过滤、历史滚动验证、guard 单日差异检查后才允许提交。
```

当前最佳线上锚点仍为：

```text
outputs/output_stochastic_conservative_online5135_20260526.csv
score=5135.148567685195
```

## 2026-05-26 离线 RL / 策略改进候选落地

本轮没有直接使用在线 PPO/SAC 端到端生成 59 天策略，而是实现了保守离线 RL / 策略改进层：

```text
src/offline_policy_improvement.py
tests/test_offline_policy_improvement.py
```

核心思路：

```text
1. 用 2025 已知真实价格生成 baseline 附近的小位移动作。
2. 训练 LightGBM ensemble 近似 Q(s,a) / advantage(s,a)，标签为 true_delta_profit。
3. 在 2026 只生成相对当前 online-best 的单日小扰动候选。
4. 用 offline lower bound、三 seed 正收益、动作形态安全门和 guard 脚本共同过滤。
```

新增验证与守门：

```text
python -m unittest tests.test_offline_policy_improvement tests.test_stochastic_optimizer
Ran 12 tests
OK

python -m src.offline_policy_improvement ... --validation-split-date 2025-02-01 ...
offline_policy_validation=none
```

解释：按“离线下界 + 三 seed 全正”的严格验证口径，2025-02 holdout 没有放行候选。这说明不能把初版离线策略候选直接当成已验证 RL 结果。

初版候选被排除：

```text
candidate=outputs/output_offline_policy_online5135_20260526.csv
date=2026-02-05
action=56/88 -> 58/86
reason=充电后移、放电前移、间隔缩短；2025-02 同形态历史均值为负，胜率约 22%
decision=DO_NOT_SUBMIT
```

随后加入 shape-safe 门：

```text
--min-delta-gap-slots 0
--forbid-charge-later-discharge-earlier
```

最终生成的候选：

```text
outputs/output_offline_policy_shape_safe_online5135_20260526.csv
sha256=3A5B79D3D1EA8CDAD377082C40B02FA305F221159E348A6AAF66836C62A4915D
date=2026-02-02
baseline=charge=51-58;discharge=71-78
candidate=charge=50-57;discharge=73-80
offline_pred_delta_lower=431.176399
pred_seed_delta_min=168.506238
pred_seed_delta_mean=242.229857
pred_seed_delta_positive_count=3
```

最终校验：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_conservative_online5135_20260526.csv --reference-name online5135 --candidate-name output_current --baseline-score 5135.148567685195 --manifest outputs/offline_policy_shape_safe_manifest_online5135_20260526.csv --max-changed-days 1
decision=PASS
changed_date=2026-02-02
changed_days=1
changed_actions:
  2026-02-02: charge=51-58;discharge=71-78 -> charge=50-57;discharge=73-80
```

当前 `output.csv` 已同步为该 shape-safe 候选，等待线上评分。

提交后处理规则：

```text
submit=output.csv
if online_score > 5135.148567685195:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
  将 outputs/output_offline_policy_shape_safe_online5135_20260526.csv 加入禁止重复提交名单
```

## 2026-05-26 07:44:56 第六轮失败复盘与下一阶段计划

用户反馈第六轮线上得分：

```text
score=5087.6977470609945
previous_best=5135.148567685195
delta=-47.450820624201
candidate=outputs/output_offline_policy_shape_safe_online5135_20260526.csv
sha256=3A5B79D3D1EA8CDAD377082C40B02FA305F221159E348A6AAF66836C62A4915D
```

失败动作：

```text
date=2026-02-02
baseline=charge=51-58;discharge=71-78
candidate=charge=50-57;discharge=73-80
```

已执行回退：

```text
Copy-Item -LiteralPath outputs/output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
output.csv sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

已更新守门脚本：

```text
BLOCKED_SINGLE_DAY_DATES += 2026-02-02
BLOCKED_CANDIDATES += outputs/output_offline_policy_shape_safe_online5135_20260526.csv
```

原因分析：

```text
1. 这次失败不是格式问题，check_submission 和 guard 都通过；失败来自收益评估口径错配。
2. offline_policy_improvement 用 outputs/test_predictions_nwp.csv 的三 seed 场景收益训练/筛选，manifest 显示 pred_seed_delta_min=+168.506238。
3. 但提交文件自身“实时价格”列估算同一动作 delta 约为 -0.664，和 manifest 的正收益方向相反。
4. 线上反馈进一步证明该动作真实收益为负，而且回撤幅度明显大于前两次失败；说明当前离线 RL 只是学到了预测器/价格口径偏差，不是可直接提交的利润策略。
5. 2025-02 holdout 形态统计对“放电后移”只给出弱正均值，尾部亏损极大；这个过滤不足以承担线上提交风险。
```

下一阶段计划：

| 阶段 | 任务 | 验收标准 | 输出 |
|---|---|---|---|
| P0 | 保底与黑名单同步 | `output.csv` 字节级等于 5135 锚点；5087 候选和 `2026-02-02` 被 guard 拒绝 | 当前已完成 |
| P1 | 修复奖励口径 | 候选池同时计算 `test_predictions_nwp` 场景收益和 submission price-column 收益；两者方向不一致时自动 blocked | 新增多口径收益列和单测 |
| P2 | 失败模式归因 | 汇总线上失败日 `2026-01-05/01-27/02-02` 的动作形态、预测收益、提交价格收益、历史形态尾部风险 | 当前已生成 `outputs/online_failure_pattern_20260526.csv` |
| P3 | 只生成诊断候选 | 离线 RL 先只输出候选池，不覆盖 `output.csv`；必须满足多口径均为正、历史 tail 不差、且不命中失败形态 | blocked/allowed manifest |
| P4 | 重新选择下一次提交 | 如果没有候选同时通过 P1-P3，则保持 5135 不提交；如果有，只允许单日小扰动并先给出表格复核 | 新 `output.csv` 或 no-submit 结论 |

当前结论：

```text
recommended_submission=output.csv
recommended_score=5135.148567685195
submit_new_candidate=false
reason=需要先修 reward/evaluation guard，不能继续用预测场景收益直接驱动提交
```

## 2026-05-26 P1 多口径 reward/evaluation guard 执行结果

本轮按上一节计划执行 P1：把离线策略候选从“只看预测场景收益”升级为“预测场景收益 + 提交文件价格列收益”双口径守门。

代码改动：

```text
src/offline_policy_improvement.py
tests/test_offline_policy_improvement.py
src/guard_submission_candidate.py
tests/test_guard_submission_candidate.py
```

新增字段：

```text
submission_price_baseline_profit
submission_price_candidate_profit
submission_price_delta
multi_price_delta_min
multi_price_delta_agree
passes_policy_gate
```

守门规则：

```text
min_submission_price_delta=0.0
预测场景收益为正但 submission_price_delta < 0 的候选必须 blocked
manifest 中若 multi_price_delta_agree=false，也必须被 guard 拒绝
```

回归验证：

```text
python -m unittest tests.test_offline_policy_improvement tests.test_guard_submission_candidate tests.test_stochastic_optimizer
Ran 15 tests
OK
```

用 5135 锚点重新生成候选池：

```text
python -m src.offline_policy_improvement ... --submission-price-col 实时价格 --min-submission-price-delta 0
offline_policy_candidate=none
train_rows=4111
test_rows=1148
pool=outputs/offline_policy_multiprice_candidate_pool_online5135_20260526.csv
```

诊断结论：

```text
top_k_rows=80
passes_policy_gate=0
submission_price_delta >= 0 rows=0
max_submission_price_delta=-0.429092
```

这说明上一轮失败不是孤例：在当前 5135 锚点附近，离线模型/预测场景认为高分的一批动作，在提交文件价格口径下全部为负收益。继续提交会高概率消耗评分机会。

当前执行结论：

```text
recommended_submission=output.csv
recommended_score=5135.148567685195
new_candidate_generated=false
reason=P1 多口径守门没有放行任何候选
```

## 2026-05-26 P3 tail-risk 诊断守门执行结果

本轮继续执行 P3：在 P1 多口径收益一致性之外，加入历史同形态动作的尾部风险统计。

新增字段：

```text
shape_sample_count
shape_true_delta_min
shape_true_delta_mean
shape_true_delta_p10
shape_positive_rate
```

新增守门参数：

```text
--shape-risk-quantile 0.10
--min-shape-sample-count 2
--min-shape-positive-rate 0.60
--min-shape-p10-delta 0
--diagnostic-only
```

新增/更新验证：

```text
python -m unittest tests.test_offline_policy_improvement
Ran 7 tests
OK
```

用 5135 锚点重新生成 tail-risk 诊断候选池：

```text
python -m src.offline_policy_improvement ... --diagnostic-only --min-submission-price-delta 0 --min-shape-positive-rate 0.60 --min-shape-p10-delta 0
offline_policy_candidate=none
train_rows=4111
test_rows=1148
pool=outputs/offline_policy_tailrisk_candidate_pool_online5135_20260526.csv
```

诊断统计：

```text
top_k_rows=120
passes_policy_gate=0
submission_price_delta >= 0 rows=0
shape_pass_count=0
```

解释：

```text
1. 多口径收益门已经把全部候选挡下：提交文件价格口径下没有一个候选为正。
2. 即使不看提交价格，历史同形态 tail-risk 也很差：前排候选的 shape_true_delta_p10 大多显著为负。
3. 因此当前不是“再调阈值找一个候选”的阶段，而是应暂停提交，回到 5135 锚点。
```

当前执行结论：

```text
recommended_submission=output.csv
recommended_score=5135.148567685195
new_candidate_generated=false
reason=P3 多口径 + tail-risk 守门没有放行任何候选
```
