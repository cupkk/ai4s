# online feedback 2026-05-26

## 01:02:43 第四轮保守 all-seed 候选

```text
submitted=output.csv
candidate=outputs/output_stochastic_conservative_online5124_20260525.csv
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
online_score=5135.148567685195
previous_best=5124.643279527319
delta=+10.505288157876
decision=ACCEPT_AS_ONLINE_BEST
```

动作差异：

```text
2026-01-18: charge=56-63;discharge=80-87 -> charge=55-62;discharge=76-83
```

已固化为新锚点：

```text
outputs/output_stochastic_conservative_online5135_20260526.csv
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
score=5135.148567685195
```

## 第五轮待提交候选：seed-agreement 单日候选

```text
submit=output.csv
candidate=outputs/output_stochastic_seedagree_online5135_20260526.csv
sha256=85079822DAF14909F260B44EDE2EAC6CB128581BD4A5C8620CDED7EFF69FF7C6
reference=outputs/output_stochastic_conservative_online5135_20260526.csv
reference_score=5135.148567685195
status=WAITING_ONLINE_SCORE
```

新增动作：

```text
2026-01-05: charge=53-60;discharge=67-74 -> charge=53-60;discharge=69-76
```

筛选依据：

```text
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.5
pred_delta_score=+299.127391
expected_delta_profit=+326.231832
top1_top2_margin=1.649489
all_seed_delta_min=+221.843994
all_seed_delta_positive_count=3
```

相对当前 5135 锚点只新增 1 天：

```text
changed_days=1
2026-01-05: 53/67 -> 53/69
```

相对原始 5117 锚点累计动作：

```text
2026-01-05: 53/67 -> 53/69
2026-01-18: 56/80 -> 55/76
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
```

如果第五轮低于 `5135.148567685195`，恢复：

```powershell
Copy-Item -LiteralPath outputs/output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

## 02:34:22 第五轮 seed-agreement 候选失败

```text
submitted=output.csv
candidate=outputs/output_stochastic_seedagree_online5135_20260526.csv
sha256=85079822DAF14909F260B44EDE2EAC6CB128581BD4A5C8620CDED7EFF69FF7C6
online_score=5129.413866405826
previous_best=5135.148567685195
delta=-5.734701279369
decision=REJECT_AND_ROLL_BACK
```

失败动作：

```text
2026-01-05: charge=53-60;discharge=67-74 -> charge=53-60;discharge=69-76
```

已回退：

```text
output.csv restored from outputs/output_stochastic_conservative_online5135_20260526.csv
restored_sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
restored_score=5135.148567685195
```

结论：

```text
三条 seed 同向正收益仍不足以防止线上回撤。继续用预测场景局部搜索的边际收益已经变薄，
下一步应转向保守离线 RL / 策略改进层，而不是继续按预测候选池贪心追加单日动作。
```

## 07:44:56 第六轮 offline-policy shape-safe 候选失败

```text
submitted=output.csv
candidate=outputs/output_offline_policy_shape_safe_online5135_20260526.csv
sha256=3A5B79D3D1EA8CDAD377082C40B02FA305F221159E348A6AAF66836C62A4915D
online_score=5087.6977470609945
previous_best=5135.148567685195
delta=-47.450820624201
decision=REJECT_AND_ROLL_BACK
```

失败动作：

```text
2026-02-02: charge=51-58;discharge=71-78 -> charge=50-57;discharge=73-80
```

失败前的离线证据：

```text
offline_pred_delta_lower=431.176399
pred_seed_delta_min=168.506238
pred_seed_delta_mean=242.229857
pred_seed_delta_positive_count=3
guard_changed_days=1
```

已回退：

```text
output.csv restored from outputs/output_stochastic_conservative_online5135_20260526.csv
restored_sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
restored_score=5135.148567685195
```

原因分析：

```text
1. 候选生成使用 outputs/test_predictions_nwp.csv 的三 seed 场景收益作为主奖励，但该奖励与线上真实利润发生明显错配。
2. 提交文件自身的“实时价格”列对同一动作估算为负 delta（约 -0.664），而不是 manifest 中的正收益；说明当前候选池混用了不同价格口径。
3. 2025-02 历史形态回放虽然显示“放电后移”类动作均值略正，但尾部亏损很大，单日小扰动仍可能导致线上大回撤。
4. 线上连续失败的三个动作都属于预测边际收益驱动的局部位移，说明当前阶段不能继续靠单一预测收益或三 seed 一致性追加提交。
```

后续约束：

```text
blocked_file=outputs/output_offline_policy_shape_safe_online5135_20260526.csv
blocked_date=2026-02-02
failure_pattern_table=outputs/online_failure_pattern_20260526.csv
recommended_submission=output.csv
recommended_score=5135.148567685195
next_step=先修 reward/evaluation guard，再生成候选；不要直接提交新的 offline-policy 候选
```

## P1 多口径守门执行结果

```text
guard_update=enabled
submission_price_delta_required=>=0
multi_price_delta_agree_required=true
candidate_pool=outputs/offline_policy_multiprice_candidate_pool_online5135_20260526.csv
offline_policy_candidate=none
```

重跑 5135 锚点附近离线策略候选池后：

```text
top_k_rows=80
passes_policy_gate=0
submission_price_delta>=0 rows=0
max_submission_price_delta=-0.429092
```

结论：

```text
recommended_submission=output.csv
recommended_score=5135.148567685195
submit_new_candidate=false
reason=预测高分候选在提交文件价格口径下全部为负，不能继续提交
```

## P3 tail-risk 诊断守门执行结果

```text
tail_risk_guard=enabled
shape_risk_quantile=0.10
min_shape_sample_count=2
min_shape_positive_rate=0.60
min_shape_p10_delta=0
candidate_pool=outputs/offline_policy_tailrisk_candidate_pool_online5135_20260526.csv
offline_policy_candidate=none
```

重跑结果：

```text
top_k_rows=120
passes_policy_gate=0
submission_price_delta>=0 rows=0
shape_pass_count=0
```

结论：

```text
recommended_submission=output.csv
recommended_score=5135.148567685195
submit_new_candidate=false
reason=多口径和历史 tail-risk 同时不放行，保持 5135 锚点
```
