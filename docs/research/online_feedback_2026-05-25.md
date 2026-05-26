# online feedback 2026-05-25

## 12:06:55 第一轮随机场景候选

```text
submitted=output.csv
candidate=outputs/output_stochastic_single_day_seed_risk025_20260525.csv
sha256=14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC
online_score=5118.064870304419
previous_best=5117.832037755039
delta=+0.232832549380
decision=ACCEPT_AS_ONLINE_BEST
```

动作差异：

```text
2026-01-23: charge=52-59;discharge=70-77 -> charge=51-58;discharge=72-79
```

## 第二轮待提交候选

```text
submit=output.csv
candidate=outputs/output_stochastic_chain2_seed_risk025_20260525.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
reference=outputs/output_stochastic_seed_risk025_online5118_20260525.csv
reference_score=5118.064870304419
status=WAITING_ONLINE_SCORE
```

相对 5118 锚点新增动作：

```text
2026-01-22: charge=51-58;discharge=69-76 -> charge=50-57;discharge=68-75
```

相对原始 5117 锚点累计动作：

```text
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
```

如果第二轮低于 5118.064870304419，恢复：

```powershell
Copy-Item -LiteralPath outputs/output_stochastic_seed_risk025_online5118_20260525.csv -Destination output.csv -Force
```

## 12:45:17 第二轮随机场景候选

```text
submitted=output.csv
candidate=outputs/output_stochastic_chain2_seed_risk025_20260525.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
online_score=5124.643279527319
previous_best=5118.064870304419
delta=+6.578409222900
decision=ACCEPT_AS_ONLINE_BEST
```

动作差异：

```text
2026-01-22: charge=51-58;discharge=69-76 -> charge=50-57;discharge=68-75
```

## 第三轮待提交候选：top-K pool

```text
submit=output.csv
candidate=outputs/output_stochastic_pool_top1_online5124_20260525.csv
sha256=D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
status=WAITING_ONLINE_SCORE
```

本轮已从批量候选池选择 rank 1：

```text
pool=outputs/stochastic_candidate_pool_online5124_20260525.csv
date=2026-01-27
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.0
pred_delta_score=+1969.657878
top1_top2_margin=0.028588
2026-01-27: charge=52-59;discharge=69-76 -> charge=49-56;discharge=73-80
```

相对原始 5117 锚点累计动作：

```text
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
2026-01-27: 52/69 -> 49/73
```

如果第三轮低于 5124.643279527319，恢复：

```powershell
Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
```

## 13:32:15 第三轮 pool top1 失败

```text
submitted=output.csv
candidate=outputs/output_stochastic_pool_top1_online5124_20260525.csv
sha256=D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7
online_score=5113.038426444253
previous_best=5124.643279527319
delta=-11.604853083066
decision=REJECT_AND_ROLL_BACK
```

失败动作：

```text
2026-01-27: charge=52-59;discharge=69-76 -> charge=49-56;discharge=73-80
```

失败解释：

```text
pool top1 使用 scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026，risk_lambda=0.0，
且 top1_top2_margin=0.028588，属于低 margin 的激进双 seed 候选。
线上结果证明该方向过拟合，应拉黑 2026-01-27 的这组动作，并提高后续候选门槛。
```

已回退：

```text
output.csv restored from outputs/output_stochastic_chain2_online5124_20260525.csv
restored_sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
restored_score=5124.643279527319
```

## 第四轮待提交候选：保守 all-seed 风险惩罚候选

```text
submit=output.csv
candidate=outputs/output_stochastic_conservative_online5124_20260525.csv
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
status=WAITING_ONLINE_SCORE
```

本轮不继续提交激进 pool top1，而是使用收紧后的候选：

```text
date=2026-01-18
scenario_set=all_seed
risk_lambda=0.5
pred_delta_score=+233.687168
expected_delta_profit=+248.593783
top1_top2_margin=0.102853
2026-01-18: charge=56-63;discharge=80-87 -> charge=55-62;discharge=76-83
```

三条 seed 的预测收益差均为正：

```text
pred_price_seed42   delta=+213.892175
pred_price_seed2024 delta=+290.686645
pred_price_seed2026 delta=+241.202530
```

相对 5124 锚点新增动作：

```text
2026-01-18: 56/80 -> 55/76
```

相对原始 5117 锚点累计动作：

```text
2026-01-18: 56/80 -> 55/76
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
```

如果第四轮低于 `5124.643279527319`，恢复：

```powershell
Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
```
