# online feedback 2026-05-27

## 12:14:13 线上反馈

```text
submitted=outputs/output_breakeven_residual_seed_strong_max2_online5135_20260526.csv
score=5135.148567685195
previous_best=5135.148567685195
delta=0.0
decision=NEUTRAL_AND_STOP_SINGLE_DAY_BREAKEVEN
```

结论：

```text
上一轮 break-even residual+seed 探针没有带来线上增益。继续找 submission_price_delta=0 的单日微调意义很低，不能支撑 8000 目标。
```

已恢复/确认锚点：

```text
anchor=outputs/output_stochastic_conservative_online5135_20260526.csv
anchor_score=5135.148567685195
anchor_sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

## 高方差冲刺候选

用户要求加快优化速度，目标 8000。因此本轮不再继续保守单日微调，而是生成多日组合候选。

当前 `output.csv` 已切换为：

```text
candidate=outputs/output_portfolio_calendar_exact_8000push_20260527_top20.csv
candidate_sha256=9B976853E0E75A7413AAAC5C9BA74331EF5413388E8C6DD72182E8758DFEEA5A
changed_days=20
manifest=outputs/portfolio_candidate_manifest_exact_8000push_20260527.csv
decision_table=outputs/portfolio_candidate_decision_table_20260527.csv
```

候选逻辑：

```text
用 2025 同月同日真实 oracle 充放电窗口替换 2026 当前锚点中排名靠前的 20 天。
```

选择依据：

```text
hist_replay_estimated_score=8163.437509
seed_delta_min_sum=87771.096457
residual_delta_p10_sum=13097.197468
submission_price_delta_sum=-61532.777224
```

解释：

```text
这是冲 8000 的高风险候选。它不是“稳健升级”，而是利用历史同日形态假设进行大幅跳跃。若线上分数不超过 5135，立即回滚。
```

回滚命令：

```powershell
Copy-Item -LiteralPath outputs\output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

## 12:45 后续修正：从 top20 改为 target8000 top16

对 top-N 累计回放重新计算后，发现 `top16` 已经越过 8000 目标，因此不再提交漂移更大的 `top20`。

当前 `output.csv`：

```text
candidate=outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
candidate_sha256=3EF871CC46B495254E534E6734909C28A48344CF3714ECF591BD798F849DD95A
changed_days=16
hist_replay_estimated_score=8034.424874
submission_price_delta_sum=-45587.477301
manifest=outputs/portfolio_candidate_manifest_exact_target8000_20260527.csv
decision_table=outputs/portfolio_candidate_decision_table_target8000_20260527.csv
```

对比：

```text
top15: hist_replay_estimated_score=7974.958859, changed_days=15, submission_price_delta_sum=-44016.890005
top16: hist_replay_estimated_score=8034.424874, changed_days=16, submission_price_delta_sum=-45587.477301
top20: hist_replay_estimated_score=8163.437509, changed_days=20, submission_price_delta_sum=-61532.777224
```

结论：

```text
top16 是当前最小达标 8000 的候选；相比 top20，估计分略低但风险显著收窄。仍属于 high-risk jump，若线上不超过 5135.148567685195，立即回滚到 5135 锚点。
```

## 12:52:46 线上反馈：target8000 top16 失败，最后一次机会回滚

```text
submitted=outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
score=4599.494317323246
previous_best=5135.148567685195
delta=-535.654250361949
decision=FAILED_AND_ROLLBACK_TO_ONLINE_BEST
```

原因判断：

```text
2025 同月同日形态回放假设在线上失败。该候选虽然在 2025 same-day replay 中估计 8034.424874，但真实线上只得到 4599.494317323246，说明多日 exact calendar oracle 迁移存在严重年份漂移，最后一次机会不能继续试 top15/top20/cap 系列。
```

已执行：

```text
Copy-Item -LiteralPath outputs\output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

当前最终提交文件：

```text
output.csv
source=outputs/output_stochastic_conservative_online5135_20260526.csv
known_online_score=5135.148567685195
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

最终规则：

```text
只剩最后一次提交机会时，提交 output.csv，不再尝试任何新候选。
```
