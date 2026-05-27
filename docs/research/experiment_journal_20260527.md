# experiment journal 20260527

## 总体研究进展

项目目标仍是 AI4S 蒙西电力现货储能策略冲分。当前已知线上最好锚点为：

```text
score=5135.148567685195
anchor=outputs/output_stochastic_conservative_online5135_20260526.csv
anchor_sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

2026-05-26 的 break-even residual+seed 探针线上反馈没有超过 5135，说明单日微调已经接近失效。2026-05-27 的策略切换为“高方差组合候选”：用 2025 同月同日真实 oracle 低谷/高峰窗口作为 2026 的强先验，再用当前三 seed 预测、残差场景和提交价格口径做表格化排序。

当前阻塞点：

```text
1. 预测场景和线上真实价格仍有明显错配，提交价格口径也经常与 seed/residual 方向相反。
2. 继续单日小扰动很难接近 8000 分目标。
3. 想冲 8000 必须接受多日大位移候选的回撤风险。
```

## 2026-05-27 12:14:13 线上反馈与锚点恢复

用户反馈：

```text
score=5135.148567685195
time=2026-05-27 12:14:13
```

该分数等于 5135 锚点，说明上一轮 `outputs/output_breakeven_residual_seed_strong_max2_online5135_20260526.csv` 没有产生线上增益。已确认 `output.csv` 先回到 5135 锚点并通过格式检查：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

## 2026-05-27 高方差 portfolio 候选生成

新增代码：

```text
src/portfolio_candidate_generator.py
tests/test_portfolio_candidate_generator.py
```

更新代码：

```text
src/guard_submission_candidate.py
tests/test_guard_submission_candidate.py
```

新增能力：

```text
1. 从 2025 训练标签提取同月同日真实 oracle 窗口。
2. 将 2026 当前 5135 锚点中的动作替换为历史同日 oracle 窗口。
3. 同时生成 exact、cap8、cap4 三类 portfolio 候选。
4. 输出 action pool、candidate summary、manifest 和 decision table。
5. guard 支持 manifest_stage=portfolio_high_upside，用于结构检查和显式高风险确认。
```

排除日期：

```text
2026-01-05,2026-01-11,2026-01-18,2026-01-22,2026-01-23,2026-01-27,2026-02-02,2026-02-19
```

关键输出：

```text
outputs/portfolio_calendar_action_pool_20260527.csv
outputs/portfolio_candidate_summary_20260527.csv
outputs/portfolio_candidate_manifest_20260527.csv
outputs/portfolio_candidate_summary_exact_8000push_20260527.csv
outputs/portfolio_candidate_manifest_exact_8000push_20260527.csv
outputs/portfolio_candidate_decision_table_20260527.csv
```

## 2026-05-27 候选对比表

| candidate | changed_days | 2025同日形态回放估计分 | seed_delta_min_sum | residual_delta_p10_sum | submission_price_delta_sum | max_shift | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `portfolio_calendar_exact_8000push_20260527_top20` | 20 | `8163.437509` | `87771.096457` | `13097.197468` | `-61532.777224` | 50 | 唯一越过 8000 目标线，但风险最高 |
| `portfolio_calendar_exact_8000push_20260527_top15` | 15 | `7974.958859` | `83230.206831` | `21670.902419` | `-44016.890005` | 50 | 接近 8000，风险略低 |
| `portfolio_calendar_exact_8000push_20260527_top10` | 10 | `7680.587232` | `74562.768803` | `30669.262452` | `-38011.526280` | 50 | 更均衡但上限不足 |
| `portfolio_calendar_cap8_20260527_top15` | 15 | `6628.741063` | `42679.371182` | `12407.778000` | `-36944.037889` | 8 | 稳健备选，不适合 8000 冲刺 |
| `portfolio_calendar_cap4_20260527_top8` | 8 | `6089.384409` | `27965.799261` | `23929.022027` | `-10914.129341` | 4 | 保守备选，不适合当前目标 |

选择：

```text
promoted_to_output=outputs/output_portfolio_calendar_exact_8000push_20260527_top20.csv
output.csv_sha256=9B976853E0E75A7413AAAC5C9BA74331EF5413388E8C6DD72182E8758DFEEA5A
reason=当前明确目标是 8000；top20 是唯一按 2025 同日形态回放估计越过 8000 的候选。
```

验证：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_conservative_online5135_20260526.csv --reference-name online5135 --candidate-name output_current --baseline-score 5135.148567685195 --manifest outputs/portfolio_candidate_manifest_exact_8000push_20260527.csv --max-changed-days 20
decision=PASS
stage=portfolio_high_upside
changed_days=20
submission_price_delta=-61532.777223632685
```

重要风险：

```text
1. 这是显式 high-risk jump，不是保守候选。
2. submission_price_delta_sum 为负，说明提交文件价格口径不支持该策略。
3. 该候选依赖“2026 测试期日内形态接近 2025 同月同日”的假设。
4. 如果线上分数未超过 5135，应立即回滚。
```

回滚命令：

```powershell
Copy-Item -LiteralPath outputs\output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

## 2026-05-27 target8000 top16 收敛与替换

本轮继续加速冲 8000，但没有继续扩大到 top20，而是修复了候选生成脚本的一个决策缺口：之前只生成 top10/top15/top20，因此误判为“只有 top20 能越过 8000”。重新计算 top-N 累计 2025 同月同日回放后，发现 top16 已经达到目标。

代码更新：
```text
src/portfolio_candidate_generator.py
tests/test_portfolio_candidate_generator.py
```

新增能力：
```text
1. add_calendar_replay_diagnostics：把 2025 同月同日真实价格下的 baseline/candidate 利润差写入 action pool。
2. expand_top_counts_for_target：当设置 --target-score 8000 时，自动补齐最小达标 top-N 及其相邻 N。
3. summary/manifest 现在直接记录 hist_2025_same_day_delta_sum 与 hist_replay_estimated_score，避免后续靠临时表推断。
```

重新生成的关键文件：
```text
outputs/portfolio_calendar_exact_target8000_action_pool_20260527.csv
outputs/portfolio_candidate_summary_exact_target8000_20260527.csv
outputs/portfolio_candidate_manifest_exact_target8000_20260527.csv
outputs/portfolio_candidate_summary_cap8_target8000_20260527.csv
outputs/portfolio_candidate_summary_cap4_target8000_20260527.csv
outputs/portfolio_candidate_decision_table_target8000_20260527.csv
outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
```

最新候选对比：
| candidate | changed_days | 2025同日回放估计分 | submission_price_delta_sum | 结论 |
|---|---:|---:|---:|---|
| `portfolio_calendar_exact_target8000_20260527_top16` | 16 | `8034.424874` | `-45587.477301` | 当前推荐提交；最小越过 8000 |
| `portfolio_calendar_exact_target8000_20260527_top20` | 20 | `8163.437509` | `-61532.777224` | 更激进，漂移更大 |
| `portfolio_calendar_exact_target8000_20260527_top15` | 15 | `7974.958859` | `-44016.890005` | 更稳但未达 8000 |

已执行替换：
```text
output.csv <- outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
output.csv_sha256=3EF871CC46B495254E534E6734909C28A48344CF3714ECF591BD798F849DD95A
```

验证结果：
```text
python -m unittest tests.test_portfolio_candidate_generator
Ran 5 tests, OK

python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_conservative_online5135_20260526.csv --reference-name online5135 --candidate-name output_current --baseline-score 5135.148567685195 --manifest outputs/portfolio_candidate_manifest_exact_target8000_20260527.csv --max-changed-days 16
decision=PASS
stage=portfolio_high_upside
changed_days=16
submission_price_delta=-45587.4773010005
```

决策：
```text
recommended_submission=output.csv
source=outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
rollback_anchor=outputs/output_stochastic_conservative_online5135_20260526.csv
rollback_command=Copy-Item -LiteralPath outputs\output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

## 2026-05-27 12:52:46 target8000 top16 线上失败与最终回滚

用户反馈：
```text
score=4599.494317323246
time=2026-05-27 12:52:46
remaining_submission_chance=1
```

对应候选：
```text
candidate=outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv
candidate_sha256=3EF871CC46B495254E534E6734909C28A48344CF3714ECF591BD798F849DD95A
hist_replay_estimated_score=8034.424874
online_score=4599.494317323246
delta_vs_5135=-535.654250361949
```

结论：
```text
2025 同月同日 exact calendar oracle 的跨年迁移失败。最后一次机会不能再尝试 top15/top20/cap4/cap8 或新的高方差组合；唯一合理动作是回滚到已知线上最高 5135 锚点。
```

已执行回滚：
```powershell
Copy-Item -LiteralPath outputs\output_stochastic_conservative_online5135_20260526.csv -Destination output.csv -Force
```

当前最终提交文件：
```text
output.csv
source=outputs/output_stochastic_conservative_online5135_20260526.csv
known_online_score=5135.148567685195
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
```

守门更新：
```text
src/guard_submission_candidate.py 已加入：
- outputs/output_stochastic_conservative_online5135_20260526.csv allowlist
- outputs/output_portfolio_calendar_exact_target8000_20260527_top16.csv blacklist
- outputs/output_portfolio_calendar_exact_8000push_20260527_top20.csv blacklist
```

最终建议：
```text
submit=output.csv
do_not_experiment=true
```
