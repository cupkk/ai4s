# experiment journal 20260525

## 总体研究进展

项目目标是为 AI4S 蒙西电力现货储能优化竞赛生成合法的 `output.csv` 提交表格，在已知线上 5117.832037755039 分基线之上，用尽量小的动作扰动验证是否能继续抬分。

当前主线不再追求单纯降低电价预测 MSE，而是把预测不确定性接入最终储能动作选择。历史约束仍然保留：`outputs/output_nwp_unconstrained_online5117.csv` 是已知安全基线，宽范围全量替换曾多次线上掉分，任何新候选必须先相对该基线做按日动作差异检查，并通过 `src.guard_submission_candidate`。

本轮采用方案中的路径 C：随机/场景优化。由于仓库已有多 seed NWP LightGBM 预测 `outputs/test_predictions_nwp.csv`，本轮先用 `pred_price_seed42`、`pred_price_seed2024`、`pred_price_seed2026` 作为三条价格场景，不重训大模型；优化器按“期望收益 - risk_lambda * 场景收益标准差”选择窗口，再只挑一个相对 5117 基线的小步单日替换，避免全 59 天策略漂移。

## 2026-05-25 实时更新

### 做了什么

1. 阅读现有项目结构、README、`docs/research/candidate_status.md`、`docs/research/experiment_journal_20260507.md`、`src/storage_optimizer.py`、`src/train_quantile_lgb.py`、`src/make_robust_submission.py`、`src/guard_submission_candidate.py` 等文件。
2. 确认当前 `output.csv` 原本是 5117 安全锚点，且 `python -m src.check_submission --submission output.csv` 合法。
3. 新增 `src/stochastic_optimizer.py`，实现基于多场景价格路径的储能窗口优化：
   - 自动检测场景列，优先使用 `pred_price_seed*`，其次使用 `pred_q*`。
   - 支持风险惩罚 `risk_lambda`。
   - 支持从完整随机策略中筛选单日替换候选。
   - 输出 manifest，供现有 guard 校验。
4. 新增 `tests/test_stochastic_optimizer.py` 和 `tests/__init__.py`，验证风险惩罚会从高均值高方差窗口切到更稳定窗口，并验证场景列检测优先级。
5. 用已有 NWP 多 seed 预测生成单日候选，并将 guard 通过的候选同步为根目录 `output.csv`。

### 变更文件

- `src/stochastic_optimizer.py`
- `tests/test_stochastic_optimizer.py`
- `tests/__init__.py`
- `output.csv`
- `outputs/output_stochastic_single_day_seed_risk025_20260525.csv`
- `outputs/stochastic_strategy_meta_seed_risk025_20260525.csv`
- `outputs/stochastic_single_day_manifest_seed_risk025_20260525.csv`
- `outputs/action_diff_safe5117_vs_stochastic_seed_risk025_20260525.csv`
- `outputs/action_diff_safe5117_vs_stochastic_seed_risk025_20260525_summary.csv`
- `outputs/action_diff_safe5117_vs_output_current_20260525.csv`
- `outputs/action_diff_safe5117_vs_output_current_20260525_summary.csv`

### 本轮候选

当前可提交入口已经是：

```text
output.csv
```

SHA256：

```text
14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC
```

候选只修改 1 天：

```text
date=2026-01-23
baseline=charge 52-59; discharge 70-77
candidate=charge 51-58; discharge 72-79
delta_charge_start=-1
delta_discharge_start=+2
pred_delta_score=+3968.318035
expected_delta_profit=+4239.335188
score_std=4844.785985
top1_top2_margin=8.031930
```

### 验证结果

```text
python -m unittest tests.test_stochastic_optimizer
Ran 2 tests in 0.014s
OK

python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_nwp_unconstrained_online5117.csv --candidate-name output_current --manifest outputs/stochastic_single_day_manifest_seed_risk025_20260525.csv --max-changed-days 1 --diff-output outputs/action_diff_safe5117_vs_output_current_20260525.csv --summary-output outputs/action_diff_safe5117_vs_output_current_20260525_summary.csv
decision=PASS
same_both_days=58
changed_days=1
changed_date=2026-01-23
check_errors=0
check_warnings=0
```

### 决策

本轮不采用 RL 或自定义 LightGBM loss，因为它们需要重新定义训练/回测闭环，短期内风险更高。路径 C 可以复用现有预测产物和 guard，且只做单日小步扰动，适合作为下一次线上评分探针。

### 需要保留

- 必须保留 `outputs/output_nwp_unconstrained_online5117.csv`，这是可回退的 5117 安全基线。
- 必须保留本轮 manifest 和 action diff，用于线上反馈后判断是否把 2026-01-23 候选加入允许列表或拉入黑名单。

### 下一步

1. 用户提交当前 `output.csv` 到线上系统评分。
2. 如果线上分数高于 5117.832037755039，更新 `docs/research/candidate_status.md` 和 guard allowlist，把该单日替换记录为可用候选。
3. 如果线上分数低于 5117.832037755039，立刻执行：

```powershell
Copy-Item -LiteralPath outputs/output_nwp_unconstrained_online5117.csv -Destination output.csv -Force
```

然后把 `outputs/output_stochastic_single_day_seed_risk025_20260525.csv` 记入禁止重复提交名单。

## 2026-05-25 12:06:55 线上反馈

用户提交本轮 `output.csv` 后，线上系统返回：

```text
score=5118.064870304419
previous_safe_score=5117.832037755039
delta=+0.232832549380
```

结论：`2026-01-23: 52/70 -> 51/72` 的随机场景单日替换是正向候选，可以作为新的线上最佳锚点继续小步迭代。

已将当前 `output.csv` 固化为：

```text
outputs/output_stochastic_seed_risk025_online5118_20260525.csv
sha256=14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC
```

后续规划：

```text
1. 继续路径 C：以 5118.064870304419 文件为 reference，排除已验证日期 2026-01-23 和已知禁用日期 2026-01-11。
2. 只生成“相对 5118 锚点再多改 1 天”的候选；相对原始 5117 文件会变成 changed_days=2，但相对当前线上最佳必须保持 changed_days=1。
3. 路径 B 自定义 LightGBM loss 暂不覆盖提交入口，先作为后续模型训练分支，因为当前已有能在线提升的小步策略。
```

## 2026-05-25 第二轮路径 C 候选

以新的线上最佳文件为 reference：

```text
reference=outputs/output_stochastic_seed_risk025_online5118_20260525.csv
reference_score=5118.064870304419
reference_sha256=14C5F43FCDD9E67E3B342A93A5207C33674CA03552AC7F82096E5E8326C0CFDC
blocked_dates=2026-01-11,2026-01-23
```

运行命令：

```powershell
python -m src.stochastic_optimizer `
  --price-csv outputs/test_predictions_nwp.csv `
  --reference-submission outputs/output_stochastic_seed_risk025_online5118_20260525.csv `
  --output outputs/output_stochastic_chain2_seed_risk025_20260525.csv `
  --meta-output outputs/stochastic_strategy_meta_chain2_seed_risk025_20260525.csv `
  --manifest-output outputs/stochastic_single_day_manifest_chain2_seed_risk025_20260525.csv `
  --risk-lambda 0.25 `
  --min-delta-score 0 `
  --max-abs-start-delta 4 `
  --blocked-dates 2026-01-11,2026-01-23
```

生成候选：

```text
output.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
source=outputs/output_stochastic_chain2_seed_risk025_20260525.csv
```

本轮新增改动：

```text
date=2026-01-22
reference_5118=charge 51-58; discharge 69-76
candidate=charge 50-57; discharge 68-75
delta_charge_start=-1
delta_discharge_start=-1
pred_delta_score=+2369.235546
expected_delta_profit=+2512.146124
score_std=4886.914998
top1_top2_margin=13.228136
```

相对原始 5117 安全基线累计改动：

```text
changed_days=2
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
```

验证：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_seed_risk025_online5118_20260525.csv --reference-name online5118 --candidate-name output_current --baseline-score 5118.064870304419 --manifest outputs/stochastic_single_day_manifest_chain2_seed_risk025_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-22
changed_days=1
```

提交建议：

```text
submit=output.csv
expected_reference_score=5118.064870304419
if_online_score_improves:
  固化当前 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_seed_risk025_online5118_20260525.csv -Destination output.csv -Force
```

## 2026-05-25 12:45:17 线上反馈与 A-D 执行

用户提交第二轮候选后，线上系统返回：

```text
score=5124.643279527319
previous_best=5118.064870304419
delta=+6.578409222900
```

结论：第二轮 `2026-01-22: 51/69 -> 50/68` 是强正向候选，当前链式随机场景路线仍有效。涨分慢的问题主要来自“一轮只看一个候选”，因此本轮按用户要求执行 A-D：

```text
A. 固化 5124 线上最佳锚点
B. 实现 top-K 随机场景候选池
C. 生成候选池并选一个相对 5124 的单日候选
D. 覆盖 output.csv 并验证
```

### A. 固化 5124 锚点

```text
outputs/output_stochastic_chain2_online5124_20260525.csv
sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
score=5124.643279527319
```

### B. top-K 候选池实现

新增：

```text
src/stochastic_candidate_pool.py
```

能力：

```text
1. 自动生成 all_seed 和 seed pair 场景组合。
2. 扫 risk_lambda=0,0.1,0.25,0.5。
3. 扫 max_abs_start_delta=1,2,4。
4. 排除 blocked_dates。
5. 输出 ranked candidate pool。
6. 从 pool top1 生成单日 submission 和 manifest。
```

新增测试覆盖：

```text
tests/test_stochastic_optimizer.py
```

### C. 候选池结果

运行：

```powershell
python -m src.stochastic_candidate_pool `
  --price-csv outputs/test_predictions_nwp.csv `
  --reference-submission outputs/output_stochastic_chain2_online5124_20260525.csv `
  --reference-score 5124.643279527319 `
  --output outputs/output_stochastic_pool_top1_online5124_20260525.csv `
  --pool-output outputs/stochastic_candidate_pool_online5124_20260525.csv `
  --manifest-output outputs/stochastic_candidate_pool_top1_manifest_online5124_20260525.csv `
  --risk-lambdas 0,0.1,0.25,0.5 `
  --max-abs-start-deltas 1,2,4 `
  --blocked-dates 2026-01-11,2026-01-22,2026-01-23 `
  --top-k 50
```

pool top1：

```text
date=2026-01-27
reference_5124=charge 52-59; discharge 69-76
candidate=charge 49-56; discharge 73-80
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.0
pred_delta_score=+1969.657878
expected_delta_profit=+1969.657878
top1_top2_margin=0.028588
```

注意：该候选是更激进的 pool top1，`top1_top2_margin` 很小，说明预测窗口排序接近并列。它通过 guard，但如果线上失败，应直接退回 5124 锚点，并从候选池中改选更保守的 all_seed 候选。

### D. 当前待提交表格

```text
output.csv
sha256=D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7
source=outputs/output_stochastic_pool_top1_online5124_20260525.csv
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
```

验证：

```text
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_chain2_online5124_20260525.csv --reference-name online5124 --candidate-name output_current --baseline-score 5124.643279527319 --manifest outputs/stochastic_candidate_pool_top1_manifest_online5124_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-27
changed_days=1
```

相对原始 5117 安全基线累计改动：

```text
changed_days=3
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
2026-01-27: 52/69 -> 49/73
```

提交建议：

```text
submit=output.csv
if_online_score > 5124.643279527319:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
```

## 2026-05-25 13:32:15 pool top1 失败与止损

用户提交 pool top1 后，线上系统返回：

```text
score=5113.038426444253
previous_best=5124.643279527319
delta=-11.604853083066
```

已立即回退：

```text
Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
output.csv sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0
```

失败候选：

```text
outputs/output_stochastic_pool_top1_online5124_20260525.csv
sha256=D556061BAB752A456DF034117E2813D91A64BCDC9ECDCF54A83B03A60D6079F7
date=2026-01-27
action=52/69 -> 49/73
scenario_set=seed_pair_pred_price_seed2024_pred_price_seed2026
risk_lambda=0.0
pred_delta_score=+1969.657878
top1_top2_margin=0.028588
```

根因判断：

```text
1. 候选池 rank 1 是 risk_lambda=0 的风险中性候选。
2. 只使用两个 seed，不是 all_seed 全场景一致信号。
3. top1_top2_margin 极小，说明窗口排序接近并列，optimizer 对微小预测误差敏感。
4. 线上大幅掉分证明该候选是场景过拟合，不应继续沿 2026-01-27 方向试探。
```

后续收紧策略：

```text
blocked_dates += 2026-01-27
候选优先级改为：
1. all_seed
2. risk_lambda > 0
3. top1_top2_margin 足够大
4. 单日相对 5124 锚点 changed_days=1
```

## 2026-05-25 14:40 后续：收紧候选池并生成第四轮保守候选

### 背景

第三轮 pool top1 线上从 `5124.643279527319` 掉到 `5113.038426444253`。根因不是随机场景路线整体失效，而是候选选择过于激进：失败候选只依赖两个 seed、`risk_lambda=0`，且 `top1_top2_margin=0.028588`，属于低 margin 的场景过拟合动作。

### 本轮做了什么

1. 先确认 `output.csv` 已恢复为 5124 线上最佳锚点：

```text
output.csv sha256=3AA4A21C9D9391AC22C1276720F928BC5196575DEB2FCB09199AFA46B66251A9
source=outputs/output_stochastic_chain2_online5124_20260525.csv
score=5124.643279527319
```

2. 对候选池重新筛选，只保留更保守的信号：

```text
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
blocked_dates=2026-01-11,2026-01-22,2026-01-23,2026-01-27
prefer_scenario_set=all_seed
prefer_risk_lambda>0
changed_days_vs_reference=1
```

3. 对比两个可选保守候选：

```text
2026-01-18: all_seed, risk_lambda=0.5, 56/80 -> 55/76, pred_delta_score=+233.687168
2026-01-02: all_seed, risk_lambda=0.5, 52/73 -> 54/71, pred_delta_score=+9.089281
```

4. 逐 seed 复核收益差后，选择 `2026-01-18`：

```text
2026-01-18
pred_price_seed42   delta=+213.892175
pred_price_seed2024 delta=+290.686645
pred_price_seed2026 delta=+241.202530
mean_delta=+248.593783
min_delta=+213.892175
```

`2026-01-02` 虽然动作位移更小，但一条 seed 为负，信号太弱：

```text
pred_price_seed42   delta=+2.652756
pred_price_seed2024 delta=-10.194596
pred_price_seed2026 delta=+30.079510
```

### 当前第四轮待提交表格

```text
output.csv
sha256=7A11E1D8B0D2D3ADCA8368F17E29AF7F8A7E966D13FCF2A1E2784B06A3B8A14C
source=outputs/output_stochastic_conservative_online5124_20260525.csv
reference=outputs/output_stochastic_chain2_online5124_20260525.csv
reference_score=5124.643279527319
```

相对 5124 锚点只新增 1 天动作：

```text
2026-01-18: charge=56-63;discharge=80-87 -> charge=55-62;discharge=76-83
```

相对原始 5117 锚点累计 3 天动作：

```text
2026-01-18: 56/80 -> 55/76
2026-01-22: 51/69 -> 50/68
2026-01-23: 52/70 -> 51/72
```

### 守门脚本同步收紧

已更新 `src/guard_submission_candidate.py`：

```text
BLOCKED_SINGLE_DAY_DATES += 2026-01-27
BLOCKED_CANDIDATES += outputs/output_stochastic_pool_top1_online5124_20260525.csv
```

验证失败候选会被拒绝：

```text
python -m src.guard_submission_candidate --candidate outputs/output_stochastic_pool_top1_online5124_20260525.csv ...
decision=FAIL
errors:
  candidate is blocked: online score 5113.038426444253; overfit single-day pool top1 on 2026-01-27
```

### 最终验证

```text
python -m unittest tests.test_stochastic_optimizer
Ran 6 tests
OK

python -m src.check_submission --submission output.csv
submission_check=rows=5664, days=59, traded_days=59, errors=0, warnings=0

python -m src.guard_submission_candidate --candidate output.csv --reference outputs/output_stochastic_chain2_online5124_20260525.csv --reference-name online5124 --candidate-name output_current --baseline-score 5124.643279527319 --manifest outputs/stochastic_conservative_manifest_online5124_20260525.csv --max-changed-days 1
decision=PASS
changed_date=2026-01-18
changed_days=1
```

### 提交后处理规则

```text
submit=output.csv
if online_score > 5124.643279527319:
  固化 output.csv 为新的 online-best 锚点
else:
  Copy-Item -LiteralPath outputs/output_stochastic_chain2_online5124_20260525.csv -Destination output.csv -Force
  将 outputs/output_stochastic_conservative_online5124_20260525.csv 加入禁止重复提交名单
```
