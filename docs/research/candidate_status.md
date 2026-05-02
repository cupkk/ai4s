# 提交候选状态表

更新日期：2026-05-02

这份文档只回答一个问题：现在应该提交哪个文件，哪些文件只是实验候选，哪些文件不能再当主线。

## 当前结论

| 文件 | 状态 | 已知线上分数 | 说明 |
|---|---:|---:|---|
| `output.csv` | 当前冲分候选 | 待线上确认 | 当前与 `outputs/output_nwp_c0_55_d72_88.csv` 字节一致。它来自 NWP LightGBM 多 seed 预测，并限制充电窗口为 `0-55`、放电窗口为 `72-88`。 |
| `outputs/output_nwp_c0_55_d72_88.csv` | 当前冲分候选 | 待线上确认 | 2025 年 1-2 月验证集中收益最高；滚动验证中也优于无约束策略。 |
| `outputs/output_nwp_unconstrained_online5117.csv` | 保底候选 | `5117.832037755039` | 旧版线上验证过的最高分文件，已单独备份，避免被 pipeline 覆盖。 |

## 当前不建议提交

| 文件 | 线上分数或本地结果 | 不建议原因 |
|---|---:|---|
| `outputs/output_blend_fine_w025_t1000.csv` | `4703.505815153465` | 已线上验证低于主线，说明该 blend/threshold 组合过拟合。 |
| `outputs/output_nwp_unconstrained_t2000.csv` | `4903.504068225546` | `threshold=2000` 会减少交易天数，线上低于无阈值主线。 |
| `outputs/output_residual_nwp.csv` | 本地验证 `avg_profit=9906.3860` | residual 单模型明显低于当前主线 `13370.6`，暂不提交。 |
| `outputs/output_window_ranker_c055_d7288.csv` | 本地验证 `avg_profit=7712.8071` | 直接窗口收益模型早停过早，收益低于主线，暂不提交。 |
| `outputs/output_nwp_robust_lambda0p25_t0.csv` | 本地验证 `avg_profit=13360.5001` | 不确定性惩罚略低于主线；可保留作实验，不占用提交次数。 |

## 本轮新增候选

| 文件 | 本地验证平均收益 | capture ratio | loss days | 结论 |
|---|---:|---:|---:|---|
| `outputs/output_nwp_window_c0_55_d72_88_t100.csv` | `13370.6122` | `0.8432` | `0` | 线下略优于 `threshold=0`，但测试集上与当前 `output.csv` 完全相同。 |
| `outputs/output_nwp_window_c0_55_d72_88_t0.csv` | `13370.5786` | `0.8432` | `1` | 与当前 `output.csv` 相同。 |
| `outputs/output_nwp_window_c0_55_d68_88_t0.csv` | `13340.3553` | `0.8266` | `0` | 收益略低，可作为备选但不优先提交。 |

## 为什么当前仍提交 `output.csv`

当前 `output.csv` 的文件哈希与以下文件一致：

```text
output.csv
outputs/output_nwp_c0_55_d72_88.csv
outputs/output_nwp_window_c0_55_d72_88_t100.csv
```

也就是说，虽然线下发现 `threshold=100` 可以少亏一天，但测试集里每天预测价差都超过 100，因此它没有改变测试期任何一天的充放电动作。

提交优先级建议：

1. 冲分：提交 `output.csv`。
2. 如果冲分低于 `5117.8320`：改提交 `outputs/output_nwp_unconstrained_online5117.csv`。
3. 不再提交 `threshold>=500`、`blend_fine_w025_t1000`、`residual_nwp`、`window_ranker` 这些已被验证排除的候选。

## 提交前检查

每次提交前运行：

```powershell
python -m src.check_submission --submission output.csv
```

当前检查结果：

```text
rows=5664
days=59
traded_days=59
errors=0
warnings=0
```
