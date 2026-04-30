# 提交候选状态表

更新日期：2026-04-30

这份文档只回答一个问题：现在到底该提交哪个文件，哪些文件只是实验候选。

## 当前主线与当前文件状态

| 文件 | 状态 | 线上分数 | 说明 |
|---|---:|---:|---|
| `outputs/output_nwp_unconstrained.csv` | 已验证主线 | `5117.832037755039` | 目前线上最高的已验证结果，使用 NWP LightGBM 预测 + 每天一次充放电，`threshold=0`。 |
| `output.csv` | 当前强候选 | 待以最新提交记录为准 | 当前仓库里的 `output.csv` 与 `outputs/output_nwp_c0_55_d72_88.csv` 一致，不是 `output_nwp_unconstrained.csv` 的字节级副本。 |
| `outputs/output_nwp_c0_55_d72_88.csv` | 强候选 | 待线上确认 | 旧验证窗口和滚动验证都优于无约束版本；滚动验证平均收益 `5783.5865`，无约束为 `5045.5199`。 |

## 不能再当主线的版本

| 文件 | 线上分数 | 问题 |
|---|---:|---|
| `outputs/output_blend_fine_w025_t1000.csv` | `4703.505815153465` | 本地验证收益高，但线上明显下降，说明季节先验混合或阈值在公开评分区过拟合。 |
| `outputs/output_nwp_unconstrained_t2000.csv` | `4903.504068225546` | `threshold=2000` 会少交易若干天，线上收益低于无阈值主线。 |
| `outputs/output_nwp_moderate_t2000.csv` | 未作为主线 | 同样受 `threshold=2000` 影响，除非滚动验证证明有效，否则不提交。 |

## 待验证候选

| 文件 | 本地依据 | 风险 |
|---|---|---|
| `outputs/output_nwp_c0_55_d72_88.csv` | 2025 年 1-2 月验证收益略高于无约束版本。 | 只在单一验证窗口上更好，可能对 2026 年测试期过拟合。 |
| `outputs/output_nwp_c0_55.csv` | 限制充电窗口，保留放电自由度。 | 比完整窗口约束更保守，但仍需滚动验证。 |
| `outputs/output_nwp_bias.csv` | 加入预测偏差类特征。 | 尚无线上结果，不能直接替换主线。 |

## 当前提交规则

1. 若要保底，提交 `outputs/output_nwp_unconstrained.csv` 或先把它复制为 `output.csv`。
2. 若要冲分，提交当前 `output.csv`，它等同于 `outputs/output_nwp_c0_55_d72_88.csv`。
3. 每次生成新候选后，先运行 `python -m src.check_submission --submission <文件>`。
4. 每个候选必须进入 `outputs/strategy_compare.csv`，至少包含本地平均收益、oracle 捕获率、regret、窗口命中率。
5. 线上失败的文件不要覆盖 `output.csv`，只能留在 `outputs/` 里做复盘。

## 为什么这样做

比赛分数不是单纯看 RMSE，而是看充放电动作在真实价格上的收益。之前出现过“本地验证更高、线上更低”的情况，所以现在主线只认线上验证过的文件；新想法必须先进入统一比较表，再决定是否占用每天有限的提交次数。
