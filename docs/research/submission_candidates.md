# 提交候选说明

## 2026-04-30 01:33 线上反馈更新

`outputs/output_blend_fine_w025_t1000.csv` 已提交，线上分数为：

```text
4703.505815153465
```

该分数明显低于当前最佳 `5117.832037755039`。因此融合 `w025` 系列已经被线上验证为失败候选，不再推荐继续提交。

## 当前主提交文件

```text
output.csv
```

来源：

```text
outputs/output_nwp_c0_55_d72_88.csv
```

说明：

```text
基于当前线上最好 NWP 价格预测；
不加阈值；
限制 charge_start <= 55；
限制 discharge_start >= 72；
```

本地验证：

```text
NWP 无约束 avg_profit: 13340.960251
NWP c0_55_d72_88 avg_profit: 13662.442830
```

这是当前用于冲分的中风险候选。已线上验证的稳妥版本仍保留为：

```text
outputs/output_nwp_unconstrained.csv
线上分数: 5117.832037755039
```

如果 `output.csv` 本次提交失败，应立即恢复稳妥版本：

```powershell
Copy-Item outputs\output_nwp_unconstrained.csv output.csv -Force
```

## 已验证稳妥版本

```text
outputs/output_nwp_unconstrained.csv
```

线上结果：

```text
5117.832037755039
```

这个文件是当前已经线上验证过的最高分版本。

## 新冲分候选：NWP 主线 + 窗口约束

```text
outputs/output_nwp_c0_55_d72_88.csv
```

为什么选择它：

- 仍使用线上最好 NWP 价格模型，不使用已失败的融合模型。
- 不加阈值，避免重复 `threshold=2000` 过度保守的问题。
- 2025 年 1-2 月验证集中，限制放电不早于 slot 72 后收益提升明显。
- 2025 年 1-2 月真实最优窗口里，充电开始时间没有超过 55，因此 `charge_start <= 55` 有历史依据。

风险：

- 它会改变 23 / 59 天的窗口，风险高于只改 7 天的 `c0_55`。
- 如果 2026 年隐藏集高价峰提前到 slot 68-71，这个约束会变差。

备选低风险版本：

```text
outputs/output_nwp_c0_55.csv
```

它只改变 7 / 59 天，提升空间小，但风险也低。

## 已失败候选 1：收益导向融合 + 小阈值

```text
outputs/output_blend_fine_w025_t1000.csv
```

含义：

- 预测价格 = `75% 非 NWP LightGBM + 25% NWP LightGBM`
- 策略阈值 = `1000`
- 本地验证目标 = 储能收益，不是单纯 RMSE

本地验证结果：

```text
RMSE: 0.875604
MAE: 0.532203
avg_profit: 13960.943080
loss_days: 0
```

线上结果：

```text
4703.505815153465
```

结论：

- 不再推荐提交。
- 本地验证收益过拟合。
- 窗口变化过大，偏离了当前线上最优的 NWP 无阈值策略。

风险：

- 线上数据分布可能更偏向纯 NWP。
- 本地曾经出现阈值判断和线上不一致，因此它不是替代主版本的稳妥选择。

## 不建议候选 2：收益导向融合 + 无阈值

```text
outputs/output_blend_fine_w025.csv
```

原因：

- `t1000` 版本已经明显变差，说明融合权重本身不可靠。
- 无阈值虽然少了跳过交易的问题，但仍然会大幅改变 NWP 的窗口选择。

## 低优先级小实验：NWP 主线小阈值

已生成：

```text
outputs/output_nwp_unconstrained_t500.csv
outputs/output_nwp_unconstrained_t1000.csv
```

它们和当前最佳 `outputs/output_nwp_unconstrained.csv` 的差异很小：

```text
t500:  只跳过 1 天，58 / 59 天窗口完全相同
t1000: 只跳过 3 天，56 / 59 天窗口完全相同
t2000: 跳过 5 天，54 / 59 天窗口完全相同，线上已降到 4903.50
```

建议：

- 不要优先提交。
- 如果当天有多余提交次数，并且想验证“低预测价差日是否亏损”，可以先试 `t500`。
- 由于 `t2000` 已经明显变差，阈值方向整体风险偏高。

## 不建议继续提交的版本

```text
outputs/output_nwp_unconstrained_t2000.csv
outputs/output_nwp_moderate_t2000.csv
```

原因：

- 线上已经验证 `threshold=2000` 过度保守。
- 分数约 `4903.50`，低于 NWP 无阈值的 `5117.83`。

## 下一步可做但不建议当天强行提交

1. 以 `outputs/output_nwp_unconstrained.csv` 为锚点，所有新候选都尽量少改它的窗口。
2. 建立 rolling validation，不再只依赖 2025 年 1-2 月单一验证集。
3. 训练更多 LightGBM 参数组时，只把新模型作为 NWP 的辅助信号，不直接替代窗口。
4. 用窗口稳定性指标筛选候选，例如和 NWP 主线相同窗口的天数不能太低。
5. 复赛时尝试 PatchTST / TFT，但要先建立更严格的 rolling validation。
