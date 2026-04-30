# 实验日志与线上反馈

## 已知线上提交结果

| 文件 | 提交时间 | 线上分数 | 结论 |
| --- | --- | ---: | --- |
| `output_unconstrained_score4796.csv` | 2026-04-29 17:19:51 | `4796.899814419311` | 早期无 NWP 版本，能跑通但上限不足。 |
| `output.csv` 旧版本 | 2026-04-29 17:49:39 | `4564.856620223643` | 季节模板混合过强，线上变差。 |
| `output.csv` 旧版本 | 2026-04-29 18:49:53 | `4903.504068225546` | NWP + `threshold=2000`，过度保守，错过收益。 |
| `outputs/output_nwp_unconstrained.csv` | 2026-04-30 00:49:49 | `5117.832037755039` | 当前线上最好版本，NWP 有效，无阈值优于高阈值。 |
| `outputs/output_blend_fine_w025_t1000.csv` | 2026-04-30 01:33:46 | `4703.505815153465` | 本地收益导向融合过拟合，线上明显低于纯 NWP，不再推荐。 |

## 本地验证区间

为了模拟官方测试集的 2026 年 1-2 月，本地验证固定使用：

```text
2025-01-01 到 2025-02-28
```

收益回测自动跳过标签数据中不完整的日子。

## 关键本地实验

### 非 NWP LightGBM

```text
RMSE: 0.881363
MAE: 0.532830
收益回测最优阈值: 约 1000
```

特征重要性靠前：

- `supply_margin`
- `load_minus_windsolar_total`
- `wind_ratio`
- `net_load`
- `dayofyear`
- `hist_month_slot_mean`

说明供需平衡特征比单纯时间特征更关键。

### NWP LightGBM

```text
RMSE: 0.878636
MAE: 0.547152
本地收益回测 threshold=2000 略好
线上 threshold=0 明显更好
```

这里出现了一个重要复盘：本地验证建议 `threshold=2000`，但线上 `threshold=0` 更好。这说明验证集和测试集存在分布差异，策略不能过度保守。

### NWP / 非 NWP 收益导向融合：已线上证伪

脚本：

```powershell
python -m src.tune_prediction_blend `
  --val-first outputs\val_predictions.csv `
  --val-second outputs\val_predictions_nwp.csv `
  --test-first outputs\test_predictions.csv `
  --test-second outputs\test_predictions_nwp.csv `
  --weights 0.2,0.25,0.3,0.35 `
  --threshold-grid 0,500,1000,1500,2000,3000,5000 `
  --emit-weights 0.25
```

本地最优：

```text
weight_second: 0.25
含义: 75% 非 NWP + 25% NWP
RMSE: 0.875604
MAE: 0.532203
best_threshold: 1000
avg_profit: 13960.943080
loss_days: 0
```

生成候选：

```text
outputs/output_blend_fine_w025.csv
outputs/output_blend_fine_w025_t1000.csv
```

线上反馈显示 `outputs/output_blend_fine_w025_t1000.csv` 只有 `4703.505815153465`，低于当前最佳的 `5117.832037755039`。这说明该融合权重虽然在 2025 年 1-2 月验证集上收益最高，但没有泛化到 2026 年 1-2 月隐藏测试集。

复盘后发现，融合候选和纯 NWP 版本只有 `7 / 59` 天的充放电窗口完全相同，窗口改动幅度太大。后续不能再把它作为推荐提交。

## 当前文件状态

根目录 `output.csv` 已恢复为当前线上最高分版本：

```text
output.csv == outputs/output_nwp_unconstrained.csv
```

校验哈希一致：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

## 提交建议

如果只剩 1 次提交：

```text
提交根目录 output.csv
```

如果还有额外提交次数，也不要继续提交融合 `w025` 系列。后续候选必须以 NWP 无阈值版本为锚点，尽量少改窗口。

已额外生成低优先级小实验：

```text
outputs/output_nwp_unconstrained_t500.csv
outputs/output_nwp_unconstrained_t1000.csv
```

其中 `t500` 只比当前最佳少交易 1 天，`t1000` 少交易 3 天。由于 `t2000` 已经线上变差，这两个文件只能作为额外提交次数充足时的小实验，不作为主推荐。

## 2026-04-30 02:20 后的新一轮优化

重新阅读评分规则后，确认最终分数只由真实电价和 `power` 决定，因此优化重点转向策略窗口本身。

### 直接窗口收益模型

尝试训练“候选窗口收益模型”：枚举每天全部合法 `(charge_start, discharge_start)`，用边界条件和 NWP 特征直接预测该窗口真实收益。

验证结果：

```text
direct window model avg_profit: 11617.062476
NWP baseline avg_profit:        13340.960251
```

结论：该方向本地不如 NWP，不进入提交候选。

### NWP 价格模型 + 窗口约束搜索

在 `outputs/val_predictions_nwp.csv` 上搜索窗口约束，保持 `threshold=0`。

最优约束：

```text
charge_start_min=0
charge_start_max=55
discharge_start_min=72
discharge_start_max=88
avg_profit=13662.442830
loss_days=0
```

对比：

```text
NWP 无约束 avg_profit=13340.960251
```

生成文件：

```text
outputs/output_nwp_c0_55_d72_88.csv
```

当前根目录 `output.csv` 已覆盖为该中风险冲分候选。已线上验证的 5117 分版本仍保留在：

```text
outputs/output_nwp_unconstrained.csv
```

### 其他模型尝试

`NWP + exact calendar + forecast bias` 和 `NWP + forecast bias` 在本地收益略有变化，但测试集窗口改动过大，风险类似之前失败的融合模型，因此不作为当前提交推荐。
