# 项目后续任务执行规划

基于 `项目分析集后续规划.md`、当前仓库代码、已知线上分数和最近实验结果，后续任务不能再简单按“多加模型”推进，而要围绕评分函数建立闭环。

当前已知最好线上版本：

```text
outputs/output_nwp_unconstrained.csv
线上分数: 5117.832037755039
```

当前根目录 `output.csv` 状态：

```text
output.csv = outputs/output_nwp_c0_55_d72_88.csv
```

这是基于 NWP 主线加窗口约束的冲分候选，尚未线上验证。如果提交后低于 5117，应立即恢复：

```powershell
Copy-Item outputs\output_nwp_unconstrained.csv output.csv -Force
```

---

## 1. 对原分析文档的核心判断

原文最重要的判断是正确的：

1. 这个比赛本质是储能套利比赛，不是单纯电价预测比赛。
2. `power` 列决定最终分数，`实时价格` 只是生成策略的中间信息。
3. RMSE 下降不等于收益上升，窗口排序更重要。
4. 需要建立 `oracle_profit / capture_ratio / regret / window_hit` 诊断体系。
5. 当前最该修的是闭环、验证和提交一致性，而不是盲目上深度模型。

但有几处需要结合当前实验修正：

| 原建议 | 当前结论 | 后续处理 |
| --- | --- | --- |
| NWP 维度可能错 | 已核对 `.nc`：`data=(1,24,7,104,225)`，当前按 `hour,channel,lat,lon` 读取是对的 | 不作为首要 bug，但要保留自适应维度检查 |
| NWP UTC/BJ 可能错 | GHI 峰值在北京时间 13:00 左右，当前错 8 小时的概率不高 | 继续加诊断脚本，不急着改时间 |
| 直接窗口收益模型最重要 | 已做原型，本地 `11617`，低于 NWP `13341` | 不作为当前提交主线，后续要重做 ranking 而不是简单 regression |
| 融合/季节先验可冲分 | `w025_t1000` 线上只有 `4703` | 融合类候选降级，除非有 rolling validation 支撑 |
| 阈值兜底稳健 | `threshold=2000` 线上从 `5117` 降到 `4903` | 暂不使用高阈值，优先保持交易 |

---

## 2. P0：当天必须完成

### P0.1 修正 `run_pipeline.ps1` 的最终提交逻辑

问题：

`scripts/run_pipeline.ps1` 当前最后一步写死：

```powershell
--threshold 2000
```

这会复现已经线上变差的版本。

目标：

脚本默认生成当前稳妥主线：

```text
outputs/output_nwp_unconstrained.csv
```

并额外生成候选，不直接覆盖稳妥版本。

建议输出：

```text
outputs/output_nwp_unconstrained.csv
outputs/output_nwp_c0_55.csv
outputs/output_nwp_c0_55_d72_88.csv
outputs/submission_compare.csv
```

验收标准：

1. 运行脚本后不会自动生成 `threshold=2000` 作为默认 `output.csv`。
2. 每个候选都有对应 meta 文件。
3. 文档中明确哪个是线上验证版，哪个是冲分候选。

### P0.2 增加提交合法性检查

新增脚本：

```text
src/check_submission.py
```

检查项：

1. 是否 5664 行。
2. 是否 59 天，每天 96 行。
3. `times` 是否连续 15 分钟。
4. 列名是否为 `times, 实时价格, power`。
5. `power` 是否只包含 `-1000, 0, 1000`。
6. 每天最多一段充电、一段放电。
7. 每段长度是否等于 8。
8. 是否先充电后放电。

验收标准：

```powershell
python -m src.check_submission --csv output.csv
```

输出 `PASS`，否则列出具体日期和错误。

### P0.3 扩展收益诊断指标

修改：

```text
src/validate_profit.py
```

新增指标：

```text
oracle_profit
capture_ratio
regret
oracle_charge_start
oracle_discharge_start
pred_charge_start
pred_discharge_start
charge_abs_error
discharge_abs_error
```

验收标准：

回测输出不只看 `avg_profit`，还输出：

```text
capture_ratio_mean
regret_mean
charge_hit_rate_±2
discharge_hit_rate_±2
```

用途：

之后判断模型坏在哪里：是低价充电窗口错，还是高价放电窗口错。

### P0.4 固化当前提交候选状态

当前候选分层：

| 类型 | 文件 | 状态 |
| --- | --- | --- |
| 稳妥版 | `outputs/output_nwp_unconstrained.csv` | 已线上 5117 |
| 当前冲分版 | `outputs/output_nwp_c0_55_d72_88.csv` | 待线上验证 |
| 低风险候选 | `outputs/output_nwp_c0_55.csv` | 只改 7 天，可作为备选 |
| 失败候选 | `outputs/output_blend_fine_w025_t1000.csv` | 已线上 4703，不再提交 |
| 高阈值候选 | `outputs/output_nwp_unconstrained_t2000.csv` | 已线上 4903，不再提交 |

---

## 3. P1：1 到 2 天内完成

### P1.1 建立候选策略对比表

新增脚本：

```text
scripts/compare_strategies.ps1
```

或 Python 版：

```text
src/compare_strategies.py
```

输入多个预测/提交文件，输出：

```text
strategy_name
avg_profit
loss_days
capture_ratio
regret
trade_days
same_window_vs_best_known
```

验收标准：

生成：

```text
outputs/strategy_compare.csv
```

### P1.2 Rolling validation

当前最大问题是只用 2025 年 1-2 月做验证，容易过拟合。需要至少加：

```text
val=2025-01~02
val=2025-03~04
val=2025-05~06
val=2025-07~08
val=2025-09~10
val=2025-11~12
```

对每个策略输出：

```text
avg_profit_mean
avg_profit_std
loss_days_total
capture_ratio_mean
```

验收标准：

如果某策略只在 1-2 月好，但其他月份差，不能作为强提交候选。

### P1.3 验证联络线符号

当前特征里只有：

```text
net_load_with_intertie = load - wind - pv - hydro - non_market - intertie
```

应新增：

```text
net_load_with_intertie_plus = load - wind - pv - hydro - non_market + intertie
```

并输出相关性：

```text
corr(price, net_load_with_intertie_minus)
corr(price, net_load_with_intertie_plus)
```

验收标准：

让模型同时拥有 plus/minus 两个特征，减少方向猜错风险。

### P1.4 NWP 特征升级但不改时间

当前 NWP 维度读取基本正确，GHI 峰值也在合理时段。因此先不动时间对齐，优先增强特征：

```text
wind_speed_p10 / p50 / p90
wind_speed_cube_mean
ghi_p90
daylight_ghi_sum
noon_ghi_mean
ghi_x_1_minus_tcc
t2m_celsius_mean
heating_degree
```

验收标准：

新 NWP 特征单独训练一版模型，必须在收益诊断上超过当前 NWP baseline，才进入候选。

---

## 4. P2：3 到 5 天内完成

### P2.1 残差 LightGBM

思路：

```text
price = seasonal_prior + residual
```

其中：

```text
seasonal_prior = month_slot_mean / dow_slot_mean / dayofyear_slot_mean
target = A - seasonal_prior
```

注意：

之前 `exact calendar history + forecast bias` 模型测试窗口变化过大，不应直接提交。残差模型必须和窗口稳定性一起评估。

验收标准：

1. 本地 `capture_ratio` 提升。
2. 测试集窗口相对 NWP baseline 不应大面积漂移，除非 rolling validation 支撑。

### P2.2 价格历史特征

已有 `src/lag_features.py` 和 `src/train_lgb_lag.py`，但递归测试效果不稳定。下一步不要只做递归 lag，而是做不泄漏的历史统计：

```text
same_month_slot_median
same_month_slot_q25
same_month_slot_q75
dayofyear_slot_mean
month_day_slot_mean
recent_slot_mean
recent_slot_std
```

测试集只能使用训练历史，不能使用测试真实价格。

### P2.3 窗口 ranking 模型第二版

已做过简单窗口收益 regression，结果不如 NWP。下一版要改：

1. 使用 LightGBM ranker，而不是普通 regression。
2. 每天作为一个 query group。
3. 标签用候选窗口收益排序或 top-k gain。
4. 特征必须包含 NWP 预测价差、供需窗口差、历史先验价差。

验收标准：

在 2025 年 1-2 月之外的 rolling validation 中也超过 NWP baseline，才可提交。

---

## 5. P3：时间充足再做

这些属于复赛或长线冲刺：

1. Quantile LightGBM：低价窗口用 q20，高价窗口用 q80。
2. 交易分类模型：判断当天是否跳过交易。
3. PatchTST / TFT / TimesNet。
4. Stacking 二层模型。
5. Robust spread：

```text
pred_spread - λ * model_disagreement - μ * historical_error_std
```

当前不建议优先做深度模型，因为数据量只有一年，且已有线上反馈证明错误策略比模型复杂度更致命。

---

## 6. 最近两次行动建议

### 下一次提交

提交当前根目录：

```text
output.csv
```

它等于：

```text
outputs/output_nwp_c0_55_d72_88.csv
```

这是冲分候选，不是稳妥候选。

### 如果分数低于 5117

立即恢复：

```powershell
Copy-Item outputs\output_nwp_unconstrained.csv output.csv -Force
```

然后下一步只提交：

```text
outputs/output_nwp_c0_55.csv
```

因为它只改 7 天，风险更低。

### 如果分数高于 5117

保留该策略为新主线，并继续在其附近做小范围搜索：

```text
discharge_start_min = 70, 71, 72, 73
charge_start_max = 54, 55, 56
threshold = 0
```

不要再回到大幅融合模型。

---

## 7. 总结

原文档的方向是对的，但执行顺序需要结合当前线上反馈调整。

最优先不是深度学习，也不是继续堆融合，而是：

1. 修提交脚本，避免复现低分版本。
2. 加提交合法性检查。
3. 加 oracle / regret / capture ratio 诊断。
4. 做 rolling validation，避免单一 1-2 月过拟合。
5. 在 NWP 主线附近做小幅策略搜索。
6. 再做残差模型、历史特征、ranking 模型。

当前项目的主线应是：

```text
NWP LightGBM 价格预测
+ 收益诊断
+ 窗口稳定性分析
+ 小范围策略约束搜索
```

而不是：

```text
大幅融合模型 / 高阈值 / 未验证深度模型
```
