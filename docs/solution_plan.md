# AI4S 电价预测与储能优化赛题解题方案

## 1. 赛题理解

本赛题本质上不是单纯的“电价预测题”，而是一个 **预测 + 优化决策** 的组合赛题。整体目标包括两个连续任务：

1. 根据历史实时电价、边界条件和气象信息，预测未来一天（D+1 日）96 个 15 分钟实时电价点；
2. 根据预测电价制定储能设备的充放电计划，在给定约束下最大化储能收益。

赛题中的储能系统具有明确约束：每天最多执行一次完整充放电操作；充电必须连续 8 个时间点，放电也必须连续 8 个时间点；充电功率固定为 `-1000`，放电功率固定为 `+1000`；初始 SOC 为 0。

收益公式为：

$$
Profit=\sum_{t=0}^{95}P_tE_t
$$

其中，$P_t$ 为真实实时电价，$E_t$ 为提交的储能功率。低价时充电、高价时放电可以获得正收益；如果预测错误，则可能出现亏损。

因此，本题真正要优化的不是单纯的 RMSE，而是：

> 预测曲线能否准确识别“低价连续 2 小时窗口”和“高价连续 2 小时窗口”的相对位置。

---

## 2. 与优化课程内容的对应关系

本题可以自然对应前面 15 节优化课程中的多个核心内容。

第 9、10 节的问题分解思想可以用于整体架构设计：将赛题拆分为“电价预测模块”和“储能优化模块”。预测模型负责输出未来价格序列，优化模块负责基于价格序列生成最优充放电策略。

第 11、12 节的组合优化思想对应储能调度部分。每天只能选择一个充电起点和一个放电起点，且都必须连续 8 个点，这本质上是一个离散决策问题。由于一天只有 96 个点，可以直接枚举所有合法组合并精确求解。

第 13 节的随机优化和鲁棒优化思想可以用于处理预测误差。预测价格存在不确定性，因此不应只要预测收益大于 0 就交易，而应引入安全阈值 $\tau$，只有预测价差足够大时才交易，从而降低亏损风险。

第 14、15 节的 MDP 和动态控制思想在本赛题当前阶段不是主方法，因为本题是 day-ahead 离线决策，不是实时滚动控制。但如果后续扩展到多次充放电、实时调度、SOC 连续控制，则可以进一步使用动态规划、MDP 或强化学习方法。

当前最合理的路线是：

> **机器学习预测电价 + 组合优化精确枚举充放电窗口 + 鲁棒阈值控制交易风险。**

---

## 3. 整体解题框架

方案建议分成三层。

第一层是 **强基线预测模型**。先使用 LightGBM 作为主模型，因为它对表格数据非常强，而且官方基线代码已经采用 LightGBM。基线使用系统负荷预测值、风光总加预测值、联络线预测值、风电预测值、光伏预测值、水电预测值、非市场化机组预测值，以及 hour、minute、dayofweek、month 等时间特征。这条路线是正确的，但仍然偏简单，需要进一步加强特征工程和验证方式。

第二层是 **收益导向的模型选择**。不能只看验证集 RMSE / MAE，还要在验证集上模拟每天的储能交易，并用真实电价计算真实收益。最终模型选择应同时关注：

$$
RMSE,\quad MAE,\quad Validation\ Profit
$$

其中，真正与比赛排名更相关的是 `Validation Profit`。

第三层是 **确定性 + 鲁棒储能优化**。对每一天的 96 点预测价格，精确枚举所有合法的充电起点和放电起点，找到预测收益最大的组合；然后加入风险阈值，如果预测价差不够大，则不交易，以避免因预测误差导致亏损。

---

## 4. 电价预测模块设计

当前 LightGBM baseline 是最小可运行版本，但建议在其基础上做四类增强。

### 4.1 时间特征增强

不要只使用 `hour`、`minute`、`dayofweek`、`month`，还应加入一天中的 15 分钟时间槽：

$$
slot=\frac{hour\times 60+minute}{15}
$$

`slot` 的范围为 0 到 95。进一步加入周期特征：

$$
\sin(2\pi slot/96),\quad \cos(2\pi slot/96)
$$

电价具有明显的日内周期。直接使用 `hour` 是离散值，模型不一定能理解 23 点和 0 点相近；而 sin/cos 特征可以表达周期连续性。

### 4.2 供需平衡特征

电价本质上受供需关系影响，因此建议构造：

$$
net\_load=系统负荷预测值-风电预测值-光伏预测值-水电预测值-非市场化机组预测值
$$

如果联络线代表外来电输入，可以根据数据含义尝试将其加入或减去，最终以验证集收益为准。

还可以构造：

$$
renewable\_ratio=\frac{风电预测值+光伏预测值}{系统负荷预测值+\epsilon}
$$

$$
supply\_margin=风光总加预测值+水电预测值+非市场化机组预测值-系统负荷预测值
$$

这些特征比原始列更接近电力市场出清价格的形成逻辑。

### 4.3 变化率特征

电价经常受 ramp 影响，例如新能源突然下降或负荷突然升高都可能导致价格波动。因此建议对预测边界条件做差分：

$$
\Delta x_t=x_t-x_{t-1}
$$

例如系统负荷预测值变化、风电预测值变化、光伏预测值变化等。注意这些特征可以在测试集计算，因为它们来自未来边界条件预测值，不属于标签泄漏。

### 4.4 历史统计特征

不要使用测试未来真实价格。可以基于训练集历史价格构造统计特征，例如：

$$
price\_slot\_mean(slot)
$$

$$
price\_month\_slot\_mean(month,slot)
$$

$$
price\_dow\_slot\_mean(dayofweek,slot)
$$

这些历史统计特征不会泄漏测试标签，能够为模型提供“某个时刻通常价格水平”的先验。

如果后续使用气象 `.nc` 文件，可以对 `u100`、`v100`、`t2m`、`tp`、`tcc`、`sp`、`ghi` 等变量在经纬度网格上取 `mean`、`max`、`min`、`std`，再将逐小时气象特征插值或 repeat 到 15 分钟粒度。第一阶段建议先不使用气象数据，优先把边界条件和时间特征做到位。

---

## 5. 验证方式设计

不能使用随机划分。由于赛题是按天预测和按天决策，因此验证也应按天评估。

建议采用 **按天滚动验证**：

1. 用前 70% 天训练，后 10% 天验证；
2. 再向后滚动一段时间；
3. 每次输出验证集 RMSE、MAE 和真实回测收益。

最终模型选择标准不应只看价格误差，而应看验证集真实收益：

$$
ValidationScore=\frac{1}{D_{val}}\sum_d Profit_d
$$

如果一个模型 RMSE 较低，但总是错过最高价或最低价窗口，它未必适合本题；另一个模型 RMSE 稍高，但能够抓住高低价时段，反而可能收益更高。

---

## 6. 储能优化模块

储能优化部分要精确求解，不建议使用简单 heuristic。

对某一天的预测价格：

$$
\hat{P}_0,\hat{P}_1,\dots,\hat{P}_{95}
$$

如果选择充电起点 $t_c$，连续充 8 个点，则充电成本为：

$$
-1000\sum_{t=t_c}^{t_c+7}\hat{P}_t
$$

如果选择放电起点 $t_d$，连续放 8 个点，则放电收益为：

$$
1000\sum_{t=t_d}^{t_d+7}\hat{P}_t
$$

因此预测总收益为：

$$
\widehat{Profit}(t_c,t_d)=1000\left(\sum_{t=t_d}^{t_d+7}\hat{P}_t-\sum_{t=t_c}^{t_c+7}\hat{P}_t\right)
$$

约束为：

$$
0\le t_c\le 80
$$

$$
t_d\ge t_c+8
$$

$$
t_d\le 88
$$

最优策略为：

$$
(t_c^*,t_d^*)=\arg\max_{t_c,t_d}\widehat{Profit}(t_c,t_d)
$$

如果最大预测收益小于阈值 $\tau$，则全天不交易，即功率全部为 0。阈值 $\tau$ 应在验证集上搜索得到。

---

## 7. 储能优化代码逻辑

### 7.1 单日优化

```python
import numpy as np
import pandas as pd


def optimize_one_day(prices, threshold=0.0):
    """
    prices: 长度为96的一天预测电价
    threshold: 交易安全阈值，预测收益低于该值则不交易
    return: 长度96的power序列
    """
    prices = np.asarray(prices)
    power = np.zeros(96)

    # 8点连续窗口价格和
    block_sum = np.array([prices[i:i+8].sum() for i in range(89)])

    best_profit = -np.inf
    best_tc, best_td = None, None

    for tc in range(0, 81):      # 0 <= tc <= 80
        charge_cost = block_sum[tc]

        for td in range(tc + 8, 89):  # td >= tc+8, td <= 88
            discharge_revenue = block_sum[td]
            profit = 1000 * (discharge_revenue - charge_cost)

            if profit > best_profit:
                best_profit = profit
                best_tc, best_td = tc, td

    if best_profit > threshold:
        power[best_tc:best_tc+8] = -1000
        power[best_td:best_td+8] = 1000

    return power, best_profit, best_tc, best_td
```

### 7.2 按天生成策略

```python
def generate_strategy(price_csv, save_path, threshold=0.0):
    df = pd.read_csv(price_csv)
    df['times'] = pd.to_datetime(df['times'])

    # 兼容列名：可能是 A，也可能是 实时价格
    if 'A' in df.columns:
        price_col = 'A'
    elif '实时价格' in df.columns:
        price_col = '实时价格'
    else:
        raise ValueError("找不到价格列，应为 A 或 实时价格")

    df = df.sort_values('times').reset_index(drop=True)
    df['date'] = df['times'].dt.date

    out_list = []

    for date, g in df.groupby('date'):
        g = g.sort_values('times').copy()

        if len(g) != 96:
            raise ValueError(f"{date} 不是96个点，而是 {len(g)} 个点")

        prices = g[price_col].values
        power, pred_profit, tc, td = optimize_one_day(prices, threshold=threshold)

        g['实时价格'] = prices
        g['power'] = power
        out_list.append(g[['times', '实时价格', 'power']])

    out = pd.concat(out_list, axis=0)
    out.to_csv(save_path, index=False)
    return out
```

这部分应作为 `src/storage_optimizer.py` 的核心逻辑。

---

## 8. 鲁棒优化思想

预测价格存在误差，因此不能只看预测收益是否大于 0。建议在验证集上对阈值 $\tau$ 做网格搜索，例如：

```python
threshold_grid = [0, 5000, 10000, 20000, 30000, 50000]
```

对于每个阈值，用验证集预测价格生成策略，再用验证集真实价格计算真实收益，选择平均真实收益最高的阈值。

进一步可以做不确定性估计。训练多个 LightGBM 模型，例如：

- 不同随机种子；
- 不同特征子集；
- 不同时间验证折。

得到多个预测价格：

$$
\hat{P}_t^{(1)},\hat{P}_t^{(2)},\dots,\hat{P}_t^{(K)}
$$

对每个候选充放电策略，计算预测收益均值和标准差：

$$
\mu_{profit},\quad \sigma_{profit}
$$

再定义风险调整收益：

$$
Score_{risk}=\mu_{profit}-\lambda\sigma_{profit}
$$

选择风险调整后收益最高的策略。这对应第 13 节随机优化与鲁棒优化思想：不只看收益高不高，还要看收益稳不稳。

---

## 9. 完整比赛路线

第一版：跑通强 baseline。目标是从原始数据读取、训练 LightGBM、预测测试集、生成 `output.csv`。第一阶段不建议使用气象数据，也不建议使用深度模型，优先补齐时间特征、供需特征和收益优化函数。

第二版：做验证集收益优化。将训练集最后一段按天作为验证集，每一天用预测价格选充放电窗口，再用真实价格计算真实收益。重点调三个部分：LightGBM 参数、特征组合、交易阈值 $\tau$。

第三版：做模型集成和鲁棒策略。可以训练 LGBM + CatBoost + XGBoost，或者多个不同随机种子的 LGBM。预测价格取平均，策略使用风险调整收益。该版本能进一步提高稳定性。

如果时间有限，做到第二版即可；如果想冲高分，再做第三版。

---

## 10. GitHub 仓库结构建议

建议将 `ai4s` 仓库整理为以下结构：

```text
ai4s/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── train_lgb.py
│   ├── predict.py
│   ├── storage_optimizer.py
│   ├── validate_profit.py
│   └── make_submission.py
├── scripts/
│   ├── run_train.sh
│   ├── run_predict.sh
│   └── run_submission.sh
├── notebooks/
│   └── eda.ipynb
└── outputs/
    └── .gitkeep
```

各文件作用如下：

- `data_loader.py`：读取训练边界条件、标签、测试边界条件和气象数据；
- `features.py`：生成时间特征、供需特征、差分特征和历史统计特征；
- `train_lgb.py`：训练 LightGBM 并保存模型；
- `predict.py`：对测试集生成电价预测；
- `storage_optimizer.py`：实现连续 8 点充放电的枚举优化器；
- `validate_profit.py`：验证集按天回测收益，用真实电价算分；
- `make_submission.py`：生成最终 `output.csv`，列名应为 `times`、`实时价格`、`power`。

---

## 11. README 示例

```markdown
# AI4S 电价预测与储能优化

本项目解决电价预测与储能优化赛题。整体流程包括：

1. 基于边界条件和时间特征预测 D+1 日 96 点实时电价；
2. 将储能调度建模为带连续 8 点充放电约束的组合优化问题；
3. 通过精确枚举求解最优充电/放电窗口；
4. 在验证集上使用收益回测选择交易阈值，提高鲁棒性；
5. 生成符合提交格式的 output.csv。

核心思想：

- 预测模型：LightGBM / 集成学习；
- 优化模型：组合优化精确枚举；
- 鲁棒策略：验证集收益驱动阈值选择。
```

---

## 12. 方案数学建模

价格预测模型写成：

$$
\hat{P}_{d,t}=f_\theta(X_{d,t})
$$

其中，$X_{d,t}$ 是边界条件、时间特征和气象特征等。

储能优化模型写成：

$$
\max_{t_c,t_d}1000\left(\sum_{k=0}^{7}\hat{P}_{d,t_d+k}-\sum_{k=0}^{7}\hat{P}_{d,t_c+k}\right)
$$

约束为：

$$
0\le t_c\le 80
$$

$$
t_c+8\le t_d\le 88
$$

如果最大值小于阈值 $\tau$，则：

$$
E_{d,t}=0,\quad \forall t
$$

否则：

$$
E_{d,t}=\begin{cases}
-1000, & t_c\le t<t_c+8\\
1000, & t_d\le t<t_d+8\\
0, & otherwise
\end{cases}
$$

这个数学表达与课程中的组合优化、鲁棒优化和分解思想直接对应。

---

## 13. 实验与调参设计

建议建立本地验证回测表：

| 模型 | RMSE | MAE | 验证集平均收益 | 交易天数 | 亏损天数 |
|---|---:|---:|---:|---:|---:|
| baseline LGBM |  |  |  |  |  |
| + 时间周期特征 |  |  |  |  |  |
| + 供需特征 |  |  |  |  |  |
| + 历史统计特征 |  |  |  |  |  |
| + 阈值鲁棒策略 |  |  |  |  |  |
| + ensemble |  |  |  |  |  |

这张表比单纯报告 RMSE 更有价值，因为赛题最终比拼的是储能收益，而不是预测误差本身。

建议绘制以下图表：

1. 某几天真实价格 vs 预测价格；
2. 某天充电和放电窗口可视化；
3. 阈值 $\tau$ 与验证集平均收益关系；
4. 不同模型的收益对比。

---

## 14. 常见风险与避坑

第一，不要使用测试集中不存在的实际值列。训练时如果使用“系统负荷实际值”“风电实际值”等，而测试集没有这些字段，线上会失效。只能使用测试阶段可获得的预测值列，或者使用历史统计特征。

第二，不要只优化 RMSE。收益由充放电窗口决定，RMSE 小但高低价排序错了，收益也可能很差。

第三，不要每天强制交易。如果预测价差不够大，应选择不交易。很多时候少交易反而更稳。

第四，不要一开始使用复杂深度模型。LSTM、Transformer 并非不能用，但本赛题是表格数据 + 时间序列 + 气象特征，LightGBM 强特征工程通常性价比更高。

第五，不要把储能优化写成“找全局最低 8 点和最高 8 点”。赛题要求必须是连续 8 个时间点，而且放电必须在充电之后，正确方法是枚举连续 block。

---

## 15. 最终实现顺序

建议按照以下顺序推进：

1. 将 `lgb_baseline.py` 改成可配置路径，并整理为 `src/train_lgb.py`；
2. 编写 `features.py`，加入 `slot`、`sin/cos`、`net_load`、`renewable_ratio`、`ramp` 等特征；
3. 编写 `storage_optimizer.py`，实现连续 8 点充放电的精确枚举；
4. 编写 `validate_profit.py`，使用验证集真实价格回测收益；
5. 编写 `make_submission.py`，生成最终 `output.csv`。

最终核心策略是：

> 用 LightGBM 把电价曲线预测到“高低价排序尽量正确”，再用组合优化精确选择连续 2 小时低价充电窗口和连续 2 小时高价放电窗口，最后用验证集调交易阈值，避免预测不确定时亏损。
