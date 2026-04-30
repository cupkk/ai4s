# 电价预测与储能套利调研报告

## 1. 为什么这次调研重点不是“换一个更深的模型”

本赛题的最终分数不是单纯预测电价的 RMSE，而是把预测电价送入储能充放电策略后得到的收益。因此模型有两个任务：

1. 尽量预测价格曲线的大致形态。
2. 更重要的是，选对每天“相对低价的连续 8 个点”和“相对高价的连续 8 个点”。

这就是为什么只追求点预测误差可能不够。一个模型即使 RMSE 更低，也可能把最高价窗口排错；另一个模型 RMSE 稍高，但能更稳定地分出低谷和高峰，线上收益反而更高。

## 2. 成熟方法给我们的 4 个启发

### 2.1 先做强 baseline 和可复现验证

Lago 等人的电价预测综述指出，很多电价预测论文的问题不是模型不够复杂，而是比较不规范：数据不公开、测试时间太短、没有和强 baseline 认真比较。这个观点对比赛很实用：不要每次只凭一次榜单提交判断，而要固定验证区间、固定实验表、固定生成脚本。

本项目已经落实：

- 固定用 2025 年 1-2 月模拟官方 2026 年 1-2 月测试期。
- 同时记录 RMSE、MAE 和储能收益回测。
- 每个提交文件保留到 `outputs/`，并记录线上反馈。

### 2.2 外生变量很重要，NWP 是上游信号

电价受负荷、新能源出力、联络线、水电、非市场机组等因素影响。NWP 气象数据位于更上游：风速和辐照度会影响风电、光伏，再影响供需平衡和电价。

本项目已经落实：

- 读取官方 `.nc` 文件。
- 提取 `u100/v100/ghi/t2m/tcc/tp/sp` 的空间统计特征。
- 派生 `nwp_wind_speed_*`。
- 按小时扩展到 15 分钟粒度后并入 LightGBM。

线上反馈证明这个方向有效：`outputs/output_nwp_unconstrained.csv` 的线上分数为 `5117.832037755039`，高于原来的 `4903.504068225546`。

### 2.3 从“预测准”转向“收益高”

Sang 等人的 ESS arbitrage 决策导向论文指出，传统电价预测模型通常只优化预测误差，却没有把下游储能套利目标纳入训练。完整的 decision-focused learning 比较复杂，但核心思想可以简化落地：

- 不只看 RMSE。
- 用储能收益回测选择模型、阈值和融合权重。
- 让模型选择服务于最终策略。

本项目新增了 `src/tune_prediction_blend.py`：

- 输入两套验证预测：普通 LightGBM 与 NWP LightGBM。
- 按不同权重融合预测。
- 对每个权重执行储能收益回测。
- 输出本地收益最高的融合候选。

当前本地结果显示，`25% NWP + 75% 非 NWP` 的融合在 2025 年 1-2 月验证集上收益最高：

```text
weight_second=0.25
rmse=0.875604
mae=0.532203
best_threshold=1000
avg_profit=13960.943080
loss_days=0
```

这不保证线上一定超过纯 NWP，但它是一个有实验依据的下一次提交候选。

### 2.4 稳健策略比激进策略更适合预测不确定场景

Wu 等人的储能套利不确定性论文强调，价格预测存在不确定性时，要考虑收益和风险的折中。比赛里我们不能做完整 robust optimization，但可以做轻量版：

- 保留无阈值版本，避免过度保守。
- 同时生成低阈值候选，比如 `threshold=1000`。
- 不再使用过高阈值，因为线上已经证明 NWP 的 `threshold=2000` 从 `5117` 降到 `4903`。

这说明本赛题里“过度防守”会错过收益，阈值只能小幅使用。

## 3. 为什么暂时不把 LSTM / Transformer 作为主线

PatchTST、TFT、Informer、Autoformer 等模型适合长序列预测，但它们通常需要更多样本、更细的调参和更稳定的验证框架。本赛题初赛时间紧，且已有线上反馈表明 NWP 特征和策略后处理能直接涨分。

因此当前优先级是：

1. LightGBM + NWP + 供需平衡特征。
2. 收益导向的模型融合。
3. 小阈值稳健策略。
4. 深度学习作为复赛或长线冲刺方向。

## 4. 当前推荐路线

### 稳妥提交

当前根目录 `output.csv` 已复制为线上最高分文件：

```text
outputs/output_nwp_unconstrained.csv
线上分数：5117.832037755039
```

如果今天只剩一次提交，优先提交这个稳妥版本。

### 冲分候选

```text
outputs/output_blend_fine_w025_t1000.csv
```

它的依据是：在本地同季节验证集上，收益导向融合优于纯 NWP 和纯非 NWP。但因为线上已出现“本地阈值 2000 更好、线上反而更差”的情况，所以该候选适合有额外提交次数时尝试。

2026-04-30 01:33 线上反馈更新：该候选实际分数为 `4703.505815153465`，明显低于纯 NWP 无阈值版本。因此它已经被证伪，不再推荐提交。这个结果反过来说明：本地单一验证集收益会过拟合，后续要把 `outputs/output_nwp_unconstrained.csv` 当作锚点，只做小幅、可解释的窗口调整。

## 5. 答辩可讲的主线

可以按下面逻辑讲：

1. Baseline 只做价格点预测，特征少，策略过度依赖预测。
2. 我们先补充供需平衡、时间周期、历史统计、NWP 气象特征。
3. 线上验证表明 NWP 是有效信号，分数从 `4903.50` 提升到 `5117.83`。
4. 阅读文献后发现，储能套利不能只看 RMSE，要关注下游收益。
5. 因此新增收益导向融合脚本，用本地储能回测选择模型融合权重。
6. 当前最稳版本是纯 NWP 无阈值；下一候选是 `25% NWP + 75% 非 NWP + threshold=1000`。

## 6. 参考资料

- Lago et al., *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*, [arXiv:2008.08004](https://arxiv.org/abs/2008.08004)
- Sang et al., *Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-focused Approach*, [arXiv:2305.00362](https://arxiv.org/abs/2305.00362)
- Wu et al., *Energy Storage Arbitrage Under Price Uncertainty: Market Risks and Opportunities*, [arXiv:2501.08472](https://arxiv.org/abs/2501.08472)
- Yi et al., *Perturbed Decision-Focused Learning for Modeling Strategic Energy Storage*, [arXiv:2406.17085](https://arxiv.org/abs/2406.17085)
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)
- `epftoolbox` open benchmark, [GitHub](https://github.com/jeslago/epftoolbox)
