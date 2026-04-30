# 电价预测与储能套利调研资料索引

本索引记录本项目参考过的主要论文和资料。筛选原则不是“模型越新越好”，而是看它能不能解释本赛题中的提分动作：价格预测、NWP 外生特征、收益导向验证、储能策略稳健性。

## 论文与资料

| 方向 | 资料 | 关键信息 | 对本项目的启发 |
| --- | --- | --- | --- |
| 电价预测规范 | Lago et al., *Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark*, 2020/2021, [arXiv:2008.08004](https://arxiv.org/abs/2008.08004) | 论文批评很多电价预测研究只在私有短样本上比较，缺少强 baseline、公开 benchmark 和显著性检验。 | 不盲目换复杂模型，先建立可复现的本地验证、收益回测和实验表。 |
| 决策导向预测 | Sang et al., *Electricity Price Prediction for Energy Storage System Arbitrage: A Decision-focused Approach*, 2023, [arXiv:2305.00362](https://arxiv.org/abs/2305.00362) | 论文指出只优化预测误差会忽略下游 ESS arbitrage 收益，主张把储能优化问题纳入训练目标。 | 本项目新增“收益导向模型融合”，用储能回测收益选择预测融合权重，而不是只看 RMSE。 |
| 储能不确定性 | Wu et al., *Energy Storage Arbitrage Under Price Uncertainty: Market Risks and Opportunities*, 2025, [arXiv:2501.08472](https://arxiv.org/abs/2501.08472) | 论文从 robust optimization / chance-constrained optimization 角度分析价格不确定性下的套利收益和风险。 | 策略层保留阈值和多候选提交，不让模型预测的微弱价差直接决定交易。 |
| 储能决策学习 | Yi et al., *Perturbed Decision-Focused Learning for Modeling Strategic Energy Storage*, 2024, [arXiv:2406.17085](https://arxiv.org/abs/2406.17085) | 论文把物理储能模型放进机器学习 pipeline，用扰动让优化层可微。 | 初赛时间有限，不做复杂端到端可微优化；但采用同一思想的简化版：用策略收益来选模型和调后处理。 |
| 长序列时序模型 | Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023, [arXiv:2211.14730](https://arxiv.org/abs/2211.14730) | PatchTST 把时间序列切成 patch，并用 channel-independence 降低 Transformer 成本。 | 深度模型可作为复赛方向；初赛数据量和时间有限，当前先保留 LightGBM + 强特征工程。 |
| 电价预测工具 | `epftoolbox`, [GitHub](https://github.com/jeslago/epftoolbox) / [文档](https://epftoolbox.readthedocs.io/) | 面向电价预测的开源 benchmark 工具箱，强调统一数据、统一指标和可复现比较。 | 当前项目采用脚本化 pipeline、固定验证区间、输出实验 CSV，便于复现和答辩。 |

## 本地资料位置

- PDF 缓存：`outputs/research_pdfs/`
- 文献综述：`docs/research/literature_review.md`
- 实验日志：`docs/research/experiment_log.md`
- 提交候选：`docs/research/submission_candidates.md`

PDF 缓存用于本机阅读，Markdown 报告用于答辩和复盘。
