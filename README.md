# AI4S 电价预测与储能优化

本仓库把初赛任务拆成两步：

1. 用历史边界条件预测次日 96 点实时电价；
2. 基于预测电价枚举连续 8 点充电窗口和连续 8 点放电窗口，生成 `output.csv`。

当前第一版不使用外部数据，也暂不接入 `.nc` 气象文件，优先完成可验证、可提交的 LightGBM 集成 + 储能优化流水线。

## 数据放置

```text
data/
  train/
    mengxi_boundary_anon_filtered.csv
    mengxi_node_price_selected.csv
  test/
    test_in_feature_ori.csv
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 训练并调阈值

```bash
python -m src.train_lgb \
  --train-feature data/train/mengxi_boundary_anon_filtered.csv \
  --train-label data/train/mengxi_node_price_selected.csv \
  --val-start-date 2025-01-01 \
  --val-end-date 2025-02-28 \
  --seeds 42,2024,2026
```

训练脚本支持按日期指定验证集。官方测试集是 2026 年 1-2 月，因此默认建议用 2025 年 1-2 月做同季节验证；收益回测会自动跳过训练标签中不完整的日子。输出：

- `outputs/lgb_model.txt`
- `outputs/lgb_model_metadata.json`
- `outputs/val_predictions.csv`
- `outputs/threshold_search.csv`
- `outputs/best_threshold.txt`

验证指标同时包含价格误差和储能回测收益。阈值网格默认是：

```text
0,5000,10000,20000,30000,50000,80000,100000
```

## 预测测试集价格

```bash
python -m src.predict \
  --test-feature data/test/test_in_feature_ori.csv \
  --model outputs/lgb_model.txt \
  --metadata outputs/lgb_model_metadata.json \
  --output outputs/test_predictions.csv
```

## 生成提交文件

```bash
python -m src.make_submission \
  --price-csv outputs/test_predictions.csv \
  --metadata outputs/lgb_model_metadata.json \
  --output output.csv
```

生成文件列名为 `times,实时价格,power`，每天严格要求 96 行。`power` 只会取 `-1000, 0, 1000`，并满足先连续充电 8 点、再连续放电 8 点的约束。

## 低分后的稳健策略

如果直接用模型逐日选窗口在线上分数偏低，可以生成更稳健的“季节先验 + 模型预测混合”策略：

```bash
python -m src.make_blended_submission \
  --train-label to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv \
  --price-csv outputs/test_predictions.csv \
  --alpha 0.25 \
  --output output.csv
```

`alpha=0.25` 表示策略窗口 75% 参考 2025 同月份平均价格曲线，25% 参考模型预测。它比原模型更不容易被单日预测噪声带偏。

Windows PowerShell 可直接运行：

```powershell
.\scripts\run_pipeline.ps1
```

如果官方数据保持在当前仓库的解压目录，可直接运行默认脚本；默认路径为 `to_sais_new/to_sais_new/...`。

## 关键模块

- `src/features.py`：时间周期、供需平衡、变化率、历史统计特征；
- `src/train_lgb.py`：按天验证、LightGBM 训练、收益阈值搜索；
- `src/predict.py`：测试集电价预测；
- `src/storage_optimizer.py`：储能策略精确枚举；
- `src/validate_profit.py`：验证集收益回测；
- `src/make_submission.py`：生成 `output.csv`。
