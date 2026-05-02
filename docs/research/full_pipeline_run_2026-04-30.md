# Full Pipeline Run 2026-04-30

运行命令：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
```

运行结果：成功。

## 关键输出

最终自动选择：

```text
output.csv <- outputs/output_nwp_c0_55_d72_88.csv
```

选择依据来自 `outputs/strategy_compare.csv`：

| candidate | avg_profit | capture_ratio | loss_days | traded_days |
|---|---:|---:|---:|---:|
| `nwp_c0_55_d72_88` | `13370.5786` | `0.8432` | `1` | `56` |
| `base_unconstrained` | `12631.7918` | `0.7584` | `1` | `56` |
| `nwp_c0_55` | `12534.9543` | `0.7526` | `3` | `56` |
| `nwp_unconstrained` | `12496.3491` | `0.7503` | `3` | `56` |
| `nwp_bias_unconstrained` | `10564.3179` | `0.6343` | `2` | `56` |
| `nwp_exact_bias_unconstrained` | `9488.6351` | `0.5697` | `5` | `56` |

最终提交检查：

```text
rows=5664
days=59
traded_days=59
errors=0
warnings=0
```

当前 hash：

```text
output.csv: 3EF3D82B60E05C9CCD5533558BC64D392FFAF10F150F7752B01EA0BD09E78E47
outputs/output_nwp_c0_55_d72_88.csv: 3EF3D82B60E05C9CCD5533558BC64D392FFAF10F150F7752B01EA0BD09E78E47
```

## 保底文件

完整 pipeline 会重新生成 `outputs/output_nwp_unconstrained.csv`，所以旧的线上 5117 文件需要单独保留。

已备份为：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

hash：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

对应已知线上分数：

```text
5117.832037755039
```

## 本轮观察

1. 新接入的历史价格统计特征能完整跑通，但在 2025 年 1-2 月验证窗口上，NWP unconstrained 的本地收益从旧版本约 `13340.96` 变为 `12496.35`，不一定改善主线。
2. 窗口约束 `c0_55_d72_88` 仍然是本地最优策略，说明策略层约束比本轮新增的 bias/exact-bias 特征更有价值。
3. `threshold=500/1000/2000` 在本轮验证中明显减少交易并降低收益，暂时不应作为主线。
4. `nwp_bias` 和 `nwp_exact_bias` 明显弱于普通 NWP，不应提交。

## 当前提交建议

冲分：

```text
output.csv
```

保底：

```text
outputs/output_nwp_unconstrained_online5117.csv
```

## 后续任务规划

### P0：下一次提交前

1. 提交当前 `output.csv`，记录线上分数。
2. 若线上低于 `5117.83`，立即回退提交 `outputs/output_nwp_unconstrained_online5117.csv`。
3. 不再提交 `nwp_bias`、`nwp_exact_bias`、`threshold >= 500` 候选，除非新的滚动验证推翻本轮结论。

### P1：短期提分

1. 跑新特征版本的 rolling validation，判断历史价格统计是否跨窗口稳定有效。
2. 对 `c0_55_d72_88` 做更细的窗口网格搜索，例如：
   - `charge_start_max`: 48-60
   - `discharge_start_min`: 68-76
   - `discharge_start_max`: 84-88
3. 做 `lambda_uncertainty` 网格，比较 robust strategy：
   - `0.25`
   - `0.5`
   - `1.0`
   - `1.5`
4. 用 `tune_monthly_threshold.py` 做更细阈值搜索，但目前粗搜显示阈值不如 `0`。

### P2：模型方向

1. 正式跑 `train_residual_lgb.py`，用同一套 `strategy_compare` 纳入候选。
2. 正式跑 `train_window_ranker.py`，但先用较少 seed 和滚动验证确认方向。
3. 如果 window ranker 有提升，再把它加入 pipeline 默认候选。

### P3：答辩与复盘

1. 把每次线上提交写入 `docs/research/online_feedback_2026-04-30.md`。
2. 更新 `docs/research/candidate_status.md`，明确线上分数和是否可继续提交。
3. 答辩时重点说明：本项目已经从“预测电价”升级为“验证收益、自动选策略、保留线上保底、尝试窗口收益模型”的闭环。
