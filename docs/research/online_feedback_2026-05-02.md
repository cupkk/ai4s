# 线上反馈记录 2026-05-02

## 保底分来源补充

用户补充确认：当前保底分 `5117.832037755039` 来自一次较早提交。

```text
线上反馈时间：2026-04-30 02:05:29
提交文件：D:\github\ai4s\output.csv
用户判断：应该是 2026-04-29 的老版本 output 文件
线上分数：5117.832037755039
```

当前本地核对结果：

```text
output.csv LastWriteTime: 2026-04-29 18:41:58
output.csv SHA256: AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
outputs/output_nwp_unconstrained_online5117.csv SHA256: AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

解释：`5117.832037755039` 不是新候选的反馈，而是确认当前保底文件对应 2026-04-29 老版本 `output.csv` 的线上成绩。后续仍应把这个文件作为保底基线，所有新候选先与它做逐日 action diff。

## 反馈结果

提交文件：

```text
output.csv
```

当时 `output.csv` 等同于：

```text
outputs/output_nwp_c0_55_d72_88.csv
```

线上反馈：

```text
日期：2026-05-02 10:34:01
分数：3798.629342284567
```

## 直接结论

`c0_55_d72_88` 不能再作为主线。它虽然在本地验证和滚动验证里表现好，但在 2026 测试集上明显低于已知保底分：

```text
outputs/output_nwp_unconstrained_online5117.csv
5117.832037755039
```

因此已经执行止损：

```powershell
Copy-Item outputs/output_nwp_unconstrained_online5117.csv output.csv -Force
python -m src.check_submission --submission output.csv
```

当前 `output.csv` hash：

```text
AD83C1BE3298381D39CC0848ACBE4E664A8E0860E9333D75BE7073C64D6D0AF8
```

## 动作差异分析

新增脚本：

```text
src/analyze_submission_diff.py
```

运行命令：

```powershell
python -m src.analyze_submission_diff `
  --reference outputs/output_nwp_unconstrained_online5117.csv `
  --candidate outputs/output_nwp_c0_55_d72_88.csv `
  --reference-name safe5117 `
  --candidate-name bad3798 `
  --output outputs/action_diff_safe5117_vs_bad3798.csv `
  --summary-output outputs/action_diff_safe5117_vs_bad3798_summary.csv
```

结果摘要：

| 指标 | 数值 |
|---|---:|
| days | 59 |
| same_both_days | 0 |
| same_charge_days | 9 |
| same_discharge_days | 8 |
| changed_days | 59 |
| mean_abs_charge_delta | 7.3729 |
| mean_abs_discharge_delta | 7.3390 |
| max_abs_charge_delta | 49 |
| max_abs_discharge_delta | 50 |

解释：低分文件不是在保底文件基础上小修小补，而是 59 天每天都改变了交易窗口，因此线上掉分很大。

## 候选相似度扫描

新增输出：

```text
outputs/candidate_similarity_to_safe5117.csv
```

最接近保底文件的候选：

| 文件 | same_both_days | 说明 |
|---|---:|---|
| `outputs/output1.csv` | 59 | 与保底完全相同。 |
| `outputs/output_blend_w100.csv` | 59 | 与保底完全相同。 |
| `outputs/output_nwp_robust_l1_smoke.csv` | 59 | 与保底完全相同。 |
| `outputs/output_nwp_moderate_t2000.csv` | 52 | 只改 7 天，但未线上验证。 |

## 新生成的小扰动候选

基于保底预测 `outputs/test_predictions_blend_w100.csv`，只做“低预测收益日不交易”，不改变其他日期窗口。

生成命令逻辑：

```powershell
python -m src.make_submission `
  --price-csv outputs/test_predictions_blend_w100.csv `
  --threshold <threshold> `
  --output outputs/output_safe5117_skip_t<threshold>.csv
```

候选：

| 文件 | traded_days | 相对保底改动 |
|---|---:|---|
| `outputs/output_safe5117_skip_t500.csv` | 58 | 只跳过 2026-01-11 |
| `outputs/output_safe5117_skip_t1000.csv` | 56 | 跳过 2026-01-08、2026-01-11、2026-02-20 |
| `outputs/output_safe5117_skip_t1500.csv` | 55 | 跳过 4 天 |
| `outputs/output_safe5117_skip_t2000.csv` | 54 | 跳过 5 天 |
| `outputs/output_safe5117_skip_t2500.csv` | 52 | 跳过 7 天 |
| `outputs/output_safe5117_skip_t3000.csv` | 51 | 跳过 8 天 |

## 下一次提交建议

如果目标是保住当前最高分：

```text
提交 output.csv
```

如果目标是尝试小幅冲分，且当天还有提交次数：

```text
优先提交 outputs/output_safe5117_skip_t500.csv
```

理由：它只改 1 天，属于最小风险线上探测。如果它高于 5117，再考虑更高阈值；如果它低于 5117，就说明保底文件中低预测收益日也可能有正收益，不应继续提高阈值。
