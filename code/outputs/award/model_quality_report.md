# 模型稳健性检验报告

## 基准结果锁定
- 策略：稳健优先。只有扩展检验显示主模型明显更优时，才调整正文主结果。
- 已锁定主表数量：26
- 已锁定主 PDF：是

## 基准结果摘要
- Q1 Huber CV MAE: 1.2894880059115161
- Q2 基准转换节点: [8473, 9501]
- Q3 基准 GBDT R2: 0.8744984905949988
- Q4 基准表格结果: [{'时间点': '2025-05-09 12:00:00', '实验集行号': 1125, '阶段标签': 1, '表面位移预测值_mm': 5.1627542885470525}, {'时间点': '2025-05-27 08:00:00', '实验集行号': 3693, '阶段标签': 2, '表面位移预测值_mm': 30.76735742646368}, {'时间点': '2025-06-01 12:00:00', '实验集行号': 4437, '阶段标签': 2, '表面位移预测值_mm': 60.259424049093376}, {'时间点': '2025-06-03 22:00:00', '实验集行号': 4785, '阶段标签': 3, '表面位移预测值_mm': 383.18547525682766}, {'时间点': '2025-06-04 01:40:00', '实验集行号': 4807, '阶段标签': 3, '表面位移预测值_mm': 383.69700818330716}]
- Q5 基准最优组合: ['降雨量_mm', '孔隙水压力_kPa', '微震事件数', '爆破点距离_m', '单段最大药量_kg']

## 扩展检验结论
- Q1：模型对照第一名为 Huber；Huber delta 推荐值为 1.0；bootstrap 区间 NaN 数为 0。
- Q2：复合证据节点至少由 3 类证据支持；权重扰动最小稳定比例为 0.840。
- Q3：验证集最优模型为 HistGBDT；最稳定贡献因子为 pore；补齐后缺失数为 0。
- Q4：广义验证最优消融模型为 baseline_plus_gbdt_residual_CV；残差收缩推荐值为 0.8，主模型取值为 0.8，经验覆盖率为 0.9241。
- Q5：一致排序最优变量组合为 降雨量_mm+孔隙水压力_kPa+微震事件数+爆破点距离_m+单段最大药量_kg；逆速度预警记录数为 34；阈值严格递增：True。

## 正文结果一致性
- Q1结果处理：维持原结果。
- Q4结果处理：已采用 CV 推荐的残差收缩系数 0.8，当前主表维持。
- 其他问题：新增结果用于模型稳健性分析和附录复现说明，不替换原主结果。

## 新增关键文件
- `baseline_lock.json`
- `q1_huber_delta_sensitivity.csv`
- `q1_residual_diagnostics.csv`
- `q2_weight_sensitivity.csv`
- `q2_transition_weight_stability_summary.csv`
- `q3_continuous_fill_diagnostics.csv`
- `q3_common_anomaly_event_summary.csv`
- `q4_residual_shrinkage_sensitivity.csv`
- `q4_prediction_interval_coverage.csv`
- `q5_consensus_conflict_analysis.csv`
- `q5_warning_event_summary.csv`
- `q5_threshold_stability.csv`
- `validation_summary.json`

## 建模合理性说明
- 高阶校准：用 OLS、Huber、Theil-Sen 和 RANSAC 对照，并以时间块 CV 和 P95AE 防止少量异常点主导。
- 位移分段偏早或偏晚：阶段节点由位移趋势、持续加速度和速度跃迁三类证据融合，并做权重网格扰动。
- 真实降雨峰误删：稀疏驱动量与连续状态量分治，降雨和微震采用事件型阈值，不用普通 Hampel 直接削峰。
- 黑箱预测绝对位移：问题四先建阶段归一化单调基线，再让机器学习只修正扰动残差，保留工程解释。
- 单分位数预警：问题五同时使用速度阈值、持续时间、多源扰动增强和逆速度趋势，形成事件级闭环。

## Git Diff Stat
```
code/outputs/award/award_audit_summary.json        |   37 +-
 code/outputs/award/model_quality_report.md         |  147 +-
 code/outputs/award/q1_calibration_diagnostics.png  |  Bin 156505 -> 173482 bytes
 .../outputs/award/q2_final_transition_decision.csv |    4 +-
 .../award/q2_transition_candidate_comparison.csv   |   12 +-
 code/outputs/award/q3_missing_heatmap.png          |  Bin 85359 -> 101071 bytes
 code/outputs/award/q3_partial_dependence.png       |  Bin 189999 -> 218824 bytes
 .../award/q3_variable_contribution_stability.csv   |   18 +-
 code/outputs/award/q3_variable_trace_panels.png    |  Bin 480080 -> 542624 bytes
 code/outputs/award/q4_ablation_comparison.csv      |   16 +-
 .../award/q4_table_4_1_prediction_intervals.csv    |    8 +-
 .../award/q5_variable_selection_stability.csv      |   14 +-
 code/outputs/award/q5_warning_timeline.png         |  Bin 208405 -> 235299 bytes
 .../figures/q3_variable_contribution_bar.png       |  Bin 98527 -> 98530 bytes
 .../figures/q4_experiment_prediction_curve.png     |  Bin 71126 -> 70996 bytes
 code/outputs/models/all_model_summaries.json       |    9 +-
 code/outputs/models/q4_model_summary.json          |    9 +-
 .../tables/q4_experiment_surface_predictions.csv   | 5422 ++++++++++----------
 code/outputs/tables/q4_table_4_1_predictions.csv   |    8 +-
 code/pyproject.toml                                |    5 +
 code/scripts/make_paper_diagrams.py                |    2 +-
 code/src/slope_warning/award_audit.py              |  616 +--
 code/src/slope_warning/common/preprocessing.py     |    4 +-
 code/src/slope_warning/config.py                   |   23 +
 code/src/slope_warning/questions/q4_prediction.py  |   74 +-
 code/src/slope_warning/questions/q5_warning.py     |    1 +
 code/uv.lock                                       |   54 +
 frontmatter/abstract.tex                           |    2 +-
 main.pdf                                           |  Bin 8904984 -> 8989936 bytes
 main.synctex.gz                                    |  Bin 173372 -> 179106 bytes
 mainmatter/appendix.tex                            |    8 +-
 mainmatter/chapter1.tex                            |    2 +-
 mainmatter/chapter2.tex                            |    4 +-
 mainmatter/chapter3.tex                            |    4 +-
 mainmatter/chapter4.tex                            |   26 +-
 mainmatter/chapter5.tex                            |   10 +-
 mainmatter/chapter6.tex                            |   42 +-
 mainmatter/chapter7.tex                            |   20 +-
 mainmatter/chapter8.tex                            |   40 +-
 thusetup.tex                                       |    1 +
 40 files changed, 3172 insertions(+), 3470 deletions(-)
```

## 全量扩展检验输出
- `award_audit_summary.json`
- `baseline_lock.json`
- `baseline_main.pdf`
- `model_quality_report.md`
- `q1_bootstrap_correction_ci.csv`
- `q1_calibration_diagnostics.png`
- `q1_huber_delta_sensitivity.csv`
- `q1_model_comparison.csv`
- `q1_residual_diagnostics.csv`
- `q2_final_stage_models.csv`
- `q2_final_transition_decision.csv`
- `q2_transition_candidate_comparison.csv`
- `q2_transition_sensitivity.csv`
- `q2_transition_weight_stability_summary.csv`
- `q2_weight_sensitivity.csv`
- `q3_anomaly_sensitivity.csv`
- `q3_common_anomaly_event_summary.csv`
- `q3_continuous_fill_diagnostics.csv`
- `q3_missing_heatmap.png`
- `q3_model_comparison.csv`
- `q3_partial_dependence.png`
- `q3_variable_contribution_stability.csv`
- `q3_variable_trace_panels.png`
- `q4_ablation_comparison.csv`
- `q4_prediction_interval_coverage.csv`
- `q4_residual_shrinkage_sensitivity.csv`
- `q4_table_4_1_prediction_intervals.csv`
- `q5_consensus_conflict_analysis.csv`
- `q5_inverse_velocity_leadtime_summary.csv`
- `q5_inverse_velocity_warning.csv`
- `q5_threshold_stability.csv`
- `q5_variable_selection_stability.csv`
- `q5_warning_event_summary.csv`
- `q5_warning_timeline.png`
- `q5_warning_window_traces.csv`
- `validation_summary.json`
