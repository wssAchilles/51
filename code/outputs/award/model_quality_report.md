# 特等奖增强模型质量报告

## Baseline 摘要
- Q1 Huber CV MAE: 1.2894880059115161
- Q2 baseline transition nodes: [8473, 9501]
- Q3 baseline GBDT R2: 0.8744984905949988
- Q4 baseline table values: [{'时间点': '2025-05-09 12:00:00', '实验集行号': 1125, '阶段标签': 1, '表面位移预测值_mm': 5.1627542885470525}, {'时间点': '2025-05-27 08:00:00', '实验集行号': 3693, '阶段标签': 2, '表面位移预测值_mm': 30.76735742646368}, {'时间点': '2025-06-01 12:00:00', '实验集行号': 4437, '阶段标签': 2, '表面位移预测值_mm': 60.259424049093376}, {'时间点': '2025-06-03 22:00:00', '实验集行号': 4785, '阶段标签': 3, '表面位移预测值_mm': 383.18547525682766}, {'时间点': '2025-06-04 01:40:00', '实验集行号': 4807, '阶段标签': 3, '表面位移预测值_mm': 383.69700818330716}]
- Q5 baseline best combo: ['降雨量_mm', '孔隙水压力_kPa', '微震事件数', '爆破点距离_m', '单段最大药量_kg']

## Award Audit 结论
- Q1 model comparison winner: Huber; bootstrap CI NaN count: 0.
- Q2 composite transition recommendation: [{'transition': 'slow_to_accelerated', 'recommended_serial': 8108, 'recommended_time': Timestamp('2024-06-29 07:10:00.000000001'), 'recommended_displacement_mm': 353.923, 'recommended_velocity_6h_mm_h': 0.6360000000001378, 'supporting_evidence_count': 3, 'note': 'Composite decision distinguishes early curvature/onset evidence from significant velocity-level jump.'}, {'transition': 'accelerated_to_rapid', 'recommended_serial': 9501, 'recommended_time': Timestamp('2024-07-08 23:19:59.999999998'), 'recommended_displacement_mm': 785.711, 'recommended_velocity_6h_mm_h': 2.7809999999997217, 'supporting_evidence_count': 3, 'note': 'Composite decision distinguishes early curvature/onset evidence from significant velocity-level jump.'}].
- Q3 best validation model: HistGBDT; most stable factor: pore.
- Q4 strongest generalized ablation model: baseline_plus_gbdt_residual_CV; training reconstruction winner: baseline_plus_gbdt_residual; max 95% half-width: 16.063 mm.
- Q5 consensus best combo: 降雨量_mm+孔隙水压力_kPa+微震事件数+爆破点距离_m+单段最大药量_kg; inverse-velocity warning records: 34.

## Academic Method Sources
- Changepoint detection: Killick, Fearnhead and Eckley, PELT algorithm, JASA 2012, https://doi.org/10.1080/01621459.2012.737745
- Robust outlier handling: Hampel identifier with median/MAD filtering, https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html
- Landslide inverse-velocity warning: Fukuzono/modified inverse-velocity landslide failure-time literature, e.g. https://www.sciencedirect.com/science/article/pii/S001379521931751X
- Displacement prediction review: physics-based and data-driven landslide displacement prediction review, https://www.sciencedirect.com/science/article/pii/S0012825224002769

## Git Diff Stat
```
bu.aux                                             |   4 +
 code/outputs/figures/q2_segmentation.png           | Bin 134127 -> 134325 bytes
 .../figures/q3_experiment_prediction_scatter.png   | Bin 464878 -> 462624 bytes
 .../figures/q4_experiment_prediction_curve.png     | Bin 74830 -> 71126 bytes
 code/outputs/figures/q4_training_segmentation.png  | Bin 145782 -> 148319 bytes
 .../outputs/figures/q4_training_stage_baseline.png | Bin 83735 -> 81068 bytes
 .../figures/q5_best_model_displacement_fit.png     | Bin 89229 -> 85802 bytes
 code/outputs/figures/q5_stage_segmentation.png     | Bin 155695 -> 161499 bytes
 .../__pycache__/config.cpython-313.pyc             | Bin 1432 -> 1488 bytes
 .../common/__pycache__/plotting.cpython-313.pyc    | Bin 3982 -> 4900 bytes
 code/src/slope_warning/common/plotting.py          |  34 +++--
 code/src/slope_warning/config.py                   |   4 +-
 .../__pycache__/q2_segmentation.cpython-313.pyc    | Bin 6904 -> 6904 bytes
 .../__pycache__/q3_fusion.cpython-313.pyc          | Bin 15220 -> 15201 bytes
 .../__pycache__/q4_prediction.cpython-313.pyc      | Bin 16303 -> 16314 bytes
 .../__pycache__/q5_warning.cpython-313.pyc         | Bin 18146 -> 18178 bytes
 .../src/slope_warning/questions/q2_segmentation.py |   3 +-
 code/src/slope_warning/questions/q3_fusion.py      |   7 +-
 code/src/slope_warning/questions/q4_prediction.py  |   6 +-
 code/src/slope_warning/questions/q5_warning.py     |   4 +-
 frontmatter/abstract.tex                           |  33 ++---
 frontmatter/cover.tex                              |   4 +-
 main.bbl                                           |  24 +++-
 main.pdf                                           | Bin 216388 -> 1633036 bytes
 main.synctex.gz                                    | Bin 71077 -> 130452 bytes
 main.tex                                           |   7 +-
 mainmatter/appendix.tex                            |  13 +-
 mainmatter/chapter1.tex                            |  73 +++++++---
 mainmatter/chapter2.tex                            | 106 ++++++++++++---
 mainmatter/chapter3.tex                            |  92 ++++++++++---
 mainmatter/chapter4.tex                            | 110 ++++++++++++---
 mainmatter/chapter5.tex                            | 150 ++++++++++++++++++---
 ref/refs.bib                                       |  43 +++++-
 thusetup.tex                                       |   5 +-
 34 files changed, 581 insertions(+), 141 deletions(-)
```

## Generated Award Files
- `award_audit_summary.json`
- `model_quality_report.md`
- `q1_bootstrap_correction_ci.csv`
- `q1_calibration_diagnostics.png`
- `q1_model_comparison.csv`
- `q2_final_stage_models.csv`
- `q2_final_transition_decision.csv`
- `q2_transition_candidate_comparison.csv`
- `q2_transition_sensitivity.csv`
- `q3_anomaly_sensitivity.csv`
- `q3_missing_heatmap.png`
- `q3_model_comparison.csv`
- `q3_partial_dependence.png`
- `q3_variable_contribution_stability.csv`
- `q3_variable_trace_panels.png`
- `q4_ablation_comparison.csv`
- `q4_table_4_1_prediction_intervals.csv`
- `q5_inverse_velocity_warning.csv`
- `q5_variable_selection_stability.csv`
- `q5_warning_timeline.png`
