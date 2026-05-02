from __future__ import annotations

from slope_warning.common.diagnostics import git_diff_stat
from slope_warning.common.io import write_text
from slope_warning.config import AWARD_DIR, PROJECT_DIR


def build_report(results: dict[str, object], baseline: dict[str, object]) -> None:
    model_summary = baseline.get("model_summary", {}) if isinstance(baseline, dict) else {}
    q4_update = bool(results.get("q4", {}).get("update_main_table", False))
    q4_main = results.get("q4", {}).get("main_shrinkage")
    q4_recommended = results.get("q4", {}).get("recommended_shrinkage")
    if q4_main == q4_recommended:
        q4_table_action = f"已采用 CV 推荐的残差收缩系数 {q4_main}，当前主表维持"
    else:
        q4_table_action = "建议更新" if q4_update else "维持当前主表"
    q1_update = bool(results.get("q1", {}).get("update_main_table", False))
    report = [
        "# 模型稳健性检验报告",
        "",
        "## 基准结果锁定",
        "- 策略：稳健优先。只有扩展检验显示主模型明显更优时，才调整正文主结果。",
        f"- 已锁定主表数量：{len(baseline.get('tables', {})) if isinstance(baseline, dict) else 0}",
        f"- 已锁定主 PDF：{'是' if baseline.get('main_pdf') else '否'}",
        "",
        "## 基准结果摘要",
        f"- Q1 Huber CV MAE: {model_summary.get('q1', {}).get('cv_mean', {}).get('MAE_mm', 'NA')}",
        f"- Q2 基准转换节点: {model_summary.get('q2', {}).get('break_serial_numbers', 'NA')}",
        f"- Q3 基准 GBDT R2: {model_summary.get('q3', {}).get('gbdt', {}).get('R2', 'NA')}",
        f"- Q4 基准表格结果: {model_summary.get('q4', {}).get('target_predictions', 'NA')}",
        f"- Q5 基准最优组合: {model_summary.get('q5', {}).get('best_combo', 'NA')}",
        "",
        "## 扩展检验结论",
        f"- Q1：模型对照第一名为 {results['q1']['best_model']}；Huber delta 推荐值为 {results['q1']['recommended_delta_scale']}；bootstrap 区间 NaN 数为 {results['q1']['ci_nan_count']}。",
        f"- Q2：复合证据节点至少由 {results['q2']['supporting_evidence_min_count']} 类证据支持；权重扰动最小稳定比例为 {results['q2']['weight_stability_min_share']:.3f}。",
        f"- Q3：验证集最优模型为 {results['q3']['best_model']}；最稳定贡献因子为 {results['q3']['top_factor']}；补齐后缺失数为 {results['q3']['post_fill_missing_count']}。",
        f"- Q4：广义验证最优消融模型为 {results['q4']['best_ablation']}；残差收缩推荐值为 {results['q4']['recommended_shrinkage']}，主模型取值为 {results['q4']['main_shrinkage']}，经验覆盖率为 {results['q4'].get('interval_empirical_coverage', 'NA')}。",
        f"- Q5：一致排序最优变量组合为 {results['q5']['best_combo']}；逆速度预警记录数为 {results['q5']['inverse_warning_count']}；阈值严格递增：{results['q5']['thresholds_strictly_increasing']}。",
        "",
        "## 正文结果一致性",
        f"- Q1结果处理：{'建议更新' if q1_update else '维持原结果'}。",
        f"- Q4结果处理：{q4_table_action}。",
        "- 其他问题：新增结果用于模型稳健性分析和附录复现说明，不替换原主结果。",
        "",
        "## 新增关键文件",
    ]
    key_files = [
        "baseline_lock.json",
        "q1_huber_delta_sensitivity.csv",
        "q1_residual_diagnostics.csv",
        "q2_weight_sensitivity.csv",
        "q2_transition_weight_stability_summary.csv",
        "q3_continuous_fill_diagnostics.csv",
        "q3_common_anomaly_event_summary.csv",
        "q4_residual_shrinkage_sensitivity.csv",
        "q4_prediction_interval_coverage.csv",
        "q5_consensus_conflict_analysis.csv",
        "q5_warning_event_summary.csv",
        "q5_threshold_stability.csv",
        "validation_summary.json",
    ]
    report.extend(f"- `{name}`" for name in key_files if (AWARD_DIR / name).exists() or name == "validation_summary.json")
    report.extend(
        [
            "",
            "## 建模合理性说明",
            "- 高阶校准：用 OLS、Huber、Theil-Sen 和 RANSAC 对照，并以时间块 CV 和 P95AE 防止少量异常点主导。",
            "- 位移分段偏早或偏晚：阶段节点由位移趋势、持续加速度和速度跃迁三类证据融合，并做权重网格扰动。",
            "- 真实降雨峰误删：稀疏驱动量与连续状态量分治，降雨和微震采用事件型阈值，不用普通 Hampel 直接削峰。",
            "- 黑箱预测绝对位移：问题四先建阶段归一化单调基线，再让机器学习只修正扰动残差，保留工程解释。",
            "- 单分位数预警：问题五同时使用速度阈值、持续时间、多源扰动增强和逆速度趋势，形成事件级闭环。",
            "",
            "## Git Diff Stat",
            "```",
            git_diff_stat(PROJECT_DIR),
            "```",
            "",
            "## 全量扩展检验输出",
        ]
    )
    for path in sorted(AWARD_DIR.iterdir()):
        if path.is_file():
            report.append(f"- `{path.name}`")
    write_text("\n".join(report) + "\n", AWARD_DIR / "model_quality_report.md")
