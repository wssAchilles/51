from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import theilslopes
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, RANSACRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slope_warning.common.diagnostics import git_diff_stat, inverse_velocity_forecast, monotonic_violations, time_block_model_scores
from slope_warning.common.features import make_disturbance_features
from slope_warning.common.io import read_excel, write_csv, write_json, write_text
from slope_warning.common.metrics import mae, p95ae, rmse, time_block_folds
from slope_warning.common.plotting import configure_chinese_fonts
from slope_warning.common.preprocessing import hampel_flags, hampel_replace, kalman_smooth_fill, rolling_median, sparse_outlier_flags, sparse_series_fill
from slope_warning.common.segmentation import fit_stage_polynomial, two_breaks_constant_mean, two_breaks_piecewise_linear
from slope_warning.config import ATTACHMENTS, AWARD_DIR, MODEL_DIR, PROJECT_DIR, TABLE_DIR, ensure_output_dirs
from slope_warning.questions import q1_calibration, q3_fusion, q4_prediction


RNG = np.random.default_rng(20260501)
TARGET_Q1 = q1_calibration.TARGET_VALUES
TARGET_Q4 = q4_prediction.TARGET_TIMES
configure_chinese_fonts()


def _time_block_cv_affine(x: np.ndarray, y: np.ndarray, fit_fn, k: int = 5) -> dict[str, float]:
    rows = []
    for test in time_block_folds(len(x), k):
        train = np.setdiff1d(np.arange(len(x)), test, assume_unique=True)
        beta = fit_fn(x[train], y[train])
        pred = beta[0] + beta[1] * x[test]
        rows.append(
            {
                "RMSE": rmse(y[test], pred),
                "MAE": mae(y[test], pred),
                "P95AE": p95ae(y[test], pred),
                "MaxAE": float(np.max(np.abs(y[test] - pred))),
            }
        )
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def _ols_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)[0]


def _theilsen_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    slope, intercept, _, _ = theilslopes(y, x)
    return np.array([intercept, slope], dtype=float)


def _ransac_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = RANSACRegressor(estimator=LinearRegression(), random_state=20260501, min_samples=0.5, residual_threshold=5.0)
    model.fit(x.reshape(-1, 1), y)
    return np.array([model.estimator_.intercept_, model.estimator_.coef_[0]], dtype=float)


def _weighted_median(values: list[int], weights: list[float]) -> int:
    order = np.argsort(values)
    vals = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cutoff = 0.5 * w.sum()
    return int(vals[np.searchsorted(np.cumsum(w), cutoff)])


def _save_q1_plots(x: np.ndarray, y: np.ndarray, beta: np.ndarray) -> None:
    pred = beta[0] + beta[1] * x
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(x, y, s=8, alpha=0.35)
    xs = np.linspace(x.min(), x.max(), 200)
    axes[0].plot(xs, beta[0] + beta[1] * xs, color="#d62728", lw=1.5)
    axes[0].set_xlabel("校正前数据A/mm")
    axes[0].set_ylabel("基准数据B/mm")
    axes[0].set_title("问题1：稳健校准模型")
    axes[0].grid(alpha=0.25)
    axes[1].hist(pred - y, bins=60, color="#1f77b4", alpha=0.75)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_xlabel("校正残差/mm")
    axes[1].set_ylabel("频数")
    axes[1].set_title("残差分布")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q1_calibration_diagnostics.png", dpi=220)
    plt.close(fig)


def q1_award() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q1"])
    x = df["数据A_光纤位移计数据_mm"].to_numpy(float)
    y = df["数据B_振弦式位移计数据_mm"].to_numpy(float)
    fits = {
        "OLS": _ols_fit,
        "Huber": q1_calibration.huber_affine_fit,
        "Theil-Sen": _theilsen_fit,
        "RANSAC": _ransac_fit,
    }
    rows = []
    for name, fit_fn in fits.items():
        beta = fit_fn(x, y)
        pred = beta[0] + beta[1] * x
        cv = _time_block_cv_affine(x, y, fit_fn)
        rows.append(
            {
                "model": name,
                "beta0": beta[0],
                "beta1": beta[1],
                "train_RMSE": rmse(y, pred),
                "train_MAE": mae(y, pred),
                "train_P95AE": p95ae(y, pred),
                **{f"cv_{k}": v for k, v in cv.items()},
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["cv_MAE", "cv_RMSE"])
    write_csv(comparison, AWARD_DIR / "q1_model_comparison.csv")

    boot = []
    for _ in range(250):
        idx = RNG.integers(0, len(x), len(x))
        beta = q1_calibration.huber_affine_fit(x[idx], y[idx])
        boot.append(beta[0] + beta[1] * TARGET_Q1)
    boot_arr = np.asarray(boot)
    main_beta = q1_calibration.huber_affine_fit(x, y)
    main_pred = main_beta[0] + main_beta[1] * TARGET_Q1
    ci = pd.DataFrame(
        {
            "校正前数据x": TARGET_Q1,
            "Huber校正值": main_pred,
            "bootstrap_CI_lower_2.5%": np.percentile(boot_arr, 2.5, axis=0),
            "bootstrap_CI_upper_97.5%": np.percentile(boot_arr, 97.5, axis=0),
            "bootstrap_std": boot_arr.std(axis=0),
        }
    )
    write_csv(ci, AWARD_DIR / "q1_bootstrap_correction_ci.csv")
    _save_q1_plots(x, y, main_beta)
    return {"best_model": str(comparison.iloc[0]["model"]), "ci_nan_count": int(ci.isna().sum().sum())}


def _first_sustained(values: np.ndarray, threshold: float, start: int = 0, span: int = 72, ratio: float = 0.8) -> int | None:
    above = np.asarray(values) > threshold
    for idx in range(start, len(above) - span):
        if above[idx : idx + span].mean() >= ratio:
            return idx
    return None


def q2_award() -> dict[str, object]:
    df = read_excel(ATTACHMENTS["q2"])
    y = df["表面位移_mm"].to_numpy(float)
    t_hours = (df["编号"].to_numpy(float) - 1.0) / 6.0
    start_time = pd.Timestamp("2024-05-04 00:00:00")
    raw_v = np.r_[np.nan, np.diff(y) * 6.0]
    raw_v[0] = np.nanmedian(raw_v[1:20])
    clean_v, _ = hampel_replace(raw_v, window=37, threshold=5.0, center=True)

    sensitivity_rows = []
    velocity_breaks = []
    for window in [18, 36, 72]:
        v_smooth = rolling_median(clean_v, window, center=False, min_periods=max(6, window // 2))
        b1, b2, sse = two_breaks_constant_mean(v_smooth, min_len=500, step=20, refine=120)
        velocity_breaks.append((b1, b2))
        sensitivity_rows.append(
            {
                "method": "velocity_level",
                "window_h": window / 6,
                "break1_serial": b1 + 1,
                "break2_serial": b2 + 1,
                "break1_time": start_time + pd.to_timedelta(b1 / 6, unit="h"),
                "break2_time": start_time + pd.to_timedelta(b2 / 6, unit="h"),
                "objective_sse": sse,
            }
        )
    disp_b1, disp_b2, disp_sse = two_breaks_piecewise_linear(t_hours, y, min_len=500, step=20, refine=120)
    sensitivity_rows.append(
        {
            "method": "displacement_piecewise_linear",
            "window_h": 0,
            "break1_serial": disp_b1 + 1,
            "break2_serial": disp_b2 + 1,
            "break1_time": start_time + pd.to_timedelta(disp_b1 / 6, unit="h"),
            "break2_time": start_time + pd.to_timedelta(disp_b2 / 6, unit="h"),
            "objective_sse": disp_sse,
        }
    )
    sens = pd.DataFrame(sensitivity_rows)
    write_csv(sens, AWARD_DIR / "q2_transition_sensitivity.csv")

    v6 = rolling_median(clean_v, 36, center=False, min_periods=18)
    baseline = v6[:3000]
    med = np.median(baseline)
    mad = 1.4826 * np.median(np.abs(baseline - med))
    accel_onsets = []
    for mult in [3, 5, 8, 10]:
        threshold = med + mult * mad
        onset = _first_sustained(v6, threshold, span=72, ratio=0.8)
        accel_onsets.append((mult, threshold, onset))
    b1v, b2v = velocity_breaks[1]
    stage2_mean = v6[b1v:b2v].mean()
    stage3_mean = v6[b2v:].mean()
    rapid_threshold = (stage2_mean + stage3_mean) / 2.0
    rapid_onset = _first_sustained(v6, rapid_threshold, start=max(0, b1v - 72), span=36, ratio=0.8) or b2v

    candidate_rows = [
        {
            "transition": "slow_to_accelerated",
            "evidence": "displacement_trend_change",
            "serial": disp_b1 + 1,
            "weight": 0.20,
            "description": "Piecewise displacement SSE optimum; marks early curvature change.",
        },
        {
            "transition": "slow_to_accelerated",
            "evidence": "sustained_acceleration_onset",
            "serial": (accel_onsets[1][2] or b1v) + 1,
            "weight": 0.35,
            "description": "First 12h window where 6h velocity stays above baseline+5MAD.",
        },
        {
            "transition": "slow_to_accelerated",
            "evidence": "velocity_level_jump",
            "serial": b1v + 1,
            "weight": 0.45,
            "description": "Velocity-level segmentation optimum; marks significant speed regime jump.",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "displacement_trend_change",
            "serial": disp_b2 + 1,
            "weight": 0.25,
            "description": "Piecewise displacement SSE optimum.",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "rapid_velocity_persistence",
            "serial": rapid_onset + 1,
            "weight": 0.25,
            "description": "First persistent crossing of midpoint between accelerated and rapid velocity levels.",
        },
        {
            "transition": "accelerated_to_rapid",
            "evidence": "velocity_level_jump",
            "serial": b2v + 1,
            "weight": 0.50,
            "description": "Velocity-level segmentation optimum.",
        },
    ]
    candidates = pd.DataFrame(candidate_rows)
    candidates["time"] = candidates["serial"].map(lambda s: start_time + pd.to_timedelta((s - 1) / 6, unit="h"))
    candidates["displacement_mm"] = candidates["serial"].map(lambda s: y[int(s) - 1])
    candidates["velocity_6h_mm_h"] = candidates["serial"].map(lambda s: v6[int(s) - 1])
    write_csv(candidates, AWARD_DIR / "q2_transition_candidate_comparison.csv")

    decision_rows = []
    for transition, group in candidates.groupby("transition"):
        recommended = _weighted_median(group["serial"].astype(int).tolist(), group["weight"].astype(float).tolist())
        decision_rows.append(
            {
                "transition": transition,
                "recommended_serial": recommended,
                "recommended_time": start_time + pd.to_timedelta((recommended - 1) / 6, unit="h"),
                "recommended_displacement_mm": y[recommended - 1],
                "recommended_velocity_6h_mm_h": v6[recommended - 1],
                "supporting_evidence_count": int(len(group)),
                "note": "Composite decision distinguishes early curvature/onset evidence from significant velocity-level jump.",
            }
        )
    decision = pd.DataFrame(decision_rows)
    order = {"slow_to_accelerated": 1, "accelerated_to_rapid": 2}
    decision = decision.sort_values("transition", key=lambda s: s.map(order)).reset_index(drop=True)
    write_csv(decision, AWARD_DIR / "q2_final_transition_decision.csv")

    final_bounds = [
        0,
        int(decision.loc[0, "recommended_serial"]) - 1,
        int(decision.loc[1, "recommended_serial"]) - 1,
        len(y),
    ]
    stage_rows = []
    for fit in fit_stage_polynomial(t_hours, y, final_bounds, max_degree=3):
        row = fit.__dict__.copy()
        row["start_time"] = start_time + pd.to_timedelta((fit.start - 1) / 6, unit="h")
        row["end_time"] = start_time + pd.to_timedelta((fit.end - 1) / 6, unit="h")
        row["coefficients"] = ";".join(f"{v:.10g}" for v in fit.coefficients)
        stage_rows.append(row)
    write_csv(pd.DataFrame(stage_rows), AWARD_DIR / "q2_final_stage_models.csv")
    return {"recommended": decision.to_dict(orient="records")}


def q3_award() -> dict[str, object]:
    train_raw = read_excel(ATTACHMENTS["q3"], sheet_name="训练集")
    train = q3_fusion._standardize(train_raw, q3_fusion.TRAIN_COLUMNS)
    proc, flags, _ = q3_fusion._preprocess(train, has_surface=True)
    features, groups = q3_fusion._make_features(proc)
    y = proc["surface"].to_numpy(float)
    split = int(len(y) * 0.8)
    x_fit, x_val = features.iloc[:split], features.iloc[split:]
    y_fit, y_val = y[:split], y[split:]

    sensitivity_rows = []
    for threshold in [3.5, 4.0, 4.5, 5.0, 5.5]:
        row = {"continuous_hampel_threshold": threshold}
        for var in ["pore", "deep", "surface"]:
            filled = kalman_smooth_fill(train[var].to_numpy(float), nonnegative=var in {"deep", "surface"})
            var_flags, _, _ = hampel_flags(filled, window=73, threshold=threshold, center=True)
            row[f"{var}_outliers"] = int(var_flags.sum())
        for quantile in [0.995, 0.997, 0.999]:
            rain = sparse_series_fill(train["rain"].to_numpy(float))
            micro = sparse_series_fill(train["micro"].to_numpy(float), integer=True)
            row[f"rain_outliers_q{quantile}"] = int(sparse_outlier_flags(rain, quantile=quantile, min_threshold=15.0).sum())
            row[f"micro_outliers_q{quantile}"] = int(sparse_outlier_flags(micro, quantile=quantile, min_threshold=6.0).sum())
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity, AWARD_DIR / "q3_anomaly_sensitivity.csv")

    model_factories = {
        "ElasticNet": lambda: make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=10000, random_state=20260501)),
        "GBDT": lambda: GradientBoostingRegressor(random_state=20260501, n_estimators=420, learning_rate=0.035, max_depth=3, min_samples_leaf=18, subsample=0.85),
        "HistGBDT": lambda: HistGradientBoostingRegressor(random_state=20260501, max_iter=260, learning_rate=0.045, max_leaf_nodes=23, l2_regularization=0.03),
    }
    model_rows = []
    fitted_models = {}
    for name, factory in model_factories.items():
        model = factory()
        model.fit(x_fit, y_fit)
        pred = model.predict(x_val)
        fitted_models[name] = model
        model_rows.append(
            {
                "model": name,
                "RMSE_mm": mean_squared_error(y_val, pred) ** 0.5,
                "MAE_mm": mean_absolute_error(y_val, pred),
                "R2": r2_score(y_val, pred),
            }
        )
    model_comparison = pd.DataFrame(model_rows).sort_values("RMSE_mm")
    write_csv(model_comparison, AWARD_DIR / "q3_model_comparison.csv")

    contrib_rows = []
    for fold_id, test in enumerate(time_block_folds(len(y), k=5), start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        model = model_factories["GBDT"]()
        model.fit(features.iloc[train_idx], y[train_idx])
        base = mean_squared_error(y[test], model.predict(features.iloc[test])) ** 0.5
        for factor, columns in groups.items():
            x_perm = features.iloc[test].copy()
            existing = [col for col in columns if col in x_perm.columns]
            order = RNG.permutation(len(x_perm))
            x_perm.loc[:, existing] = x_perm[existing].to_numpy()[order]
            score = mean_squared_error(y[test], model.predict(x_perm)) ** 0.5
            contrib_rows.append({"fold": fold_id, "factor": factor, "RMSE_increase": score - base})
    stability = pd.DataFrame(contrib_rows).groupby("factor", as_index=False)["RMSE_increase"].agg(["mean", "std"]).reset_index()
    write_csv(stability.sort_values("mean", ascending=False), AWARD_DIR / "q3_variable_contribution_stability.csv")

    _save_q3_variable_panels(train, proc, flags)
    _save_q3_missing_heatmap(train_raw)
    _save_q3_partial_dependence(features, fitted_models["GBDT"], AWARD_DIR / "q3_partial_dependence.png")
    return {"best_model": str(model_comparison.iloc[0]["model"]), "top_factor": str(stability.sort_values("mean", ascending=False).iloc[0]["factor"])}


def _save_q3_variable_panels(raw: pd.DataFrame, proc: pd.DataFrame, flags: dict[str, np.ndarray]) -> None:
    variables = ["rain", "pore", "micro", "deep", "surface"]
    labels = {"rain": "降雨量", "pore": "孔隙水压力", "micro": "微震事件数", "deep": "深部位移", "surface": "表面位移"}
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 10), sharex=True)
    x = raw["编号"]
    for ax, var in zip(axes, variables):
        ax.plot(x, raw[var], color="#9e9e9e", lw=0.7, alpha=0.75, label="原始值")
        ax.plot(x, proc[var], color="#1f77b4", lw=0.8, label="预处理后")
        idx = np.where(flags[var])[0]
        if len(idx):
            ax.scatter(x.iloc[idx], proc[var].iloc[idx], color="#d62728", s=12, label="异常点")
        ax.set_ylabel(labels[var])
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("编号")
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q3_variable_trace_panels.png", dpi=220)
    plt.close(fig)


def _save_q3_missing_heatmap(raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    cols = list(q3_fusion.TRAIN_COLUMNS.keys())
    missing = raw[cols].isna().T
    missing.index = ["降雨量", "孔隙水压力", "微震事件数", "深部位移", "表面位移"]
    sns.heatmap(missing, cbar=False, ax=ax, cmap="Blues")
    ax.set_title("问题3：缺失值分布")
    ax.set_xlabel("样本序号")
    ax.set_ylabel("变量")
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q3_missing_heatmap.png", dpi=220)
    plt.close(fig)


def _save_q3_partial_dependence(features: pd.DataFrame, model: object, path: Path) -> None:
    top_features = ["deep", "pore", "rain_24h", "micro_6h"]
    labels = {"deep": "深部位移", "pore": "孔隙水压力", "rain_24h": "24h累计降雨", "micro_6h": "6h微震累计"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for ax, feature in zip(axes, top_features):
        grid = np.quantile(features[feature], np.linspace(0.05, 0.95, 40))
        preds = []
        sample = features.sample(n=min(2000, len(features)), random_state=20260501)
        for value in grid:
            changed = sample.copy()
            changed[feature] = value
            preds.append(float(np.mean(model.predict(changed))))
        ax.plot(grid, preds, color="#1f77b4", lw=1.5)
        ax.set_xlabel(labels.get(feature, feature))
        ax.set_ylabel("表面位移预测值/mm")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def q4_award() -> dict[str, object]:
    train = read_excel(ATTACHMENTS["q4"], sheet_name="训练集")
    exp = read_excel(ATTACHMENTS["q4"], sheet_name="实验集")
    train["时间"] = pd.to_datetime(train["时间"])
    exp["时间"] = pd.to_datetime(exp["时间"])
    y = train["表面位移_mm"].to_numpy(float)
    t_hours = (train["时间"] - train["时间"].iloc[0]).dt.total_seconds().to_numpy() / 3600.0
    b1, b2, _ = two_breaks_piecewise_linear(t_hours, y, min_len=500, step=20, refine=120)
    stage = q4_prediction._stage_from_breaks(len(train), b1, b2)
    tau = q4_prediction._stage_tau(stage)
    baselines = q4_prediction._fit_stage_baselines(y, stage)
    baseline = q4_prediction._baseline_values(stage, baselines)
    residual = y - baseline
    features, _ = make_disturbance_features(train, stage=stage, tau=tau)

    direct_cv = q4_prediction._direct_cv(features, y, stage)
    rows = [
        {"model": "global_disturbance_direct_CV", "validation_scope": "time_block_cv", "RMSE_mm": direct_cv["global_direct_RMSE_mm"], "MAE_mm": direct_cv["global_direct_MAE_mm"]},
        {"model": "staged_disturbance_direct_CV", "validation_scope": "stage_time_block_cv", "RMSE_mm": direct_cv["staged_direct_RMSE_mm"], "MAE_mm": direct_cv["staged_direct_MAE_mm"]},
        {"model": "stage_normalized_baseline", "validation_scope": "training_reconstruction", "RMSE_mm": mean_squared_error(y, baseline) ** 0.5, "MAE_mm": mean_absolute_error(y, baseline)},
    ]
    for name, factory in [
        ("baseline_plus_linear_residual", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ("baseline_plus_gbdt_residual", q4_prediction._gbdt),
    ]:
        pred = np.zeros_like(y)
        for label in sorted(np.unique(stage)):
            idx = np.where(stage == label)[0]
            model = factory()
            model.fit(features.iloc[idx], residual[idx])
            pred[idx] = baseline[idx] + model.predict(features.iloc[idx])
        rows.append(
            {
                "model": name,
                "validation_scope": "stagewise_training_reconstruction",
                "RMSE_mm": mean_squared_error(y, pred) ** 0.5,
                "MAE_mm": mean_absolute_error(y, pred),
                "monotonic_violations": monotonic_violations(pred),
            }
        )
        cv_pred = np.full_like(y, np.nan, dtype=float)
        for label in sorted(np.unique(stage)):
            idx = np.where(stage == label)[0]
            for local_test in time_block_folds(len(idx), k=3):
                test_idx = idx[local_test]
                train_idx = np.setdiff1d(idx, test_idx, assume_unique=True)
                model = factory()
                model.fit(features.iloc[train_idx], residual[train_idx])
                cv_pred[test_idx] = baseline[test_idx] + model.predict(features.iloc[test_idx])
        mask = np.isfinite(cv_pred)
        rows.append(
            {
                "model": f"{name}_CV",
                "validation_scope": "stage_residual_time_block_cv",
                "RMSE_mm": mean_squared_error(y[mask], cv_pred[mask]) ** 0.5,
                "MAE_mm": mean_absolute_error(y[mask], cv_pred[mask]),
                "monotonic_violations": monotonic_violations(cv_pred[mask]),
            }
        )
    ablation = pd.DataFrame(rows).sort_values("RMSE_mm")
    write_csv(ablation, AWARD_DIR / "q4_ablation_comparison.csv")
    generalized = ablation[ablation["validation_scope"].str.contains("cv")]
    best_generalized = generalized.iloc[0] if not generalized.empty else ablation.iloc[0]

    target_table = pd.read_csv(TABLE_DIR / "q4_table_4_1_predictions.csv")
    interval_rows = []
    for _, row in target_table.iterrows():
        stage_label = int(row["阶段标签"])
        idx = np.where(stage == stage_label)[0]
        stage_resid = residual[idx]
        width = max(1.96 * np.std(stage_resid), np.percentile(np.abs(stage_resid), 90))
        pred = float(row["表面位移预测值_mm"])
        interval_rows.append(
            {
                "时间点": row["时间点"],
                "实验集行号": int(row["实验集行号"]),
                "阶段标签": stage_label,
                "point_prediction_mm": pred,
                "lower_95_mm": max(0.0, pred - width),
                "upper_95_mm": pred + width,
                "interval_half_width_mm": width,
            }
        )
    intervals = pd.DataFrame(interval_rows)
    write_csv(intervals, AWARD_DIR / "q4_table_4_1_prediction_intervals.csv")
    return {
        "best_ablation": str(best_generalized["model"]),
        "best_training_reconstruction": str(ablation.iloc[0]["model"]),
        "target_interval_max_width": float(intervals["interval_half_width_mm"].max()),
    }


def q5_award() -> dict[str, object]:
    combo = pd.read_csv(TABLE_DIR / "q5_variable_combination_cv.csv")
    combo["gbdt_stability_rank"] = combo["gbdt_displacement_RMSE"].rank(method="min") + combo["gbdt_velocity_RMSE"].rank(method="min")
    combo["elasticnet_stability_rank"] = combo["elasticnet_displacement_RMSE"].rank(method="min") + combo["elasticnet_velocity_RMSE"].rank(method="min")
    combo["consensus_rank_score"] = 0.65 * combo["gbdt_stability_rank"] + 0.35 * combo["elasticnet_stability_rank"]
    combo = combo.sort_values("consensus_rank_score")
    write_csv(combo, AWARD_DIR / "q5_variable_selection_stability.csv")

    pred = pd.read_csv(TABLE_DIR / "q5_best_model_predictions.csv")
    pred["时间"] = pd.to_datetime(pred["时间"])
    inv = inverse_velocity_forecast(pred["时间"], pred["模型预测速度_mm_h"].to_numpy(float), stage=pred["阶段"].to_numpy(int), window_steps=144, min_velocity=0.3, min_r2=0.55)
    write_csv(inv, AWARD_DIR / "q5_inverse_velocity_warning.csv")
    _save_q5_warning_timeline(pred, inv)
    return {"best_combo": str(combo.iloc[0]["variables"]), "inverse_warning_count": int(len(inv))}


def _save_q5_warning_timeline(pred: pd.DataFrame, inv: pd.DataFrame) -> None:
    thresholds = pd.read_csv(TABLE_DIR / "q5_warning_thresholds.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(pred["时间"], pred["表面位移_mm"], color="#1f77b4", lw=1.0, label="观测位移")
    axes[0].plot(pred["时间"], pred["模型积分位移_mm"], color="#d62728", lw=1.0, label="模型位移")
    axes[0].set_ylabel("表面位移/mm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].plot(pred["时间"], pred["模型预测速度_mm_h"], color="#2ca02c", lw=0.9, label="预测速度")
    level_labels = {"关注": "关注", "预警": "预警", "严重预警": "严重预警"}
    for _, row in thresholds.iterrows():
        level = level_labels.get(str(row["level"]), str(row["level"]))
        axes[1].axhline(row["velocity_threshold_mm_h"], ls="--", lw=1, label=f"{level} {row['velocity_threshold_mm_h']:.2f}")
    if not inv.empty:
            axes[1].scatter(pd.to_datetime(inv["time"]), inv["velocity_mm_h"], color="#d62728", s=16, label="逆速度预警点")
    axes[1].set_ylabel("速度/(mm/h)")
    axes[1].set_xlabel("时间")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q5_warning_timeline.png", dpi=220)
    plt.close(fig)


def build_report(results: dict[str, object]) -> None:
    baseline_path = MODEL_DIR / "all_model_summaries.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
    diff_stat = git_diff_stat(PROJECT_DIR)
    report = [
        "# 特等奖增强模型质量报告",
        "",
        "## Baseline 摘要",
        f"- Q1 Huber CV MAE: {baseline.get('q1', {}).get('cv_mean', {}).get('MAE_mm', 'NA')}",
        f"- Q2 baseline transition nodes: {baseline.get('q2', {}).get('break_serial_numbers', 'NA')}",
        f"- Q3 baseline GBDT R2: {baseline.get('q3', {}).get('gbdt', {}).get('R2', 'NA')}",
        f"- Q4 baseline table values: {baseline.get('q4', {}).get('target_predictions', 'NA')}",
        f"- Q5 baseline best combo: {baseline.get('q5', {}).get('best_combo', 'NA')}",
        "",
        "## Award Audit 结论",
        f"- Q1 model comparison winner: {results['q1']['best_model']}; bootstrap CI NaN count: {results['q1']['ci_nan_count']}.",
        f"- Q2 composite transition recommendation: {results['q2']['recommended']}.",
        f"- Q3 best validation model: {results['q3']['best_model']}; most stable factor: {results['q3']['top_factor']}.",
        f"- Q4 strongest generalized ablation model: {results['q4']['best_ablation']}; training reconstruction winner: {results['q4']['best_training_reconstruction']}; max 95% half-width: {results['q4']['target_interval_max_width']:.3f} mm.",
        f"- Q5 consensus best combo: {results['q5']['best_combo']}; inverse-velocity warning records: {results['q5']['inverse_warning_count']}.",
        "",
        "## Academic Method Sources",
        "- Changepoint detection: Killick, Fearnhead and Eckley, PELT algorithm, JASA 2012, https://doi.org/10.1080/01621459.2012.737745",
        "- Robust outlier handling: Hampel identifier with median/MAD filtering, https://blogs.sas.com/content/iml/2021/06/01/hampel-filter-robust-outliers.html",
        "- Landslide inverse-velocity warning: Fukuzono/modified inverse-velocity landslide failure-time literature, e.g. https://www.sciencedirect.com/science/article/pii/S001379521931751X",
        "- Displacement prediction review: physics-based and data-driven landslide displacement prediction review, https://www.sciencedirect.com/science/article/pii/S0012825224002769",
        "",
        "## Git Diff Stat",
        "```",
        diff_stat,
        "```",
        "",
        "## Generated Award Files",
    ]
    for path in sorted(AWARD_DIR.iterdir()):
        if path.is_file():
            report.append(f"- `{path.name}`")
    write_text("\n".join(report) + "\n", AWARD_DIR / "model_quality_report.md")


def run() -> dict[str, object]:
    ensure_output_dirs()
    AWARD_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "q1": q1_award(),
        "q2": q2_award(),
        "q3": q3_award(),
        "q4": q4_award(),
        "q5": q5_award(),
    }
    write_json(results, AWARD_DIR / "award_audit_summary.json")
    build_report(results)
    return results
