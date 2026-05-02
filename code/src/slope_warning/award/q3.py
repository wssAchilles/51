from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slope_warning.common.io import read_excel, write_csv
from slope_warning.common.metrics import time_block_folds
from slope_warning.common.plotting import configure_chinese_fonts
from slope_warning.common.preprocessing import hampel_flags, kalman_smooth_fill, sparse_outlier_flags, sparse_series_fill
from slope_warning.config import ATTACHMENTS, AUDIT_CONFIG, AWARD_DIR, TABLE_DIR
from slope_warning.questions import q3_fusion


configure_chinese_fonts()


def _model_factories() -> dict[str, object]:
    return {
        "ElasticNet": lambda: make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=10000, random_state=AUDIT_CONFIG.rng_seed),
        ),
        "GBDT": lambda: GradientBoostingRegressor(
            random_state=AUDIT_CONFIG.rng_seed,
            n_estimators=420,
            learning_rate=0.035,
            max_depth=3,
            min_samples_leaf=18,
            subsample=0.85,
        ),
        "HistGBDT": lambda: HistGradientBoostingRegressor(
            random_state=AUDIT_CONFIG.rng_seed,
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=23,
            l2_regularization=0.03,
        ),
    }


def _save_variable_panels(raw: pd.DataFrame, proc: pd.DataFrame, flags: dict[str, np.ndarray]) -> None:
    variables = ["rain", "pore", "micro", "deep", "surface"]
    labels = {"rain": "降雨量", "pore": "孔隙水压力", "micro": "微震事件数", "deep": "深部位移", "surface": "表面位移"}
    fig, axes = plt.subplots(len(variables), 1, figsize=(11, 10), sharex=True)
    x = raw["编号"]
    for ax, var in zip(axes, variables):
        ax.plot(x, raw[var], color="#9e9e9e", lw=0.7, alpha=0.75, label="原始值")
        ax.plot(x, proc[var], color="#1f77b4", lw=0.8, label="预处理后")
        idx = np.where(flags[var])[0]
        if len(idx):
            ax.scatter(x.iloc[idx], proc[var].iloc[idx], color="#c03a3a", s=12, label="异常点")
        ax.set_ylabel(labels[var])
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("编号")
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q3_variable_trace_panels.png", dpi=240)
    plt.close(fig)


def _save_missing_heatmap(raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    cols = list(q3_fusion.TRAIN_COLUMNS.keys())
    missing = raw[cols].isna().T
    missing.index = ["降雨量", "孔隙水压力", "微震事件数", "深部位移", "表面位移"]
    sns.heatmap(missing, cbar=False, ax=ax, cmap="Blues")
    ax.set_title("问题3：缺失值分布")
    ax.set_xlabel("样本序号")
    ax.set_ylabel("变量")
    fig.tight_layout()
    fig.savefig(AWARD_DIR / "q3_missing_heatmap.png", dpi=240)
    plt.close(fig)


def _save_partial_dependence(features: pd.DataFrame, model: object, path: Path) -> None:
    top_features = ["deep", "pore", "rain_24h", "micro_6h"]
    labels = {"deep": "深部位移", "pore": "孔隙水压力", "rain_24h": "24h累计降雨", "micro_6h": "6h微震累计"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for ax, feature in zip(axes, top_features):
        grid = np.quantile(features[feature], np.linspace(0.05, 0.95, 40))
        preds = []
        sample = features.sample(n=min(2000, len(features)), random_state=AUDIT_CONFIG.rng_seed)
        for value in grid:
            changed = sample.copy()
            changed[feature] = value
            preds.append(float(np.mean(model.predict(changed))))
        ax.plot(grid, preds, color="#2b6da8", lw=1.5)
        ax.set_xlabel(labels.get(feature, feature))
        ax.set_ylabel("表面位移预测值/mm")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=240)
    plt.close(fig)


def _fill_diagnostics(train: pd.DataFrame, flags: dict[str, np.ndarray], proc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in ["pore", "deep", "surface"]:
        raw = train[var].to_numpy(float)
        filled = kalman_smooth_fill(raw, nonnegative=var in {"deep", "surface"})
        rows.append(
            {
                "variable": var,
                "missing_count_raw": int(np.isnan(raw).sum()),
                "missing_count_after_fill": int(np.isnan(filled).sum()),
                "min_after_preprocess": float(proc[var].min()),
                "max_after_preprocess": float(proc[var].max()),
                "outlier_count_threshold_4.5": int(flags[var].sum()),
            }
        )
    return pd.DataFrame(rows)


def run() -> dict[str, object]:
    train_raw = read_excel(ATTACHMENTS["q3"], sheet_name="训练集")
    train = q3_fusion._standardize(train_raw, q3_fusion.TRAIN_COLUMNS)
    proc, flags, _ = q3_fusion._preprocess(train, has_surface=True)
    features, groups = q3_fusion._make_features(proc)
    y = proc["surface"].to_numpy(float)
    split = int(len(y) * 0.8)
    x_fit, x_val = features.iloc[:split], features.iloc[split:]
    y_fit, y_val = y[:split], y[split:]

    sensitivity_rows = []
    for threshold in AUDIT_CONFIG.q3_continuous_threshold_grid:
        row = {"continuous_hampel_threshold": threshold}
        for var in ["pore", "deep", "surface"]:
            filled = kalman_smooth_fill(train[var].to_numpy(float), nonnegative=var in {"deep", "surface"})
            var_flags, _, _ = hampel_flags(filled, window=73, threshold=threshold, center=True)
            row[f"{var}_outliers"] = int(var_flags.sum())
        for quantile in AUDIT_CONFIG.q3_sparse_quantile_grid:
            rain = sparse_series_fill(train["rain"].to_numpy(float))
            micro = sparse_series_fill(train["micro"].to_numpy(float), integer=True)
            row[f"rain_outliers_q{quantile}"] = int(sparse_outlier_flags(rain, quantile=quantile, min_threshold=15.0).sum())
            row[f"micro_outliers_q{quantile}"] = int(sparse_outlier_flags(micro, quantile=quantile, min_threshold=6.0).sum())
        sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    write_csv(sensitivity, AWARD_DIR / "q3_anomaly_sensitivity.csv")
    write_csv(_fill_diagnostics(train, flags, proc), AWARD_DIR / "q3_continuous_fill_diagnostics.csv")

    model_rows = []
    fitted_models = {}
    for name, factory in _model_factories().items():
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

    generator = np.random.default_rng(AUDIT_CONFIG.rng_seed)
    contrib_rows = []
    for fold_id, test in enumerate(time_block_folds(len(y), k=5), start=1):
        train_idx = np.setdiff1d(np.arange(len(y)), test, assume_unique=True)
        for model_name in ["ElasticNet", "GBDT", "HistGBDT"]:
            model = _model_factories()[model_name]()
            model.fit(features.iloc[train_idx], y[train_idx])
            base = mean_squared_error(y[test], model.predict(features.iloc[test])) ** 0.5
            for factor, columns in groups.items():
                x_perm = features.iloc[test].copy()
                existing = [col for col in columns if col in x_perm.columns]
                order = generator.permutation(len(x_perm))
                x_perm.loc[:, existing] = x_perm[existing].to_numpy()[order]
                score = mean_squared_error(y[test], model.predict(x_perm)) ** 0.5
                contrib_rows.append({"fold": fold_id, "model": model_name, "factor": factor, "RMSE_increase": score - base})
    stability = (
        pd.DataFrame(contrib_rows)
        .groupby(["model", "factor"], as_index=False)["RMSE_increase"]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(["model", "mean"], ascending=[True, False])
    )
    write_csv(stability, AWARD_DIR / "q3_variable_contribution_stability.csv")

    common_path = TABLE_DIR / "q3_table_3_2_common_anomalies.csv"
    if common_path.exists():
        common = pd.read_csv(common_path)
        event_summary = (
            common.assign(异常变量数=common["共同异常点处的异常变量"].astype(str).str.len())
            .groupby("异常变量数", as_index=False)
            .size()
            .rename(columns={"size": "事件数"})
        )
    else:
        event_summary = pd.DataFrame(columns=["异常变量数", "事件数"])
    write_csv(event_summary, AWARD_DIR / "q3_common_anomaly_event_summary.csv")

    _save_variable_panels(train, proc, flags)
    _save_missing_heatmap(train_raw)
    _save_partial_dependence(features, fitted_models["GBDT"], AWARD_DIR / "q3_partial_dependence.png")
    return {
        "best_model": str(model_comparison.iloc[0]["model"]),
        "top_factor": str(stability.groupby("factor")["mean"].mean().sort_values(ascending=False).index[0]),
        "post_fill_missing_count": int(proc.isna().sum().sum()),
        "common_event_classes": int(len(event_summary)),
    }
