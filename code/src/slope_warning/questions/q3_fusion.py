from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from slope_warning.common.io import read_excel, write_csv, write_json
from slope_warning.common.plotting import save_scatter_plot
from slope_warning.common.preprocessing import (
    hampel_flags,
    kalman_smooth_fill,
    rolling_sum,
    sparse_outlier_flags,
    sparse_series_fill,
)
from slope_warning.config import ATTACHMENTS, FIGURE_DIR, MODEL_DIR, TABLE_DIR


TRAIN_COLUMNS = {
    "a:降雨量_mm": "rain",
    "b:孔隙水压力_kPa": "pore",
    "c:微震事件数": "micro",
    "d:深部位移_mm": "deep",
    "e:表面位移_mm": "surface",
}
EXPERIMENT_COLUMNS = {
    "降雨量_mm": "rain",
    "孔隙水压力_kPa": "pore",
    "微震事件数": "micro",
    "深部位移_mm": "deep",
    "表面位移_mm": "surface",
}
LETTERS = {"rain": "a", "pore": "b", "micro": "c", "deep": "d", "surface": "e"}
DISPLAY = {
    "rain": "a：降雨量",
    "pore": "b：孔隙水压力",
    "micro": "c：微震事件数",
    "deep": "d：深部位移",
    "surface": "e：表面位移",
}


def _standardize(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame({"编号": df["编号"].to_numpy()})
    for old, new in mapping.items():
        out[new] = df[old].to_numpy(float)
    return out


def _continuous_clean(values: np.ndarray, nonnegative: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled = kalman_smooth_fill(values, nonnegative=nonnegative)
    flags, med, score = hampel_flags(filled, window=73, threshold=4.5, center=True)
    cleaned = filled.copy()
    cleaned[flags] = med[flags]
    if nonnegative:
        cleaned = np.maximum(cleaned, 0.0)
    return cleaned, flags, score


def _preprocess(df: pd.DataFrame, has_surface: bool) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    out = pd.DataFrame({"编号": df["编号"].astype(int)})
    flags: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}

    rain = sparse_series_fill(df["rain"].to_numpy(float), integer=False)
    rain_flags = sparse_outlier_flags(rain, quantile=0.997, min_threshold=15.0)
    rain_clean = rain.copy()
    rain_clean[rain_flags] = np.nanmedian(rain_clean[~rain_flags]) if np.any(~rain_flags) else 0.0
    out["rain"] = np.maximum(rain_clean, 0.0)
    flags["rain"] = rain_flags
    scores["rain"] = np.where(rain_flags, 1.0, 0.0)

    pore, pore_flags, pore_score = _continuous_clean(df["pore"].to_numpy(float))
    out["pore"] = pore
    flags["pore"] = pore_flags
    scores["pore"] = pore_score

    micro = sparse_series_fill(df["micro"].to_numpy(float), integer=True)
    micro_flags = sparse_outlier_flags(micro, quantile=0.997, min_threshold=6.0)
    micro_clean = micro.copy()
    if micro_flags.any():
        micro_clean[micro_flags] = np.nanmedian(micro_clean[~micro_flags])
    out["micro"] = np.maximum(np.rint(micro_clean), 0.0)
    flags["micro"] = micro_flags
    scores["micro"] = np.where(micro_flags, 1.0, 0.0)

    deep, deep_flags, deep_score = _continuous_clean(df["deep"].to_numpy(float), nonnegative=True)
    out["deep"] = deep
    flags["deep"] = deep_flags
    scores["deep"] = deep_score

    if has_surface:
        surface, surface_flags, surface_score = _continuous_clean(df["surface"].to_numpy(float), nonnegative=True)
        out["surface"] = surface
        flags["surface"] = surface_flags
        scores["surface"] = surface_score
    else:
        out["surface"] = np.nan
        flags["surface"] = np.zeros(len(df), dtype=bool)
        scores["surface"] = np.zeros(len(df), dtype=float)

    return out, flags, scores


def _make_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rain = df["rain"].to_numpy(float)
    pore = df["pore"].to_numpy(float)
    micro = df["micro"].to_numpy(float)
    deep = df["deep"].to_numpy(float)
    data = {
        "rain": rain,
        "rain_1h": rolling_sum(rain, 6),
        "rain_6h": rolling_sum(rain, 36),
        "rain_24h": rolling_sum(rain, 144),
        "pore": pore,
        "pore_diff": np.r_[0.0, np.diff(pore)],
        "micro": micro,
        "micro_6h": rolling_sum(micro, 36),
        "deep": deep,
        "deep_diff": np.r_[0.0, np.diff(deep)],
        "rain24_pore": rolling_sum(rain, 144) * pore,
        "pore_deep": pore * deep,
        "micro6_deep": rolling_sum(micro, 36) * deep,
    }
    groups = {
        "rain": ["rain", "rain_1h", "rain_6h", "rain_24h", "rain24_pore"],
        "pore": ["pore", "pore_diff", "rain24_pore", "pore_deep"],
        "micro": ["micro", "micro_6h", "micro6_deep"],
        "deep": ["deep", "deep_diff", "pore_deep", "micro6_deep"],
    }
    return pd.DataFrame(data, index=df.index), groups


def _grouped_permutation_importance(model: object, x_val: pd.DataFrame, y_val: np.ndarray, groups: dict[str, list[str]]) -> pd.DataFrame:
    rng = np.random.default_rng(20260501)
    baseline = mean_squared_error(y_val, model.predict(x_val)) ** 0.5
    rows = []
    for group, columns in groups.items():
        deltas = []
        existing = [c for c in columns if c in x_val.columns]
        for _ in range(20):
            x_perm = x_val.copy()
            order = rng.permutation(len(x_perm))
            x_perm.loc[:, existing] = x_perm[existing].to_numpy()[order]
            score = mean_squared_error(y_val, model.predict(x_perm)) ** 0.5
            deltas.append(score - baseline)
        rows.append({"factor": group, "baseline_RMSE": baseline, "permutation_RMSE_increase": float(np.mean(deltas))})
    return pd.DataFrame(rows).sort_values("permutation_RMSE_increase", ascending=False)


def run() -> dict[str, object]:
    train_raw = read_excel(ATTACHMENTS["q3"], sheet_name="训练集")
    exp_raw = read_excel(ATTACHMENTS["q3"], sheet_name="实验集")
    train = _standardize(train_raw, TRAIN_COLUMNS)
    exp = _standardize(exp_raw, EXPERIMENT_COLUMNS)

    train_proc, flags, _ = _preprocess(train, has_surface=True)
    exp_proc, _, _ = _preprocess(exp, has_surface=False)

    anomaly_counts = pd.DataFrame(
        {
            "数据集变量": [DISPLAY[v] for v in ["rain", "pore", "micro", "deep", "surface"]] + ["总数"],
            "异常点数量": [int(flags[v].sum()) for v in ["rain", "pore", "micro", "deep", "surface"]]
            + [int(sum(flags[v].sum() for v in ["rain", "pore", "micro", "deep", "surface"]))],
        }
    )
    flag_matrix = np.column_stack([flags[v] for v in ["rain", "pore", "micro", "deep", "surface"]])
    common_idx = np.where(flag_matrix.sum(axis=1) >= 2)[0]
    common_rows = []
    for order, idx in enumerate(common_idx, start=1):
        variables = "".join(LETTERS[v] for v in ["rain", "pore", "micro", "deep", "surface"] if flags[v][idx])
        common_rows.append({"序号": order, "时间点对应编号": int(train_proc.loc[idx, "编号"]), "共同异常点处的异常变量": variables})
    common_df = pd.DataFrame(common_rows)

    x_train, groups = _make_features(train_proc)
    y_train = train_proc["surface"].to_numpy(float)
    split = int(len(train_proc) * 0.8)
    x_fit, x_val = x_train.iloc[:split], x_train.iloc[split:]
    y_fit, y_val = y_train[:split], y_train[split:]

    elastic = make_pipeline(
        StandardScaler(),
        ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 2, 30), cv=TimeSeriesSplit(n_splits=5), max_iter=10000),
    )
    elastic.fit(x_fit, y_fit)
    elastic_pred = elastic.predict(x_val)

    gbdt = GradientBoostingRegressor(
        random_state=20260501,
        n_estimators=450,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=18,
        subsample=0.85,
    )
    gbdt.fit(x_fit, y_fit)
    gbdt_pred = gbdt.predict(x_val)

    gbdt.fit(x_train, y_train)
    exp_features, _ = _make_features(exp_proc)
    exp_pred = gbdt.predict(exp_features)
    exp_pred = np.clip(exp_pred, y_train.min(), y_train.max())
    exp_result = pd.DataFrame({"编号": exp_proc["编号"], "表面位移预测值_mm": exp_pred})

    perm = _grouped_permutation_importance(gbdt, x_val, y_val, groups)
    feature_importance = pd.DataFrame({"feature": x_train.columns, "gbdt_importance": gbdt.feature_importances_}).sort_values("gbdt_importance", ascending=False)
    elastic_model = elastic.named_steps["elasticnetcv"]
    elastic_coef = pd.DataFrame({"feature": x_train.columns, "elasticnet_standardized_coef": elastic_model.coef_}).sort_values(
        "elasticnet_standardized_coef", key=lambda s: np.abs(s), ascending=False
    )

    validation = {
        "elasticnet": {
            "RMSE_mm": float(mean_squared_error(y_val, elastic_pred) ** 0.5),
            "MAE_mm": float(mean_absolute_error(y_val, elastic_pred)),
            "R2": float(r2_score(y_val, elastic_pred)),
        },
        "gbdt": {
            "RMSE_mm": float(mean_squared_error(y_val, gbdt_pred) ** 0.5),
            "MAE_mm": float(mean_absolute_error(y_val, gbdt_pred)),
            "R2": float(r2_score(y_val, gbdt_pred)),
        },
        "common_anomaly_count": int(len(common_df)),
        "single_variable_anomaly_counts": anomaly_counts.to_dict(orient="records"),
        "top_contribution_factor": str(perm.iloc[0]["factor"]) if not perm.empty else "",
    }

    write_csv(train_proc, TABLE_DIR / "q3_preprocessed_training.csv")
    write_csv(exp_proc.drop(columns=["surface"]), TABLE_DIR / "q3_preprocessed_experiment_features.csv")
    write_csv(anomaly_counts, TABLE_DIR / "q3_table_3_1_single_variable_anomalies.csv")
    write_csv(common_df, TABLE_DIR / "q3_table_3_2_common_anomalies.csv")
    write_csv(exp_result, TABLE_DIR / "q3_experiment_surface_predictions.csv")
    write_csv(perm, TABLE_DIR / "q3_grouped_contribution_permutation.csv")
    write_csv(feature_importance, TABLE_DIR / "q3_gbdt_feature_importance.csv")
    write_csv(elastic_coef, TABLE_DIR / "q3_elasticnet_coefficients.csv")
    write_json(validation, MODEL_DIR / "q3_model_summary.json")
    save_scatter_plot(
        exp_result["编号"].to_numpy(),
        exp_result["表面位移预测值_mm"].to_numpy(),
        FIGURE_DIR / "q3_experiment_prediction_scatter.png",
        "问题3：实验集表面位移预测散点图",
        "编号",
        "表面位移预测值/mm",
    )
    return validation
