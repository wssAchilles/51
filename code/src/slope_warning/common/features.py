from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from slope_warning.common.preprocessing import rolling_mean, rolling_sum


BASE_VARIABLES_Q5 = ["降雨量_mm", "孔隙水压力_kPa", "微震事件数", "干湿入渗系数", "爆破点距离_m", "单段最大药量_kg"]


def blast_impulse(distance: np.ndarray, charge: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    charge = np.asarray(charge, dtype=float)
    return np.where(np.isfinite(distance), np.nan_to_num(charge, nan=0.0) / ((distance + 0.5) ** 2), 0.0)


def exp_decay(values: np.ndarray, half_life_steps: float) -> np.ndarray:
    alpha = 1.0 - np.exp(-np.log(2.0) / half_life_steps)
    out = np.zeros_like(np.asarray(values, dtype=float))
    for i, value in enumerate(values):
        out[i] = value + (1.0 - alpha) * (out[i - 1] if i else 0.0)
    return out


def make_disturbance_features(df: pd.DataFrame, include: list[str] | None = None, stage: np.ndarray | None = None, tau: np.ndarray | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    include = include or [c for c in BASE_VARIABLES_Q5 if c in df.columns]
    features: dict[str, np.ndarray] = {}
    groups: dict[str, list[str]] = {}

    if stage is not None:
        features["stage_2"] = (stage == 2).astype(float)
        features["stage_3"] = (stage == 3).astype(float)
        groups["stage"] = ["stage_2", "stage_3"]
    if tau is not None:
        features["tau"] = tau.astype(float)
        features["tau2"] = tau.astype(float) ** 2
        groups["tau"] = ["tau", "tau2"]

    if "降雨量_mm" in include and "降雨量_mm" in df.columns:
        rain = df["降雨量_mm"].fillna(0.0).clip(lower=0).to_numpy(float)
        names = []
        for name, values in {
            "rain": rain,
            "rain_1h": rolling_sum(rain, 6),
            "rain_6h": rolling_sum(rain, 36),
            "rain_24h": rolling_sum(rain, 144),
        }.items():
            features[name] = values
            names.append(name)
        groups["rain"] = names

    if "孔隙水压力_kPa" in include and "孔隙水压力_kPa" in df.columns:
        pore = df["孔隙水压力_kPa"].interpolate(limit_direction="both").to_numpy(float)
        names = []
        for name, values in {
            "pore": pore,
            "pore_diff": np.r_[0.0, np.diff(pore)],
            "pore_6h_mean": rolling_mean(pore, 36),
        }.items():
            features[name] = values
            names.append(name)
        groups["pore"] = names

    if "微震事件数" in include and "微震事件数" in df.columns:
        micro = df["微震事件数"].fillna(0.0).clip(lower=0).to_numpy(float)
        names = []
        for name, values in {
            "micro": micro,
            "micro_1h": rolling_sum(micro, 6),
            "micro_6h": rolling_sum(micro, 36),
        }.items():
            features[name] = values
            names.append(name)
        groups["micro"] = names

    if "干湿入渗系数" in include and "干湿入渗系数" in df.columns:
        infil = df["干湿入渗系数"].interpolate(limit_direction="both").to_numpy(float)
        names = []
        for name, values in {
            "infiltration": infil,
            "infiltration_diff": np.r_[0.0, np.diff(infil)],
            "infiltration_6h_mean": rolling_mean(infil, 36),
        }.items():
            features[name] = values
            names.append(name)
        groups["infiltration"] = names

    has_dist = "爆破点距离_m" in include and "爆破点距离_m" in df.columns
    has_charge = "单段最大药量_kg" in include and "单段最大药量_kg" in df.columns
    if has_dist:
        dist = df["爆破点距离_m"].to_numpy(float)
        features["blast_flag"] = np.isfinite(dist).astype(float)
        features["blast_inv_distance"] = np.where(np.isfinite(dist), 1.0 / (dist + 0.5), 0.0)
        groups["blast_distance"] = ["blast_flag", "blast_inv_distance"]
    if has_charge:
        charge = df["单段最大药量_kg"].fillna(0.0).to_numpy(float)
        features["blast_charge"] = charge
        groups["blast_charge"] = ["blast_charge"]
    if has_dist and has_charge:
        dist = df["爆破点距离_m"].to_numpy(float)
        charge = df["单段最大药量_kg"].fillna(0.0).to_numpy(float)
        impulse = blast_impulse(dist, charge)
        features["blast_impulse"] = impulse
        features["blast_impulse_6h"] = exp_decay(impulse, 36)
        features["blast_impulse_24h"] = exp_decay(impulse, 144)
        groups["blast_interaction"] = ["blast_impulse", "blast_impulse_6h", "blast_impulse_24h"]

    if "rain_24h" in features and "pore" in features:
        features["rain24_pore"] = features["rain_24h"] * features["pore"]
        groups.setdefault("interactions", []).append("rain24_pore")
    if "rain_24h" in features and "infiltration_6h_mean" in features:
        features["rain24_infiltration"] = features["rain_24h"] * features["infiltration_6h_mean"]
        groups.setdefault("interactions", []).append("rain24_infiltration")
    if "pore" in features and "micro_6h" in features:
        features["pore_micro6"] = features["pore"] * features["micro_6h"]
        groups.setdefault("interactions", []).append("pore_micro6")
    if "blast_impulse_6h" in features and "micro_6h" in features:
        features["blast_micro6"] = features["blast_impulse_6h"] * features["micro_6h"]
        groups.setdefault("interactions", []).append("blast_micro6")

    feature_df = pd.DataFrame(features, index=df.index)
    return feature_df.replace([np.inf, -np.inf], 0.0).fillna(0.0), groups


def five_variable_combinations() -> list[tuple[str, ...]]:
    return list(combinations(BASE_VARIABLES_Q5, 5))

