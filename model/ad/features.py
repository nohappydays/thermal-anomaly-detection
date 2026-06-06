"""Feature engineering for anomaly-detection baselines.

Two kinds of features are provided:
  * `session_features(df)` — one feature vector per session (used by classical
    baselines that ingest a single point per example: IsolationForest, OCSVM).
  * `window_features(df, window_s)` — sliding-window feature matrix per session
    (used for per-frame anomaly scoring).
  * `time_series(df)` — raw normalized series shaped (T, C) for the LSTM-VAE.

All three operate on the CSV/XLSX session schema:
  columns: t_s, T_max, T_min, T_range, condition, session_id
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


SESSION_FEATURE_NAMES = (
    "T_max_mean", "T_max_std", "T_max_max", "T_max_p95", "T_max_p05",
    "T_min_mean", "T_min_std",
    "T_range_mean", "T_range_max",
    "slope_first5min_C_per_min",
    "slope_last5min_C_per_min",
    "slope_full_C_per_min",
    "dT_max_dt_p95",
    "autocorr_lag1",
    "thermal_time_constant_s",
    "frac_above_38C", "frac_above_42C", "frac_above_46C",
    "time_to_38C_s", "time_to_42C_s",
)


def _safe_slope(t: NDArray, y: NDArray) -> float:
    if len(t) < 2 or np.allclose(t, t[0]):
        return 0.0
    return float(np.polyfit(t, y, 1)[0])


def session_features(df: pd.DataFrame) -> dict[str, float]:
    """Scalar features summarising one session.

    Returns a dict; keys match SESSION_FEATURE_NAMES in order. Missing
    derived features (e.g. time_to_46C if the session never reaches it)
    are filled with -1.0 to keep the vector fixed-length.
    """
    if "t_s" not in df.columns or "T_max" not in df.columns:
        raise KeyError("session_features requires `t_s` and `T_max`")

    t = df["t_s"].to_numpy(dtype=np.float64)
    tmax = df["T_max"].to_numpy(dtype=np.float64)
    tmin = df["T_min"].to_numpy(dtype=np.float64) if "T_min" in df.columns else np.full_like(tmax, np.nan)
    trange = (tmax - tmin) if not np.isnan(tmin).all() else np.zeros_like(tmax)

    feats: dict[str, float] = {}
    feats["T_max_mean"] = float(np.nanmean(tmax))
    feats["T_max_std"] = float(np.nanstd(tmax))
    feats["T_max_max"] = float(np.nanmax(tmax))
    feats["T_max_p95"] = float(np.nanpercentile(tmax, 95))
    feats["T_max_p05"] = float(np.nanpercentile(tmax, 5))
    feats["T_min_mean"] = float(np.nanmean(tmin)) if not np.isnan(tmin).all() else -1.0
    feats["T_min_std"] = float(np.nanstd(tmin)) if not np.isnan(tmin).all() else -1.0
    feats["T_range_mean"] = float(np.nanmean(trange))
    feats["T_range_max"] = float(np.nanmax(trange))

    # Slopes (°C / minute)
    head = t <= (t[0] + 300)
    tail = t >= (t[-1] - 300)
    feats["slope_first5min_C_per_min"] = _safe_slope(t[head], tmax[head]) * 60
    feats["slope_last5min_C_per_min"]  = _safe_slope(t[tail], tmax[tail]) * 60
    feats["slope_full_C_per_min"]      = _safe_slope(t, tmax) * 60

    # Derivative summary
    dt = np.diff(t); dt = np.where(dt == 0, 1, dt)
    d = np.diff(tmax) / dt
    feats["dT_max_dt_p95"] = float(np.percentile(np.abs(d), 95)) if d.size else 0.0

    # Autocorrelation at lag 1
    x = tmax - np.nanmean(tmax)
    denom = float(np.sum(x * x))
    feats["autocorr_lag1"] = float(np.sum(x[:-1] * x[1:]) / denom) if denom > 0 else 0.0

    # Thermal time constant — time to reach 63.2% of the dynamic range
    t0_val, t_end_val = tmax[0], tmax[-1]
    if abs(t_end_val - t0_val) > 0.5:
        target = t0_val + 0.632 * (t_end_val - t0_val)
        sign = np.sign(t_end_val - t0_val)
        crossed = np.where((tmax - target) * sign >= 0)[0]
        feats["thermal_time_constant_s"] = float(t[crossed[0]] - t[0]) if crossed.size else -1.0
    else:
        feats["thermal_time_constant_s"] = -1.0

    # Hot-area fractions and time-to-threshold
    for thr in (38.0, 42.0, 46.0):
        feats[f"frac_above_{int(thr)}C"] = float(np.mean(tmax >= thr))
    for thr in (38.0, 42.0):
        above = np.where(tmax >= thr)[0]
        feats[f"time_to_{int(thr)}C_s"] = float(t[above[0]] - t[0]) if above.size else -1.0

    return feats


def session_features_matrix(sessions: list[pd.DataFrame]) -> tuple[NDArray, list[str]]:
    """Stack `session_features` outputs into a (N, D) feature matrix."""
    rows = [session_features(s) for s in sessions]
    keys = list(SESSION_FEATURE_NAMES)
    # Keep feature order stable if derived features are added later.
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    matrix = np.array([[r.get(k, 0.0) for k in keys] for r in rows], dtype=np.float64)
    return matrix, keys


def window_features(df: pd.DataFrame, window_s: int = 60, step_s: int = 5) -> tuple[NDArray, NDArray]:
    """Slide a window across the session, computing per-window features.

    Returns (features (n_windows, D), centre_timestamps_s (n_windows,)).
    """
    t = df["t_s"].to_numpy(dtype=np.float64)
    tmax = df["T_max"].to_numpy(dtype=np.float64)
    if "T_min" in df.columns:
        tmin = df["T_min"].to_numpy(dtype=np.float64)
    else:
        tmin = np.zeros_like(tmax)

    centres = []
    feats = []
    n = len(t)
    half = window_s // 2

    for c in range(half, n - half, step_s):
        s = slice(c - half, c + half)
        chunk = tmax[s]
        if len(chunk) < 4:
            continue
        f = [
            float(np.mean(chunk)),
            float(np.std(chunk)),
            float(np.max(chunk)),
            float(np.min(chunk)),
            float(np.max(chunk) - np.min(chunk)),
            float(np.mean(np.diff(chunk))),    # local slope proxy
            float(np.std(np.diff(chunk))),     # local roughness
            float(np.mean(tmin[s])),
        ]
        feats.append(f)
        centres.append(t[c])

    return np.array(feats, dtype=np.float64), np.array(centres, dtype=np.float64)


def time_series(df: pd.DataFrame, normalize: bool = True) -> NDArray:
    """Return a (T, C) float32 array with channels [T_max, T_min, T_range].

    If `normalize`, each channel is z-scored using global constants
    (chosen to match the realistic range 20–55 °C → roughly [0, 1]).
    """
    tmax = df["T_max"].to_numpy(dtype=np.float32)
    tmin = df["T_min"].to_numpy(dtype=np.float32) if "T_min" in df.columns else np.zeros_like(tmax)
    trng = tmax - tmin
    arr = np.stack([tmax, tmin, trng], axis=1)
    if normalize:
        # Global affine: ~ (T - 30) / 20 → most data lands in [-0.5, +1.25]
        arr[:, 0] = (arr[:, 0] - 30.0) / 20.0
        arr[:, 1] = (arr[:, 1] - 23.0) / 8.0
        arr[:, 2] = arr[:, 2] / 20.0
    return arr.astype(np.float32)
