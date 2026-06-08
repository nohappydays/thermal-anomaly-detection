"""Synthetic TopInfrared-style sessions for smoke tests.

Why this exists
---------------
Synthetic sessions were used before the real lab data arrived. They now serve
as smoke-test data matching the recorded session format:

    columns : timestamp (datetime64[ns]) | t_s (float, seconds) |
              T_min (°C) | T_max (°C) | condition (str)
    length  : 1500 rows (25 min @ 1 Hz)
    phases  : baseline t∈[0, 300)   — laptop idle
              stress   t∈[300, 1200) — mprep.info/gpu running
              cooldown t∈[1200, 1500) — stressor stopped

Thermal model
-------------
Newton's law of cooling-style first-order step response. The chassis surface
temperature follows an exponential approach toward a steady-state set point:

    T_max(t) =   T_amb                                                  if t < t_stress
               T_amb + (T_peak - T_amb) * (1 - exp(-(t - t_stress) / τ))   otherwise

where:
  T_amb   ~ Normal(μ_amb, σ_amb)                  per-session ambient
  T_peak  ~ Normal(μ_peak, σ_peak)                steady-state hot surface temp
  τ       ~ Normal(μ_τ, σ_τ)  (clipped > 30 s)     time constant
  + 1/f noise + Gaussian noise per sample

Blocked sessions differ from normal in three ways simultaneously
(reflecting real airflow obstruction):
  1. higher T_peak  (less heat lost to airflow → equilibrium shifts up)
  2. longer τ        (slower convective transfer to ambient)
  3. slight floor lift on T_min (chassis as a whole runs warmer)

The parameter ranges were calibrated against the May 12 Plane exports.

Use the generator as a smoke-test or unit-test source only. Real TopInfrared
exports live in `data/csv/` and are used for the reported results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from . import (
    DEFAULT_SEED, SESSION_DURATION_S, BASELINE_END_S, STRESS_END_S, EXPECTED_LEN,
)

# Blocked sessions cool more slowly because airflow is obstructed.
COOLDOWN_TAU_FACTOR_NORMAL = 1.0
COOLDOWN_TAU_FACTOR_BLOCKED = 1.8

log = logging.getLogger("ad.synthetic")


@dataclass
class ThermalParams:
    """Parameters for one synthetic session."""
    T_amb_C: float          # ambient / chassis-idle temperature
    T_peak_C: float         # asymptotic steady-state under load
    tau_s: float            # thermal time constant
    T_min_floor_C: float    # the steady T_min — chassis edge / coolest visible pixel
    noise_sigma_C: float    # per-sample Gaussian noise on T_max
    pink_noise_amp_C: float # 1/f noise amplitude (drift over time)


# Calibrated against the May 12 Excel files
NORMAL_PARAM_DIST = {
    "T_amb_C":         (23.0, 1.5),    # μ, σ
    "T_peak_C":        (40.0, 1.5),    # CPU-only and CPU+GPU were 39.9 and 41.1
    "tau_s":           (200.0, 50.0),  # first-5-min slope implied τ ≈ 150–270 s
    "T_min_floor_C":   (23.0, 0.8),
    "noise_sigma_C":   (0.15, 0.05),
    "pink_noise_amp_C":(0.20, 0.10),
}

# Blocked-vent perturbations: shift the same distributions
BLOCKED_PARAM_SHIFT = {
    "T_amb_C":         0.0,           # ambient itself doesn't change
    "T_peak_C":       +8.0,           # ~8 °C hotter equilibrium with no airflow
    "tau_s":          +120.0,         # slower thermal response (~50 % longer)
    "T_min_floor_C":  +1.5,           # whole chassis is warmer
    "noise_sigma_C":   0.0,
    "pink_noise_amp_C":+0.05,         # slightly noisier (turbulent air gone)
}


def _sample_params(rng: np.random.Generator, condition: Literal["normal","blocked"]) -> ThermalParams:
    base = {k: rng.normal(mu, sig) for k, (mu, sig) in NORMAL_PARAM_DIST.items()}
    if condition == "blocked":
        for k, shift in BLOCKED_PARAM_SHIFT.items():
            base[k] += shift
    # Clip to physically plausible values
    base["tau_s"] = max(30.0, base["tau_s"])
    base["noise_sigma_C"] = max(0.05, base["noise_sigma_C"])
    base["pink_noise_amp_C"] = max(0.0, base["pink_noise_amp_C"])
    base["T_peak_C"] = max(base["T_amb_C"] + 5.0, base["T_peak_C"])
    return ThermalParams(**base)


def _pink_noise(n: int, amp: float, rng: np.random.Generator) -> np.ndarray:
    """Cheap 1/f noise via cumulative random-walk filter, normalized to ±amp."""
    if amp <= 0 or n < 2:
        return np.zeros(n, dtype=np.float64)
    white = rng.normal(0, 1, size=n)
    # FFT-domain 1/sqrt(f) filter
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spec = np.fft.rfft(white) / np.sqrt(freqs)
    pink = np.fft.irfft(spec, n=n).real
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * amp
    return pink


def generate_session(
    session_id: str,
    condition: Literal["normal", "blocked"],
    seed: int,
    start_time: datetime | None = None,
    params: Optional[ThermalParams] = None,
) -> tuple[pd.DataFrame, ThermalParams]:
    """Generate one synthetic 25-minute session at 1 Hz.

    Returns (dataframe, params_used).
    """
    rng = np.random.default_rng(seed)
    if params is None:
        params = _sample_params(rng, condition)

    n = EXPECTED_LEN
    t = np.arange(n, dtype=np.float64)        # seconds

    baseline_mask = t < BASELINE_END_S
    stress_mask = (t >= BASELINE_END_S) & (t < STRESS_END_S)
    cooldown_mask = t >= STRESS_END_S
    tau_cool = params.tau_s * (
        COOLDOWN_TAU_FACTOR_BLOCKED if condition == "blocked"
        else COOLDOWN_TAU_FACTOR_NORMAL
    )

    # Stress trajectory
    t_rel_stress = np.where(stress_mask, t - BASELINE_END_S, 0.0)
    T_stress = params.T_amb_C + (params.T_peak_C - params.T_amb_C) * (
        1.0 - np.exp(-t_rel_stress / params.tau_s)
    )
    # Temperature at the moment cooldown begins (= end of stress phase)
    T_at_cooldown = params.T_amb_C + (params.T_peak_C - params.T_amb_C) * (
        1.0 - np.exp(-(STRESS_END_S - BASELINE_END_S) / params.tau_s)
    )
    t_rel_cool = np.where(cooldown_mask, t - STRESS_END_S, 0.0)
    T_cool = params.T_amb_C + (T_at_cooldown - params.T_amb_C) * np.exp(-t_rel_cool / tau_cool)

    T_max = np.where(
        baseline_mask, params.T_amb_C,
        np.where(stress_mask, T_stress, T_cool),
    )
    # Add noise
    T_max = T_max + _pink_noise(n, params.pink_noise_amp_C, rng) + rng.normal(0, params.noise_sigma_C, n)

    # T_min curve: roughly flat at the floor, follows T_max slowly during stress
    # (the laptop *base* warms up too, but more slowly and less)
    T_min_target = params.T_min_floor_C + np.where(
        stress_mask, 0.25 * (T_max - params.T_amb_C), 0.0  # 25 % of T_max rise
    )
    T_min = T_min_target + _pink_noise(n, params.pink_noise_amp_C * 0.4, rng) + rng.normal(0, params.noise_sigma_C * 0.5, n)

    # Round to one decimal place — TopInfrared export precision
    T_max = np.round(T_max, 1)
    T_min = np.round(T_min, 1)

    # Build DataFrame
    if start_time is None:
        start_time = datetime(2026, 6, 1, 14, 0, 0)
    timestamps = [start_time + timedelta(seconds=int(s)) for s in t]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "t_s": t,
        "T_max": T_max,
        "T_min": T_min,
        "T_range": T_max - T_min,
        "condition": condition,
        "session_id": session_id,
        "laptop_id": "toshiba_a01",
    })

    return df, params


def generate_dataset(
    n_normal: int = 10,
    n_blocked: int = 10,
    seed: int = DEFAULT_SEED,
    save_dir: Optional[Path] = None,
) -> tuple[list[pd.DataFrame], list[ThermalParams]]:
    """Generate a full synthetic dataset.

    If save_dir is given, also write CSV files using the project session naming
    convention.
    """
    rng = np.random.default_rng(seed)
    sessions = []
    params_log = []

    plan: list[tuple[str, int]] = []
    plan += [("normal", i) for i in range(n_normal)]
    plan += [("blocked", i) for i in range(n_blocked)]

    for sess_idx, (cond, k) in enumerate(plan, start=1):
        sid = f"S{sess_idx:03d}_{cond}_s"
        session_seed = int(rng.integers(0, 2**31 - 1))
        start = datetime(2026, 6, 1, 9, 0, 0) + timedelta(hours=sess_idx * 1.5)
        df, params = generate_session(sid, cond, session_seed, start_time=start)
        sessions.append(df)
        params_log.append(params)

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            date = start.strftime("%Y-%m-%d")
            fname = f"S{sess_idx:03d}_{date}_toshiba_{cond}_s.csv"
            df.to_csv(save_dir / fname, index=False)
    if save_dir is not None:
        log.info("Wrote %d synthetic sessions to %s (%d normal, %d blocked)",
                 len(sessions), save_dir, n_normal, n_blocked)

    return sessions, params_log


def save_params_summary(params_log: list[ThermalParams], path: Path) -> None:
    """Write a CSV summary of the parameters used per synthetic session."""
    pd.DataFrame([asdict(p) for p in params_log]).to_csv(path, index=False)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate synthetic Plane-export CSVs")
    p.add_argument("--n-normal", type=int, default=10)
    p.add_argument("--n-blocked", type=int, default=10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=Path("data/csv/synthetic"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sessions, params_log = generate_dataset(
        n_normal=args.n_normal,
        n_blocked=args.n_blocked,
        seed=args.seed,
        save_dir=args.out,
    )
    save_params_summary(params_log, args.out / "_params_used.csv")
    log.info("done.")
