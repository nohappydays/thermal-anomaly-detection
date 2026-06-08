"""CLI entry point for CSV/XLSX anomaly-detection experiments.

Usage:
    python -m model.ad.run --synthetic
    python -m model.ad.run --data data/csv
    python -m model.ad.run --data data/csv --detector vae --epochs 200
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow `python -m model.ad.run` from the project root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from model.ad import (
    DEFAULT_SEED, BASELINE_END_S, EXPECTED_LEN, SESSION_LEN_TOLERANCE_S,
)
from model.ad.evaluate import loso_evaluate, save_results
from model.ad.models import make_detector
from model.ad.synthetic import generate_dataset

log = logging.getLogger("ad.run")


_FILENAME_AMBIENT_RE = re.compile(r"_(-?\d+)C_")                              # "..._22C_a.xlsx"
# Some recordings use "faulty"; the experiment code uses "blocked".
_FILENAME_CONDITION_RE = re.compile(r"_(normal|blocked|faulty)_")


def _parse_filename_metadata(stem: str) -> dict:
    """Extract condition and ambient_C from the session filename."""
    meta: dict = {}
    m_amb = _FILENAME_AMBIENT_RE.search(stem)
    if m_amb:
        meta["ambient_C"] = float(m_amb.group(1))
    m_cond = _FILENAME_CONDITION_RE.search(stem)
    if m_cond:
        raw = m_cond.group(1)
        meta["condition"] = "blocked" if raw == "faulty" else raw
    return meta


def _load_one_file(path: Path) -> pd.DataFrame:
    """Load one .csv or TopInfrared .xlsx session into the common schema."""
    meta = _parse_filename_metadata(path.stem)
    cond = meta.get("condition", "unknown")

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        from model.excel_loader import load_TopInfrared
        exp = load_TopInfrared(path, condition=cond, laptop_id="toshiba_a01",
                          session_id=path.stem)
        df = exp.df.copy()
    elif suffix == ".csv":
        df = pd.read_csv(path)
        if "condition" not in df.columns:
            df["condition"] = cond
        if "session_id" not in df.columns:
            df["session_id"] = path.stem
        if "t_s" not in df.columns:
            if "timestamp" in df.columns:
                ts = pd.to_datetime(df["timestamp"])
                df["t_s"] = (ts - ts.iloc[0]).dt.total_seconds()
            else:
                df["t_s"] = pd.RangeIndex(len(df)).astype(float)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected .csv or .xlsx)")

    if "ambient_C" in meta:
        df["ambient_C"] = meta["ambient_C"]

    return df


def _detect_stress_onset_s(df: pd.DataFrame,
                           baseline_window_s: float = 240.0,
                           rise_sigma: float = 3.0,
                           min_rise_C: float = 1.5) -> float:
    """Estimate stress onset from the first clear T_max rise above baseline."""
    t = df["t_s"].to_numpy(dtype=np.float64)
    tmax = df["T_max"].to_numpy(dtype=np.float64)
    if len(t) < 60:
        return float(BASELINE_END_S)
    early = t <= baseline_window_s
    if early.sum() < 30:
        return float(BASELINE_END_S)
    mu = float(np.nanmean(tmax[early]))
    sigma = float(np.nanstd(tmax[early])) + 1e-6
    threshold = max(mu + rise_sigma * sigma, mu + min_rise_C)
    above = np.where(tmax > threshold)[0]
    if above.size == 0:
        log.warning("Could not detect stress onset for session; using nominal %.0f s",
                    BASELINE_END_S)
        return float(BASELINE_END_S)
    return float(t[above[0]])


def _normalize_length(df: pd.DataFrame, target_len: int = EXPECTED_LEN) -> pd.DataFrame:
    """Return a fixed-length session for LSTM-VAE stacking."""
    n = len(df)
    if n == target_len:
        return df
    if abs(n - target_len) > SESSION_LEN_TOLERANCE_S:
        log.warning("Session %s length %d differs from target %d by >%d samples",
                    df.get("session_id", pd.Series(["?"])).iloc[0],
                    n, target_len, SESSION_LEN_TOLERANCE_S)
    if n > target_len:
        return df.iloc[:target_len].reset_index(drop=True)
    # Pad by repeating the last observation and extending the timestamp.
    last = df.iloc[[-1]]
    pad = pd.concat([last] * (target_len - n), ignore_index=True)
    pad["t_s"] = pad["t_s"] + np.arange(1, target_len - n + 1, dtype=float)
    return pd.concat([df.reset_index(drop=True), pad], ignore_index=True)


def load_sessions_from_dir(csv_dir: Path,
                           target_len: int = EXPECTED_LEN,
                           auto_detect_onset: bool = True) -> list[pd.DataFrame]:
    """Load all Toshiba session files from a directory."""
    paths = sorted(list(csv_dir.glob("S*_*toshiba*.csv")) +
                   list(csv_dir.glob("S*_*toshiba*.xlsx")))
    if not paths:
        log.warning("No sessions matching S*_*toshiba*.{csv,xlsx} in %s", csv_dir)
        return []

    sessions: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = _load_one_file(p)
        except Exception as e:
            log.exception("Failed to load %s: %s", p.name, e)
            continue

        if auto_detect_onset:
            onset = _detect_stress_onset_s(df)
            df["stress_onset_s"] = onset
            log.debug("%s: detected stress onset at %.0f s (nominal %d)",
                      p.name, onset, BASELINE_END_S)

        df = _normalize_length(df, target_len=target_len)
        sessions.append(df)

    if sessions:
        onsets = [s["stress_onset_s"].iloc[0] for s in sessions if "stress_onset_s" in s.columns]
        amb = [s["ambient_C"].iloc[0] for s in sessions if "ambient_C" in s.columns]
        log.info("Loaded %d sessions from %s (%s normal, %s blocked)",
                 len(sessions), csv_dir,
                 sum(1 for s in sessions if s["condition"].iloc[0] == "normal"),
                 sum(1 for s in sessions if s["condition"].iloc[0] == "blocked"))
        if onsets:
            log.info("  stress onset: mean=%.0f s, range=%.0f–%.0f s (nominal %d)",
                     float(np.mean(onsets)), float(min(onsets)), float(max(onsets)),
                     BASELINE_END_S)
        if amb:
            log.info("  ambient: mean=%.1f °C, range=%.1f–%.1f °C",
                     float(np.mean(amb)), float(min(amb)), float(max(amb)))
    return sessions


def split_by_condition(sessions: list[pd.DataFrame]) -> tuple[list, list]:
    normal = [s for s in sessions if s["condition"].iloc[0] == "normal"]
    blocked = [s for s in sessions if s["condition"].iloc[0] == "blocked"]
    return normal, blocked

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train + evaluate AD baselines on TopInfrared Plane CSV/XLSX exports")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true",
                     help="Generate synthetic dataset on the fly")
    src.add_argument("--data", type=Path,
                     help="Directory with real CSV/XLSX files (S*_*_toshiba_*.csv/.xlsx)")
    p.add_argument("--n-normal", type=int, default=10)
    p.add_argument("--n-blocked", type=int, default=10)
    p.add_argument("--detector", nargs="+", default=["peak", "if", "ocsvm", "vae"],
                   help="Detectors to run; default runs thermal rule + IF + OC-SVM + VAE")
    p.add_argument("--epochs", type=int, default=100, help="LSTM-VAE training epochs")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "ad")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.synthetic:
        log.info("Generating synthetic dataset: %d normal + %d blocked",
                 args.n_normal, args.n_blocked)
        sessions, _ = generate_dataset(args.n_normal, args.n_blocked, seed=args.seed)
    else:
        sessions = load_sessions_from_dir(args.data)
    if not sessions:
        log.error("No data to evaluate."); return 1

    normal, blocked = split_by_condition(sessions)
    log.info("Split: %d normal, %d blocked", len(normal), len(blocked))
    if not normal:
        log.error("No normal sessions found — cannot train an unsupervised AD model.")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    comparison_rows = []

    for det_kind in args.detector:
        log.info("Running detector: %s", det_kind)
        def factory(kind=det_kind):
            if kind in ("vae", "lstmvae", "lstm-vae"):
                return make_detector(kind, epochs=args.epochs)
            return make_detector(kind)
        try:
            results = loso_evaluate(factory, normal, blocked)
        except Exception as e:
            log.exception("Detector %s failed: %s", det_kind, e)
            continue
        out_dir = args.output / results["detector"]
        save_results(results, out_dir)
        comparison_rows.append({
            "detector": results["detector"],
            "n_normal": results["n_normal"],
            "n_blocked": results["n_blocked"],
            "roc_auc": results["metrics"]["roc_auc"],
            "pr_auc": results["metrics"]["pr_auc"],
            "f1": results["metrics"]["f1"],
            "precision": results["metrics"]["precision"],
            "recall": results["metrics"]["recall"],
            "threshold": results["metrics"]["threshold"],
            "delay_mean_s": results["delay"]["delay_mean_s"],
            "delay_median_s": results["delay"]["delay_median_s"],
            "n_blocked_detected": results["delay"]["n_blocked_detected"],
            "n_blocked_total": results["delay"]["n_blocked_total"],
            "delay_after_transition_mean_s":  results["delay"]["after_transition"]["mean_s"],
            "delay_after_transition_median_s":results["delay"]["after_transition"]["median_s"],
            "n_detected_after_transition":    results["delay"]["after_transition"]["n_detected"],
            "delay_sustained_mean_s":   results["delay"]["sustained"]["mean_s"],
            "delay_sustained_median_s": results["delay"]["sustained"]["median_s"],
            "n_detected_sustained":     results["delay"]["sustained"]["n_detected"],
        })

    if comparison_rows:
        cmp_path = args.output / "comparison.csv"
        pd.DataFrame(comparison_rows).to_csv(cmp_path, index=False)
        log.info("Wrote %s", cmp_path)
        print("\n=== DETECTOR COMPARISON ===")
        print(pd.DataFrame(comparison_rows).to_string(index=False))

    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
