"""Evaluation harness for unsupervised anomaly detection.

Implements:
  * leave-one-session-out cross-validation
  * ROC-AUC, PR-AUC, precision/recall/F1 at Youden's J threshold
  * detection-delay variants based on per-frame anomaly scores
  * JSON, CSV, and plot outputs
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from . import BASELINE_END_S
from .models import Detector

log = logging.getLogger("ad.evaluate")


def compute_roc_pr(scores: NDArray, labels: NDArray) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC, and Youden's J threshold + precision/recall/F1.

    `scores` higher = more anomalous; `labels` 1 = blocked / anomalous, 0 = normal.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

    if len(np.unique(labels)) < 2:
        log.warning("Only one class present; AUC undefined.")
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "threshold": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}

    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))

    fpr, tpr, thr = roc_curve(labels, scores)
    j = tpr - fpr
    best_j_idx = int(np.argmax(j))
    threshold = float(thr[best_j_idx])
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def detection_delay_s(per_frame_scores: NDArray, threshold: float,
                      stress_onset_s: float = BASELINE_END_S) -> float:
    """Seconds between stress onset and the first frame score >= threshold.

    The "raw" detection delay. Returns -1 if the score never crosses the
    threshold during the session. Frame index is taken to be the
    second-from-session-start, matching 1 Hz sampling.

    Known weakness: if the baseline-phase noise already happens to be above
    threshold, the very first stress frame may score above threshold by
    coincidence and report a delay of 0 s that does not reflect actual fault
    detection. Use `detection_delay_after_transition_s` or
    `detection_delay_sustained_s` for noise-robust alternatives.
    """
    crossings = np.where(per_frame_scores >= threshold)[0]
    crossings = crossings[crossings >= stress_onset_s]
    if crossings.size == 0:
        return -1.0
    return float(crossings[0] - stress_onset_s)


# Ignore early transition crossings when reporting the conservative delay.
DEFAULT_TRANSITION_DURATION_S = 200.0

# At 1 Hz, this requires 30 consecutive seconds above threshold.
DEFAULT_SUSTAINED_FRAMES = 30


def detection_delay_after_transition_s(
    per_frame_scores: NDArray,
    threshold: float,
    stress_onset_s: float = BASELINE_END_S,
    transition_duration_s: float = DEFAULT_TRANSITION_DURATION_S,
) -> float:
    """Detection delay, ignoring the first `transition_duration_s` seconds
    after stress onset. Returns -1 if the score never crosses the threshold
    after the transition window.

    This avoids counting early baseline/transition noise as a real alarm.
    """
    start = stress_onset_s + transition_duration_s
    crossings = np.where(per_frame_scores >= threshold)[0]
    crossings = crossings[crossings >= start]
    if crossings.size == 0:
        return -1.0
    return float(crossings[0] - stress_onset_s)


def detection_delay_sustained_s(
    per_frame_scores: NDArray,
    threshold: float,
    stress_onset_s: float = BASELINE_END_S,
    sustained_frames: int = DEFAULT_SUSTAINED_FRAMES,
) -> float:
    """Detection delay defined as the first index after stress_onset where
    the score has been continuously above `threshold` for `sustained_frames`
    consecutive samples. Returns -1 if no such window exists.

    Single-frame crossings can be noise, so this variant requires persistence.
    """
    above = per_frame_scores >= threshold
    n = len(above)
    if n < sustained_frames:
        return -1.0
    # Rolling sum via cumulative sum: count crossings in each window.
    csum = np.concatenate(([0], np.cumsum(above.astype(np.int32))))
    win_counts = csum[sustained_frames:] - csum[:-sustained_frames]   # length n-sustained_frames+1
    full_windows = np.where(win_counts == sustained_frames)[0]
    if full_windows.size == 0:
        return -1.0
    valid = full_windows[full_windows >= stress_onset_s]
    if valid.size == 0:
        return -1.0
    return float(valid[0] - stress_onset_s)


@dataclass
class FoldResult:
    fold_idx: int
    held_out_session_id: str
    held_out_condition: str          # "normal" or "blocked"
    session_score: float             # detector's session-level anomaly score
    detection_delay_s: float         # raw: first crossing after stress onset
    detection_delay_after_transition_s: float   # ignore noisy first 200 s
    detection_delay_sustained_s: float          # require N consec frames above
    frame_threshold: float        # per-fold frame threshold used for delay
    frame_scores: NDArray            # full per-frame score series


def loso_evaluate(
    detector_factory: Callable[[], Detector],
    normal_sessions: list[pd.DataFrame],
    blocked_sessions: list[pd.DataFrame],
    threshold_strategy: str = "loso_youden",
) -> dict:
    """Run leave-one-session-out CV across normal and blocked sessions.

    For each fold:
      1. Hold out one session (could be normal or blocked).
      2. Retrain the detector on the remaining normal sessions only.
      3. Compute session-level + per-frame anomaly scores on the held-out one.
    After all folds:
      4. Aggregate session scores across folds → ROC-AUC / PR-AUC / F1.
      5. Pick the session threshold (Youden's J on aggregated session scores).
      6. Compute detection delay for each blocked fold using that fold's
         training-normal frame threshold when the detector provides one.
    """
    all_sessions = [(s, 0, "normal") for s in normal_sessions] + \
                   [(s, 1, "blocked") for s in blocked_sessions]
    n = len(all_sessions)
    if n == 0:
        raise ValueError("No sessions provided.")
    log.info("LOSO: %d sessions total (%d normal, %d blocked)",
             n, len(normal_sessions), len(blocked_sessions))

    fold_results: list[FoldResult] = []
    aggregated_scores: list[float] = []
    aggregated_labels: list[int] = []

    for i, (held, label, cond) in enumerate(all_sessions):
        train_pool = [s for j, (s, lbl, _) in enumerate(all_sessions) if j != i and lbl == 0]
        if not train_pool:
            log.warning("Fold %d: no normal sessions left to train on, skipping", i)
            continue

        det = detector_factory()
        det.fit(train_pool)
        sess_score = det.score_session(held)
        frame_scores = det.score_frames(held)
        frame_threshold = float(getattr(det, "frame_threshold_", np.nan))

        fold_results.append(FoldResult(
            fold_idx=i,
            held_out_session_id=str(held.get("session_id", pd.Series([f"S{i:03d}"])).iloc[0]),
            held_out_condition=cond,
            session_score=sess_score,
            detection_delay_s=-1.0,                      # filled below
            detection_delay_after_transition_s=-1.0,     # filled below
            detection_delay_sustained_s=-1.0,            # filled below
            frame_threshold=frame_threshold,
            frame_scores=frame_scores,
        ))
        aggregated_scores.append(sess_score)
        aggregated_labels.append(label)
        log.info("  fold %2d/%d  %s  score=%.4f", i + 1, n, cond, sess_score)

    scores = np.array(aggregated_scores)
    labels = np.array(aggregated_labels)
    metrics = compute_roc_pr(scores, labels)
    log.info("Aggregate: ROC-AUC=%.3f  PR-AUC=%.3f  F1=%.3f (thr=%.4f)",
             metrics["roc_auc"], metrics["pr_auc"], metrics["f1"], metrics["threshold"])

    # Delay uses each fold's training-normal frame threshold where available,
    # because frame scores and session scores can have different scales.
    session_threshold = metrics["threshold"]
    held_idx_to_session = {idx: sess for idx, (sess, _, _) in enumerate(all_sessions)}
    for fr in fold_results:
        if fr.held_out_condition != "blocked":
            continue
        sess = held_idx_to_session.get(fr.fold_idx)
        anchor = BASELINE_END_S
        if sess is not None and "stress_onset_s" in sess.columns:
            anchor = float(sess["stress_onset_s"].iloc[0])
        delay_threshold = fr.frame_threshold if np.isfinite(fr.frame_threshold) else session_threshold
        fr.detection_delay_s = detection_delay_s(fr.frame_scores, delay_threshold, stress_onset_s=anchor)
        fr.detection_delay_after_transition_s = detection_delay_after_transition_s(
            fr.frame_scores, delay_threshold, stress_onset_s=anchor
        )
        fr.detection_delay_sustained_s = detection_delay_sustained_s(
            fr.frame_scores, delay_threshold, stress_onset_s=anchor
        )

    def _summarize(attr: str) -> dict:
        d = [getattr(fr, attr) for fr in fold_results
             if fr.held_out_condition == "blocked" and getattr(fr, attr) >= 0]
        n_block_total = sum(1 for fr in fold_results if fr.held_out_condition == "blocked")
        return {
            "n_detected": len(d),
            "n_total": n_block_total,
            "mean_s":   float(np.mean(d)) if d else -1.0,
            "median_s": float(np.median(d)) if d else -1.0,
            "min_s":    float(np.min(d))    if d else -1.0,
            "max_s":    float(np.max(d))    if d else -1.0,
        }

    raw = _summarize("detection_delay_s")
    delay_summary = {
        "n_blocked_detected": raw["n_detected"],
        "n_blocked_total":    raw["n_total"],
        "delay_mean_s":       raw["mean_s"],
        "delay_median_s":     raw["median_s"],
        "delay_min_s":        raw["min_s"],
        "delay_max_s":        raw["max_s"],
        "raw":              raw,
        "after_transition": _summarize("detection_delay_after_transition_s"),
        "sustained":        _summarize("detection_delay_sustained_s"),
    }

    return {
        "detector": detector_factory().name,
        "n_sessions": n,
        "n_normal": len(normal_sessions),
        "n_blocked": len(blocked_sessions),
        "metrics": metrics,
        "delay": delay_summary,
        "fold_results": fold_results,
        "scores": scores,
        "labels": labels,
    }


def save_results(results: dict, output_dir: Path) -> None:
    """Persist results to disk: JSON summary + per-fold scores CSV + plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "detector": results["detector"],
        "n_sessions": results["n_sessions"],
        "n_normal": results["n_normal"],
        "n_blocked": results["n_blocked"],
        "metrics": results["metrics"],
        "delay": results["delay"],
        "folds": [
            {
                "fold_idx": fr.fold_idx,
                "session_id": fr.held_out_session_id,
                "condition": fr.held_out_condition,
                "session_score": fr.session_score,
                "detection_delay_s": fr.detection_delay_s,
                "detection_delay_after_transition_s": fr.detection_delay_after_transition_s,
                "detection_delay_sustained_s": fr.detection_delay_sustained_s,
                "frame_threshold": fr.frame_threshold,
            }
            for fr in results["fold_results"]
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote results.json")

    rows = []
    for fr in results["fold_results"]:
        for t_idx, sc in enumerate(fr.frame_scores):
            rows.append({
                "fold": fr.fold_idx,
                "session_id": fr.held_out_session_id,
                "condition": fr.held_out_condition,
                "t_s": t_idx,
                "score": float(sc),
            })
    pd.DataFrame(rows).to_csv(output_dir / "frame_scores.csv", index=False)
    log.info("Wrote frame_scores.csv (%d rows)", len(rows))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plot_roc(results, output_dir)
        _plot_score_histograms(results, output_dir)
        _plot_frame_scores(results, output_dir)
        log.info("Plots saved to %s", output_dir)
    except ImportError:
        log.warning("matplotlib not available; skipping plots")


def _plot_roc(results: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(results["labels"], results["scores"])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"{results['detector']} (AUC={results['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], lw=1, ls="--", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("LOSO ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc.png", dpi=120)
    plt.close()


def _plot_score_histograms(results: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    normal_scores = [fr.session_score for fr in results["fold_results"] if fr.held_out_condition == "normal"]
    blocked_scores = [fr.session_score for fr in results["fold_results"] if fr.held_out_condition == "blocked"]
    plt.figure(figsize=(6, 4))
    bins = 15
    plt.hist(normal_scores, bins=bins, alpha=0.6, label="normal")
    plt.hist(blocked_scores, bins=bins, alpha=0.6, label="blocked")
    plt.axvline(results["metrics"]["threshold"], color="black", ls="--", label="Youden J")
    plt.xlabel("Session anomaly score"); plt.ylabel("Count")
    plt.title(f"{results['detector']} — session-score distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_hist.png", dpi=120)
    plt.close()


def _plot_frame_scores(results: dict, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    for fr in results["fold_results"]:
        color = "tab:red" if fr.held_out_condition == "blocked" else "tab:blue"
        plt.plot(fr.frame_scores, color=color, alpha=0.4, lw=1)
    plt.axvline(BASELINE_END_S, color="black", ls=":", lw=1, label="stress onset")
    frame_thresholds = [
        fr.frame_threshold for fr in results["fold_results"]
        if np.isfinite(fr.frame_threshold)
    ]
    if frame_thresholds:
        plt.axhline(float(np.median(frame_thresholds)), color="black", ls="--", lw=1,
                    label="median frame threshold")
    else:
        plt.axhline(results["metrics"]["threshold"], color="black", ls="--", lw=1,
                    label="session threshold")
    plt.xlabel("t (s from session start)"); plt.ylabel("Per-frame anomaly score")
    plt.title(f"{results['detector']} — per-frame scores (red=blocked, blue=normal)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_scores.png", dpi=120)
    plt.close()
