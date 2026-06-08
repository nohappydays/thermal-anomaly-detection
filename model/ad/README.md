# Unsupervised Anomaly Detection (`model/ad/`)

This sub-package implements the final CSV/XLSX anomaly-detection path: detect blocked-vent thermal anomalies from TopInfrared **Plane exports** using detectors trained on normal sessions only.

## Layout

| File | Purpose |
|------|---------|
| `synthetic.py`  | Generate synthetic Plane sessions for smoke tests |
| `features.py`   | Scalar / windowed / time-series feature engineering |
| `models.py`     | Detector implementations (Peak-Tmax rule, IF, OC-SVM, LSTM-VAE) |
| `evaluate.py`   | Leave-one-session-out CV + metrics + plots |
| `run.py`        | CLI orchestration |
| `__init__.py`   | Protocol constants (`SESSION_DURATION_S`, `BASELINE_END_S`, `EXPECTED_LEN`) |

## Detectors

All detectors implement the same interface:

```python
det = make_detector("peak" | "if" | "ocsvm" | "vae")
det.fit(normal_sessions)          # list of DataFrames
session_score = det.score_session(test_session)
per_frame   = det.score_frames(test_session)
```

| Detector | When to prefer | Notes |
|----------|----------------|-------|
| `PeakTmaxRule` | Physical sanity baseline | Scores each session by peak `T_max`; answers whether a simple thermal threshold is enough |
| `IsolationForest` | Tiny dataset, fast baseline | scikit-learn, no GPU, ~1 s to fit + score |
| `OneClassSVM` | Same | RBF kernel, slightly stronger baseline |
| `LSTMVAE` | Per-frame reconstruction scoring | PyTorch; trained from scratch each fold; session score is p95 of frame scores |

Higher score = more anomalous, by convention. The LSTM-VAE z-scores reconstruction error against the training-normal reconstruction-error distribution. Classical baselines and the peak-temperature rule keep their own natural score scales; cross-detector comparison should use ROC-AUC/F1 rather than raw score magnitudes.

## Metrics

Every run reports:

- **ROC-AUC** — the headline metric (higher = better)
- **PR-AUC** — secondary, useful if class imbalance shifts later
- **Threshold (Youden's J)** — best balance of TPR and FPR on the LOSO scores
- **Precision / Recall / F1** at that threshold
- **Detection delay** — seconds between stress onset and the first frame whose score crosses a frame-level threshold calibrated from training-normal frame scores (reported as raw, after-transition, and sustained variants)

A `comparison.csv` is written for cross-detector comparison.

## Quick start

### Dry-run on synthetic data
```bash
python -m model.ad.run --synthetic --n-normal 10 --n-blocked 10 --detector peak if ocsvm
```

Outputs land in `outputs/ad/<detector>/`:
- `results.json` — summary metrics + per-fold scores
- `frame_scores.csv` — long-format per-frame scores
- `roc.png` — ROC curve
- `score_hist.png` — session-score histograms by class
- `frame_scores.png` — per-frame score traces overlaid (red = blocked, blue = normal)

### Real CSV/XLSX data

Place files in `data/csv/` using the session naming convention:
```
S<NNN>_<YYYY-MM-DD>_toshiba_<normal|blocked|faulty>_<ambient>C_a.<csv|xlsx>
```

The CSVs must have at minimum these columns:
- `t_s` — seconds from session start (float)
- `T_max` — peak Plane temperature, °C (float)
- `T_min` — floor Plane temperature, °C (float; optional but recommended)
- `condition` — `normal` or `blocked` (string)

If `condition` is missing, the loader infers it from the filename. If `t_s` is missing, it is derived from a `timestamp` column. The current real-data run normalises sessions to 1500 samples, matching the nominal 5 min baseline + 15 min stress + 5 min cooldown protocol.

Then run:
```bash
python -m model.ad.run --data data/csv --detector peak if ocsvm vae --epochs 100
```

## Reproducibility

- All RNGs (numpy + torch + sklearn) are seeded from `DEFAULT_SEED = 20260531` unless overridden via `--seed`.
- LSTM-VAE evaluation is deterministic: training uses latent sampling, while scoring uses the posterior mean.
- Results should be reproducible for the same seed and data on the same library/hardware stack.
- `comparison.csv` makes the cross-detector comparison auditable at a glance.

## Synthetic-data caveats

`synthetic.py` produces deliberately separable sessions: the blocked-vent parameters shift `T_peak`, `τ`, and the chassis floor by amounts in the realistic range. That validates the pipeline on known structure; it does not predict real-data performance.

## Design notes for the report

The architecture choice of LSTM-VAE follows Xu and Zhang (2025) for steam-turbine AD and Han et al. (2021) for maritime-component fault detection — the two closest analogues in the lit review. The training-on-normal-only setup, combined with leave-one-session-out cross-validation, is the standard one-class anomaly-detection protocol (Jakubowski et al., 2021).

The per-frame scoring path is what enables the `detection_delay_s` metric: the LSTM-VAE produces one reconstruction-error score per second, and the Peak-Tmax rule naturally produces one temperature score per second. IF and OC-SVM are session-level models; their delay values use a sliding T_max-vs-baseline proxy, so those delays should be treated as fallback diagnostics rather than native IF/OC-SVM alarm times.

## Limitations (worth flagging in the report)

- **Catastrophic fault**: both vents fully blocked produces a peak-temperature shift that is plausibly detectable by a simple T_max threshold alone. The unsupervised AD model's marginal value is therefore best demonstrated through detection delay, not session-level classification.
- **Single laptop**: results are bounded to the Toshiba; cross-device generalization claims are not supported by this dataset.
- **Ambient and operating-state variability**: ambient is encoded in the filenames, but it was not actively controlled. Warm starts, fan state, and background load remain major confounders, as shown by the S001 normal-label thermal outlier.
