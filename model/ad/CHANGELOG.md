# `model/ad/` — Changelog

A log of every change made to the anomaly-detection sub-package, with enough information to **retract** any change if it turns out to hurt rather than help. Format: most recent first.

---

## 2026-06-03 — Deterministic VAE scoring, thermal-rule baseline, and robust delay thresholding

### What changed
1. **LSTM-VAE scoring is deterministic during evaluation.** Training still uses latent sampling, but `score_frames()` and training reconstruction-error calibration decode from the posterior mean `mu` instead of sampling a random latent vector.
2. **LSTM-VAE session score changed from mean frame score to p95 frame score.** This prevents long baseline/cooldown periods from diluting persistent anomaly evidence during the stress phase.
3. **Added `PeakTmaxRuleDetector`.** This simple physical baseline scores a session by its maximum observed `T_max` and scores frames by instantaneous `T_max`.
4. **Delay thresholding now uses frame-level thresholds.** Detectors expose `frame_threshold_` calibrated from training-normal frame scores. The threshold is the median of per-session p99 frame scores, making it robust to one abnormal normal-labelled session.
5. **Added `tests/test_ad_pipeline.py`.** The tests cover deterministic VAE scoring, metric helpers, delay variants, and the peak-temperature rule.

### Why
The previous LSTM-VAE evaluation sampled from the latent distribution during scoring, so repeated scoring of the same session could change. That made the headline result harder to defend. The old session score also averaged all frame scores, which diluted anomaly evidence. Finally, the report needed an explicit "could a simple temperature threshold solve this?" baseline.

### Result of temporary real-data rerun

These results were not automatically copied to `outputs/ad`:

| Detector | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| PeakTmaxRule | 0.844 | 0.710 | 0.800 | 0.857 | 0.750 |
| Isolation Forest | 0.865 | 0.732 | 0.875 | 0.875 | 0.875 |
| One-Class SVM | 0.854 | 0.777 | 0.824 | 0.778 | 0.875 |
| LSTM-VAE p95 | 0.875 | 0.737 | 0.875 | 0.875 | 0.875 |

### How to retract
- Revert deterministic scoring by changing `_Net.forward(..., sample=False)` calls in `LSTMVAE.fit()` and `LSTMVAE.score_frames()` back to sampled calls. This is not recommended.
- Revert p95 session scoring by changing `LSTMVAE.score_session()` from `np.percentile(scores, 95)` back to `np.mean(scores)`.
- Remove `PeakTMaxRuleDetector` and its factory aliases from `models.py`, then remove `peak` from the default detector list in `run.py`.
- Revert robust frame thresholding by changing `_normal_frame_percentile_threshold()` back to a pooled percentile over all training-normal frame scores.

---

## 2026-06-XX — Real-data loader (.xlsx + variable timing + cooldown)

### What changed
The team's 20 recorded sessions arrived as **.xlsx** files (TCView Plane exports, columns "time / Lowest Temperature / Highest Temperature" with values like "23.4°C"), and the recording protocol shifted slightly without warning: each session now includes a 5-minute cooldown phase, and the actual phase boundaries vary by ±30 s per session because operators couldn't hit the stopwatch exactly.

Four coordinated changes to handle this:

1. **`run.load_sessions_from_dir` accepts both .csv and .xlsx**. For .xlsx it delegates to the existing `model/excel_loader.py` which already parses the "°C" suffix and the datetime column cleanly.
2. **`run._parse_filename_metadata` extracts ambient temperature** from filenames matching `..._<NN>C_...` and attaches it as a session column. The team adopted Jim's earlier suggestion to bake ambient into the filename, so this is now meaningful metadata.
3. **`run._detect_stress_onset_s` auto-detects the per-session stress onset** from the T_max curve: the first sample where T_max exceeds (baseline mean + 3σ) or (baseline mean + 1.5 °C), whichever is larger. Falls back to the nominal 300 s if the curve never visibly rises. The detected onset is attached as a `stress_onset_s` column.
4. **`run._normalize_length` truncates/pads every session to EXPECTED_LEN** (1500 samples = 25 min @ 1 Hz, matching the new 5+15+5 protocol). Warns if a session is more than 60 s off the target. This is required because the LSTM-VAE stacks sessions with `np.stack`, which needs uniform length.
5. **`evaluate.loso_evaluate` now uses per-session `stress_onset_s`** as the detection-delay anchor when present, falling back to BASELINE_END_S otherwise. Each fold's three delay metrics are now computed against the *actual* baseline-end of that session.
6. **`__init__.py` constants updated**: `SESSION_DURATION_S = 25*60`, `STRESS_END_S = 20*60`, `EXPECTED_LEN = 1500`. Added `SESSION_LEN_TOLERANCE_S = 60`.
7. **`synthetic.py` updated** to generate three-phase curves: baseline → stress → cooldown. Cooldown τ is 1.0× the heating τ for normal sessions, 1.8× for blocked (heat trapped, slower to dissipate). New session length is 1500 samples by default.

### Why
The previous loader only accepted .csv with clean numeric columns. The 1200-sample assumption baked into the LSTM-VAE would have rejected the new 25-min recordings. Auto-detecting per-session stress onset removes the ±30 s timing slop from the detection-delay metric — without this, sessions where the operator started the stressor 20 s late would report 20 s of artificial delay across the board.

### Files modified
- `model/ad/__init__.py` — constants
- `model/ad/run.py` — loader rewritten with the four helpers above
- `model/ad/evaluate.py` — per-session onset anchor in the LOSO delay calculation
- `model/ad/synthetic.py` — three-phase curve, blocked-vs-normal cooldown τ

### What stayed the same
- Filename pattern (`S*_*toshiba*.{csv,xlsx}`) — both extensions accepted.
- CSV path still works unchanged for synthetic data and any future clean exports.
- All three detection-delay variants (raw / after-transition / sustained) still computed — only the anchor moved.
- `BASELINE_END_S` constant still exists as the nominal fallback.

### How to retract
Each numbered change above is independent:

- **(1) Drop xlsx support**: remove the `.xlsx` branch in `_load_one_file`; revert the glob in `load_sessions_from_dir` to `.csv` only.
- **(2) Drop ambient parsing**: delete `_FILENAME_AMBIENT_RE`, the `_parse_filename_metadata` function, and the call site that attaches `ambient_C`.
- **(3) Drop onset auto-detection**: remove `_detect_stress_onset_s` and the `auto_detect_onset` parameter from `load_sessions_from_dir`. Sessions will no longer carry `stress_onset_s`; the evaluate fallback path will use BASELINE_END_S.
- **(4) Drop length normalisation**: remove `_normalize_length` and its call. The LSTM-VAE will then fail on variable-length sessions; switch to per-session training or pad inside the model instead.
- **(5) Drop per-session anchor in evaluate**: remove the `held_idx_to_session` and `anchor = float(...)` block; the three delay-variant calls will revert to defaulting to BASELINE_END_S.
- **(6) Revert constants** in `__init__.py` to the 20-min spec if the team drops cooldown.
- **(7) Revert synthetic** by deleting the cooldown block in `generate_session` and the `COOLDOWN_TAU_FACTOR_*` constants.

### Defaults (tunable)
- `baseline_window_s = 240` in `_detect_stress_onset_s` — how much of the early data to use as "baseline reference."
- `rise_sigma = 3.0`, `min_rise_C = 1.5` — detection threshold for stress onset.
- `target_len = EXPECTED_LEN = 1500` — uniform-length target.
- `SESSION_LEN_TOLERANCE_S = 60` — warn if a session is more than 60 s off target.

### Open question for the team
The auto-detect-onset logic assumes a clear baseline-to-stress transition. If a future session has the laptop already warm at t=0 (no proper cool-between), the baseline mean will be elevated and the detected onset will jump back to the nominal 300 s with a warning logged. Check `outputs/ad/<detector>/results.json[folds][i].stress_onset_s` after the first real run — if any are ≪ 240 s or ≫ 360 s, those sessions had timing issues at recording.

---

## 2026-05-31 — Detection-delay variants added

### What changed
Two new metric variants were added alongside the existing `detection_delay_s`:

1. **`detection_delay_after_transition_s`** — ignores the first 200 s after stress onset, on the grounds that the LSTM-VAE's baseline-phase noise can spuriously cross the Youden threshold by chance and report an artificial "delay of 0 s" that does not reflect real fault detection.
2. **`detection_delay_sustained_s`** — requires the per-frame score to remain above threshold for **30 consecutive frames (30 s at 1 Hz)** before counting the crossing as a detection.

Both are reported per fold in `results.json` and aggregated (mean/median/min/max/n_detected) under `delay.after_transition` and `delay.sustained` in the JSON summary. They also appear as extra columns in `outputs/ad/comparison.csv`.

### Why
The per-frame plot from the 2026-05-31 synthetic run showed that some blocked sessions had baseline-phase noise sitting above the Youden threshold (because the threshold was optimised for session-level classification and baseline noise has variance). Those sessions reported `delay = 0 s`, which inflated the "early detection" claim spuriously. The two variants give a noise-robust alternative; reporting all three together lets the reader judge robustness.

### Files modified
- `model/ad/evaluate.py` — added `detection_delay_after_transition_s()` and `detection_delay_sustained_s()`; extended `FoldResult` with two new fields; extended `loso_evaluate()` delay-summary block; extended `save_results()` per-fold serialisation.
- `model/ad/run.py` — extended the comparison-row dict with the new variants.

### What stayed the same (preserved for backward compatibility)
- The raw `detection_delay_s` field still exists in `FoldResult` and `results.json["folds"][i]`.
- The legacy aggregate keys `n_blocked_detected`, `n_blocked_total`, `delay_mean_s`, `delay_median_s`, `delay_min_s`, `delay_max_s` still exist in `results.json["delay"]` (computed from the raw variant). Anything reading those will continue to work.
- The legacy CSV columns `delay_mean_s`, `delay_median_s`, `n_blocked_detected`, `n_blocked_total` still exist in `comparison.csv`.

### Defaults
- `DEFAULT_TRANSITION_DURATION_S = 200.0` — calibrated against the synthetic LSTM-VAE per-frame plot's dip-and-recovery interval. Adjust upward if the real-data transition lasts longer.
- `DEFAULT_SUSTAINED_FRAMES = 30` — 30 s at 1 Hz. Increase to make the detector more conservative.

Both defaults live as module-level constants in `evaluate.py` so a single edit changes the behaviour for all downstream calls.

### How to retract
If either variant turns out to confuse the report more than it helps, retract it by:

1. **Remove the field from `FoldResult`** (`model/ad/evaluate.py`):
   - Delete the line `detection_delay_after_transition_s: float` (or `..._sustained_s: float`).
2. **Remove its population in `loso_evaluate`**:
   - Delete the matching `fr.detection_delay_..._s = detection_delay_..._s(...)` line.
   - Delete its entry from the `delay_summary` dict (the `"after_transition"` or `"sustained"` key).
3. **Remove its serialisation in `save_results`**:
   - Delete the `"detection_delay_..._s": fr.detection_delay_..._s` line in the `folds` list.
4. **Remove its column in `run.py`**:
   - Delete the four `delay_..._mean_s`, `delay_..._median_s`, `n_detected_...` lines.

The raw `detection_delay_s` metric is independent of both variants and will keep working in isolation.

To remove **both** variants completely, repeat each step above for each of them. Total surface area is ~20 lines of code across two files.

### Open question for the team
The "after-transition" threshold of 200 s is calibrated for the synthetic data. On real recordings the transition may be longer (slower thermal response on a real laptop than in our exponential model) or shorter (steeper ramp under mprep.info load). Suggest revisiting this constant after the first 5 real sessions are processed.
