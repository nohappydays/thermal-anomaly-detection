# Technical Model and Results Log

Purpose: keep a focused record of the code, metrics, modelling choices, and technical defence points for the CSV-only anomaly-detection part of the project. This is separate from `CLAUDE_CONTEXT_LOG.md`, which tracks the whole project.

Last updated: 2026-06-03 by Codex.

## Current Final-Report Scope

- Input data: TCView Plane CSV/XLSX exports at 1 Hz.
- Features used by the LSTM-VAE: `T_max`, `T_min`, and `T_range`.
- Dataset used for reported results: 20 sessions after quality filtering, consisting of 12 normal and 8 blocked/faulty sessions.
- Quarantined session: `data/csv/_QUARANTINE_S015_normal_24C.xlsx`.
- Known retained outlier: `S001_2026-06-01_toshiba_normal_22C_a`, labelled normal but thermally abnormal, with peak `T_max = 58.4 C`.
- Evaluation protocol: leave-one-session-out; models train only on normal sessions in each fold.

## Technical Audit, 2026-06-03

The first audit identified five improvements worth doing before freezing the report numbers:

1. Make LSTM-VAE scoring deterministic during evaluation.
   - Issue: the VAE decoder sampled from the latent distribution during `score_frames`, so the same trained model could give different scores for the same session.
   - Decision: keep stochastic latent sampling during training, but use the posterior mean `mu` during scoring and when computing training reconstruction-error statistics.

2. Add a simple physics baseline.
   - Reason: supervisors may ask whether a learned detector is necessary when the fault is thermal.
   - Audit result before implementation: simple threshold-style scores were already strong but below the reported LSTM-VAE result.
   - Manual audit on the 20 sessions:
     - `peak_Tmax`: ROC-AUC about 0.84, F1 about 0.80.
     - final-five-minute mean `Tmax`: ROC-AUC about 0.85, F1 about 0.82.
   - Decision: include a transparent thermal-rule baseline so the report can say the LSTM-VAE was compared not only against generic unsupervised baselines, but also against the obvious physical shortcut.

3. Clarify detection-delay thresholding.
   - Issue: the current delay variants apply the session-level Youden threshold to per-frame scores.
   - Decision direction: keep session-level metrics as primary; make delay reporting explicit and consider a normal-frame calibrated threshold for the LSTM-VAE.

4. Add a small test suite.
   - Reason: the pipeline has metric definitions that are easy to change accidentally.
   - Minimum tests needed: deterministic VAE scoring, delay metric behavior, filename/session loading logic for synthetic data, and thermal-rule detector scoring.

5. Keep a technical defence notebook.
   - This file should record the choices, trade-offs, and likely viva/presentation questions.

## Presentation Defence Points

- Why unsupervised? In industrial deployments, labelled fault corpora are often unavailable; training on normal sessions only better matches that constraint.
- Why compare to simple baselines? Because the fault is thermal and catastrophic; if a temperature threshold is enough, the model's added value must be shown honestly.
- Why is the S001 false positive not hidden? Removing it would improve the score but conceal a real protocol sensitivity. Keeping it makes the result more defensible.
- Why is delay reported carefully? A session-level detector and a frame-level alarm threshold are different objects. The report should not over-claim early detection.
- What is the main technical limitation? Small dataset, one laptop, binary fully-blocked fault, uncontrolled ambient temperature, and imperfect control of warm-start conditions.

## Open Follow-Ups

## Implementation Pass, 2026-06-03

Implemented changes:

- LSTM-VAE scoring is now deterministic during evaluation.
  - Training still samples from the latent distribution.
  - Scoring uses the posterior mean `mu`.
  - Training reconstruction-error statistics are also computed deterministically.

- Added `PeakTmaxRule`, a simple thermal-rule baseline.
  - Session score: maximum observed `T_max`.
  - Frame score: instantaneous `T_max`.
  - Purpose: answer whether a physical temperature threshold is already sufficient.

- Changed LSTM-VAE session aggregation.
  - Previous score: mean per-frame reconstruction score.
  - New score: 95th percentile per-frame reconstruction score.
  - Rationale: a session-level anomaly can be diluted by long normal-looking baseline/cooldown periods; p95 preserves persistent anomaly evidence without relying on one spike.

- Changed delay thresholding.
  - Session classification still uses Youden's J on held-out session scores.
  - Delay now uses a frame-level threshold calibrated from training-normal frame scores.
  - The frame threshold is the median of each training-normal session's 99th-percentile frame score, not a pooled percentile, so one S001-like outlier does not set the threshold for every fold.

- Added a focused test suite in `tests/test_ad_pipeline.py`.
  - Tests deterministic LSTM-VAE scoring, thermal-rule scoring, ROC/PR computation, and delay variants.
  - Verification command: `python3 -m unittest tests.test_ad_pipeline`.
  - Result on 2026-06-03: 4 tests passed.

## Temporary Rerun Results, 2026-06-03

The real XLSX sessions were converted to temporary normalized CSV files in `/private/tmp/ad_real_csv` so the system Python environment could run PyTorch/sklearn without needing `openpyxl`.

Temporary output directory: `/private/tmp/ad_real_results_v2`.

These results have **not** yet replaced `outputs/ad` or the report tables.

| Detector | ROC-AUC | PR-AUC | F1 | Precision | Recall | Delay mean/median | Delay note |
|---|---:|---:|---:|---:|---:|---:|---|
| PeakTmaxRule | 0.844 | 0.710 | 0.800 | 0.857 | 0.750 | 81/57 s raw; 209/200 s after-transition | 7/8 blocked sessions crossed frame threshold |
| Isolation Forest | 0.865 | 0.732 | 0.875 | 0.875 | 0.875 | 430/433 s | delay is based on its T_max proxy frame score |
| One-Class SVM | 0.854 | 0.777 | 0.824 | 0.778 | 0.875 | 430/433 s | delay is based on its T_max proxy frame score |
| LSTM-VAE p95 | 0.875 | 0.737 | 0.875 | 0.875 | 0.875 | 86/0 s raw; 257/200 s after-transition | 7/8 blocked sessions crossed frame threshold |

Interpretation:

- The old 0.92 ROC-AUC / 0.94 F1 LSTM-VAE result was not robust to deterministic VAE scoring.
- The improved deterministic LSTM-VAE remains competitive and has the best ROC-AUC among the current detectors, but only narrowly.
- Isolation Forest ties the LSTM-VAE on F1.
- One-Class SVM has the highest PR-AUC in this rerun.
- The PeakTmaxRule baseline is strong, confirming that the binary fully-blocked fault is partly solvable by direct temperature thresholding.
- The LSTM-VAE's real added value should be framed as trajectory-based anomaly scoring and engineering insight, not as a large session-classification improvement over baselines.

## Updated Defence Points

- We intentionally added a simple thermal threshold baseline because any thermal-fault detector must beat or at least contextualise the obvious physical shortcut.
- Deterministic VAE scoring reduced the headline metric, but it made the result reproducible and technically defensible.
- The VAE session score now uses p95 rather than mean because a whole-session mean can dilute a fault that appears after baseline and during stress.
- The dataset is small and internally confounded: S001 is a normal-labelled thermal outlier, while some blocked sessions are weak and overlap with warm normal sessions.
- The most honest result claim is: the LSTM-VAE is competitive with classical unsupervised baselines and slightly stronger than the simple peak-temperature rule on ROC-AUC, but the experiment mainly demonstrates the engineering process and limitations rather than a decisive model win.

## Report Update, 2026-06-03

Updated `docs/Report_Draft_v1.md` and `overleaf_ready/report.tex` to use the deterministic rerun results instead of the earlier stochastic LSTM-VAE numbers.

Main report changes:

- Replaced the old 0.92 ROC-AUC / 0.94 F1 LSTM-VAE headline with the deterministic p95 result: 0.875 ROC-AUC and 0.875 F1.
- Added the PeakTmaxRule baseline to the methods and results table.
- Reframed the conclusion from "LSTM-VAE clearly outperforms the baselines" to "LSTM-VAE has the best ROC-AUC by a narrow margin and ties Isolation Forest on F1."
- Updated the LSTM-VAE confusion matrix to 11 true normal, 1 false positive, 1 false negative, and 7 true blocked at threshold 0.420.
- Updated the detection-delay text to 257 s mean / 200 s median after-transition delay over 7/8 blocked traces with a qualifying frame-level crossing.
- Added the caveat that IF/OCSVM delay values use a `T_max` proxy frame score rather than native model trajectories.
- Refreshed `outputs/figures/F6_1_lstmvae_frame_scores.png` and `overleaf_ready/figures/F6_1_lstmvae_frame_scores.png` from `/private/tmp/ad_real_results_v2/LSTMVAE/frame_scores.png`.

Recommended oral defence:

- "We initially had a higher LSTM-VAE number, but after auditing the code we made evaluation deterministic and added a physics baseline. The headline number became lower but more defensible."
- "The simple peak-temperature rule is strong because the fault is fully blocked and binary. That is exactly why the report says a graduated fault structure would be the highest-impact improvement."
- "The LSTM-VAE's value is not a huge classification gap on this dataset; it is reproducible trajectory scoring and a more appropriate normal-only anomaly-detection setup."

## Open Follow-Ups

- Decide whether to replace the official `outputs/ad` directory with `/private/tmp/ad_real_results_v2`.
- Decide whether to copy the rerun result directories from `/private/tmp/ad_real_results_v2` into `outputs/ad`.
- Draft Dimitris's individual contribution paragraph once the exact personal ownership details are confirmed.

---

## Technical Choice Justifications (added 2026-06-04, for live assessment)

For each implementation choice in the AD pipeline, this section records the defensible rationale, the alternative considered, and the exact file/line where the choice lives. Five of the high-risk items (marked **(IN REPORT)**) are also justified inline in `overleaf_ready/report.tex` §V; the rest live only here.

### Tier 1 — almost certain viva questions

**T1.1 Leave-one-session-out cross-validation**
- *Implementation:* `model/ad/evaluate.py` (LOSO loop in `loso_evaluate`); the report describes it in §V.E.
- *Choice:* hold out one full session per fold; train on the remaining normal sessions only.
- *Alternatives considered:* stratified K-fold at the frame level (rejected: catastrophic intra-session leakage); leave-one-laptop-out (impossible: only one laptop); 80/20 holdout (rejected: with N = 20, a single split is too noisy for AUC ranking).
- *Defence:* with 20 sessions, LOSO maximises the number of independent test folds while preserving the session boundary, which is the natural unit of independence in the experimental design. Frame-level splits would leak heavily because consecutive frames within a session are highly autocorrelated. Cited as the standard one-class practice in Jakubowski et al. (2021) and Boggia et al. (2025).

**T1.2 Youden's J for threshold selection (IN REPORT)**
- *Implementation:* `model/ad/evaluate.py` (`compute_roc_pr` selects threshold at `argmax(tpr - fpr)`).
- *Choice:* operating threshold = argmax(TPR − FPR) on the aggregated LOSO scores.
- *Alternatives considered:* maximum F1 (would shift the threshold to favour the more frequent class); geometric mean of sensitivity and specificity (similar but penalises extreme imbalance harder); fixed false-positive rate (requires a deployment-time cost ratio we do not have).
- *Defence:* Youden's J weights sensitivity and specificity equally without prior assumption on the deployment-time cost of either error, and is the standard threshold-selection rule in the medical-screening and one-class AD literature.

**T1.3 ROC-AUC as headline metric (IN REPORT)**
- *Implementation:* `model/ad/evaluate.py` (`roc_auc_score` from sklearn).
- *Choice:* report ROC-AUC as the primary figure; PR-AUC secondary.
- *Alternatives considered:* PR-AUC headline (defensible if positives are rare and recall matters most); balanced accuracy (essentially a threshold-bound version of ROC-AUC).
- *Defence:* the 40 % positive-class proportion in our 8/20 dataset is an artefact of team-controlled experimental design, not a realistic deployment ratio. ROC-AUC is invariant to test-time class proportion; PR-AUC at p = 0.4 does not generalise to deployment where positives would be much rarer.

**T1.4 LSTM-VAE hyperparameters (IN REPORT)**
- *Implementation:* `model/ad/models.py` lines 164–183 (`LSTMVAE.__init__` defaults: `latent_dim=16, hidden_size=64, lr=1e-3, epochs=100, beta=0.5`).
- *Choices and defences:*
  - `latent_dim = 16`: small enough to force the encoder to compress the 3-channel × 1500-step input but large enough to span multiple modes of normal trajectories. Compression ratio (4500 → 16) forces summary rather than memorisation.
  - `hidden_size = 64`: within the 32–128 band reported for analogous low-channel-count industrial LSTM-AD work (Xu and Zhang 2025; Han et al. 2021).
  - `β = 0.5`: sub-unit β reduces KL pressure to avoid posterior collapse on the small (≤ 11 training sequences per fold) dataset. Setting β = 1 (standard VAE) on a dataset of this size frequently collapses the posterior to the prior.
  - `lr = 1e-3`: Adam default; no scheduling necessary at 100 epochs.
  - `epochs = 100`: empirical inspection of train/validation reconstruction loss showed plateau before this; not formally tuned.
- *Alternative considered:* hyperparameter search (rejected: ≤ 11 training sequences per fold makes any held-out validation set unreliable for tuning).

**T1.5 p95 session aggregation (IN REPORT — already in §V.D)**
- *Implementation:* `model/ad/models.py` `LSTMVAE.score_session` returns `np.percentile(frame_scores, 95)`.
- *Defence:* p95 ignores the lower 95 % of frames, which during a 1500-sample session means the cooldown phase (frames 1200–1500) and the baseline phase (0–300) cannot dilute the score. The 5 % retained corresponds to roughly the last 75 s of any session being above the threshold. p99 would be too sensitive to single-frame artefacts; p90 dilutes more of the stress phase.

### Tier 2 — likely viva questions

**T2.1 The 20 specific scalar features (IN REPORT)**
- *Implementation:* `model/ad/features.py` lines 21–33 (`SESSION_FEATURE_NAMES`).
- *Defence:* features grouped into four physically-grounded categories per the lumped-capacitance model in §II of the report: magnitude (T_max/T_min/T_range percentiles), shape (early/late/full slopes, derivative p95), persistence (autocorrelation, thermal time constant), threshold-crossing (frac_above and time_to for 38/42/46 °C). Feature count chosen so that with 11 training samples per LOSO fold the samples-to-features ratio (~0.55) is close to the per-sample budget for small-data anomaly detection.

**T2.2 T_range as third LSTM-VAE channel (IN REPORT)**
- *Implementation:* `model/ad/features.py` `time_series` line 162.
- *Defence:* explicit inductive bias — the chassis dynamic range is a physically-distinct signal from individual T_max and T_min trajectories, and supplying it as a channel saves the encoder from learning the subtraction implicitly. Cost is one extra LSTM input dimension, trivial.

**T2.3 Stress-onset auto-detection thresholds**
- *Implementation:* `model/ad/run.py` `_detect_stress_onset_s` uses `3σ OR 1.5 °C` (whichever larger) over a 240 s baseline window.
- *Defence:* 3σ catches statistical departures from the baseline distribution; 1.5 °C provides a floor so that quiet baselines (low σ) do not yield sub-degree spurious onsets driven by sensor noise. The 240 s baseline window is shorter than the 300 s nominal baseline so that the onset detector is not contaminated by stress onset itself when the stressor was started slightly early.

**T2.4 1500-sample length normalisation**
- *Implementation:* `model/ad/__init__.py` `EXPECTED_LEN = 1500`; `model/ad/run.py` `_normalize_length` truncates or terminal-value-pads.
- *Defence:* 1500 = 25 min × 60 s/min × 1 Hz is the nominal protocol length. Truncation discards trailing cooldown data when a session over-ran; terminal-value padding extends short sessions by holding the last observed temperature, which preserves the trajectory shape at the cost of slightly biasing the cooldown statistics for the affected sessions. Beginning-of-session padding was rejected because it would corrupt the baseline phase, which is what the model learns from.

**T2.5 Median-of-per-session-p99 frame threshold (IN REPORT — already in §V.F)**
- *Implementation:* `model/ad/models.py` lines 28–47 (`_normal_frame_percentile_threshold`).
- *Defence:* a pooled p99 (concatenate all training-normal frames, then take p99) would let one outlier session set the threshold for every fold. The two-stage estimator (per-session p99, then median across sessions) is the robust analogue.

**T2.6 200 s transition window for the after-transition delay variant**
- *Implementation:* `model/ad/evaluate.py` `detection_delay_after_transition_s` (window default 200 s).
- *Defence:* synthetic-trajectory inspection during development showed a dip-and-recovery pattern in the VAE reconstruction error during the first ≈ 100–200 s after stress onset. 200 s is the conservative upper bound; the metric is intentionally under-reporting early detection rather than over-reporting it.

### Tier 3 — possible questions

**T3.1 Why z-score reconstruction error against the training distribution**
- *Implementation:* `model/ad/models.py` `LSTMVAE.score_frames` z-scores per-frame MSE against the training-set mean and std.
- *Defence:* makes the per-frame score scale-invariant to the absolute reconstruction loss magnitude, which differs between folds. Without z-scoring the same operating threshold could not be applied across folds.

**T3.2 Isolation Forest `n_estimators=200, contamination='auto'`**
- *Implementation:* `model/ad/models.py` lines 58–67.
- *Defence:* `n_estimators=200` is the sklearn-recommended starting point for small (n < 1000) datasets. `contamination='auto'` is appropriate because the true outlier rate in the training data is unknown — by construction we believe the training set is all normal, so any non-zero contamination estimate would be a heuristic.

**T3.3 OCSVM `RBF kernel, ν=0.1, gamma='scale'`**
- *Implementation:* `model/ad/models.py` lines 111–115.
- *Defence:* RBF is the standard non-linear default. `ν=0.1` corresponds to an expected outlier proportion of 10 % in the training set, which is a defensible upper bound for what we assume is a clean normal-only set. `gamma='scale'` uses sklearn's automatic scaling, removing one hyperparameter that we lack budget to tune.

**T3.4 30-frame sustained-window for the sustained delay variant**
- *Implementation:* `model/ad/evaluate.py` `detection_delay_sustained_s` (default 30 frames = 30 s).
- *Defence:* a single-frame crossing can be sensor noise; 30 consecutive seconds of above-threshold readings rejects almost all noise spikes while still allowing detection well before the steady state.

**T3.5 60-s window, 5-s step for IF/OCSVM per-frame proxy**
- *Implementation:* `model/ad/features.py` `window_features` defaults.
- *Defence:* 60 s is the shortest window that smooths sensor noise without losing the heating ramp; 5 s step gives 12× oversampling per window for trajectory smoothness in the per-frame proxy plot.

**T3.6 Feature temperature thresholds 38/42/46 °C**
- *Implementation:* `model/ad/features.py` lines 96–100.
- *Defence:* these correspond approximately to the 5th, 50th, and 95th percentiles of T_max observed across the 20 sessions, giving the IF/OCSVM feature vector three threshold-crossing features spanning the dataset's dynamic range. Other thresholds were not tested.

**T3.7 Adam optimiser**
- *Implementation:* `model/ad/models.py` LSTMVAE training loop.
- *Defence:* Adam's adaptive learning rate is robust to the small training-set size and gradient sparsity of the LSTM-VAE. SGD with momentum would also work but requires learning-rate scheduling that we did not tune.

**T3.8 Synthetic data cooldown τ multiplier**
- *Implementation:* `model/ad/synthetic.py` (factor of 1.8× for blocked).
- *Defence:* blocked-vent cooling τ is bounded below by the unblocked τ (blocked cannot cool faster) and bounded above by the heating τ × ~2 in physical experiments on consumer-laptop chassis. 1.8× is at the upper end of that range, intentionally chosen so the synthetic dataset is deliberately separable; the synthetic generator is for pipeline validation only and is not claimed to predict real-data performance.

**T3.9 Default seed `DEFAULT_SEED = 20260531`**
- *Implementation:* `model/ad/__init__.py`.
- *Defence:* a fixed seed makes results reproducible across runs on the same library/hardware stack. The specific value (the development date) is arbitrary; any other fixed integer would serve the same role.

### Cross-reference: not justified anywhere, currently relying on convention

Items where the defence is "standard practice" and no stronger argument exists:
- `StandardScaler` for IF/OCSVM input features (sklearn-standard preprocessing for distance-based and tree-based models).
- 1 Hz sample rate (TCView Plane export constraint, not a choice).
- LSTM-VAE reconstruction loss = MSE (vs Gaussian NLL, which would be equivalent for fixed variance).

If a viva examiner asks about any of the above the answer is "standard practice; no domain-specific argument".
