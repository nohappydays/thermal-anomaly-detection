# Thermal Anomaly Detection on Laptop Surfaces

> **Project**: 5ARIP10 ATWS Team Internship, TU/e (Q3 → Q4 2026), in collaboration with ASML.

Detect thermal anomalies from infrared surface-temperature recordings of a Toshiba laptop under controlled stress, used as a proxy for an ASML machine subsystem. The pipeline trains four unsupervised detectors (Peak T_max rule, Isolation Forest, One-Class SVM, LSTM-VAE) on normal session data only and evaluates them under leave-one-session-out cross-validation.

## Quick start

```bash
git clone <this-repo>
cd <this-repo>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Sanity check: unit tests
python -m unittest tests.test_ad_pipeline

# Reproduce the reported results (~5 min on CPU, faster on GPU)
python -m model.ad.run --data data/csv \
    --detector peak if ocsvm vae --epochs 100
```

Outputs land in `outputs/ad/`:

- `comparison.csv` — one row per detector, all headline metrics
- `<detector>/results.json` — full per-fold scores and metadata
- `<detector>/frame_scores.csv` — long-format per-frame scores
- `<detector>/roc.png`, `score_hist.png`, `frame_scores.png` — diagnostic plots

## Repository layout

```
.
├── README.md                              ← you are here
├── LICENSE                                MIT
├── requirements.txt                       pinned dependencies
├── model/
│   ├── ad/                                final anomaly-detection pipeline
│   │   ├── __init__.py                      protocol constants + DEFAULT_SEED
│   │   ├── synthetic.py                     synthetic Plane sessions (smoke tests)
│   │   ├── features.py                      session-level + window + time-series features
│   │   ├── models.py                        IF, OC-SVM, LSTM-VAE, Peak-T_max baseline
│   │   ├── evaluate.py                      LOSO CV, metrics, plots
│   │   ├── run.py                           CLI entry point
│   │   ├── README.md                        sub-package design notes
│   │   └── CHANGELOG.md                     code-change log with retraction notes
│   └── excel_loader.py                    TopInfrared .xlsx parser
├── data/csv/                              20 recorded TopInfrared Plane sessions + README
├── tests/                                 regression tests for the AD path
```

## Method at a glance

| Component | Choice | Notes |
|---|---|---|
| Sensor / format | TopInfrared Plane (.xlsx) at 1 Hz | Per-frame min and max chassis temperature |
| Session length | 25 min (5 baseline + 15 stress + 5 cooldown) | 1500 samples after length normalisation |
| Stress source | `mprep.info/gpu` (CPU + GPU load) | Operator-triggered, ±30 s timing slop |
| Fault label | Vent obstruction with folded cloth | Filename uses `blocked` or `faulty` interchangeably |
| Training | Normal sessions only | One-class / unsupervised |
| Validation | Leave-one-session-out CV | Per-fold frame threshold from training-normal sessions |
| Headline metric | ROC-AUC | Class proportion is an experimental artifact, not a deployment prior |
| Delay metric | Three variants | raw / after-transition (200 s) / sustained (30 consecutive frames) |


## Reproducibility

- All RNGs (numpy, torch, sklearn) seeded from `DEFAULT_SEED = 20260531` (override via `--seed`).
- LSTM-VAE training samples from the latent posterior; **scoring decodes from the posterior mean**, so repeated scoring of the same session is bit-identical (enforced by a unit test).
- Per-fold frame thresholds use the **median of per-session p99 frame scores** from training-normal sessions, making the threshold robust to one abnormal normal-labelled session.
- Real-data results in the report and `outputs/ad/comparison.csv` are reproduced by the `--data data/csv` command above on the pinned `requirements.txt` stack.

## Data

The 20 recorded sessions live in `data/csv/` as TopInfrared Plane `.xlsx` exports. See `data/csv/README.md` for the filename convention, recording protocol, column format, and notes on the quarantined S015 session.

The recordings were taken on a single Toshiba laptop. **No ASML machine data was collected** — the laptop serves as a small-scale thermal proxy for the failure-mode pattern of interest (airflow obstruction → elevated steady-state surface temperature and slowed thermal time constant).

## Tests

```bash
python -m unittest tests.test_ad_pipeline
```

Covers metric helpers, all three delay variants, the Peak-T_max baseline, and deterministic LSTM-VAE scoring.


## Authors

5ARIP10 ATWS Team Internship, TU/e, in collaboration with ASML.

- Dimitris Laspias
- Srikar Narayan Rao Krishna Raja
- Aleks Atanasov
- Madhav Veluru

## License

MIT — see `LICENSE`.
