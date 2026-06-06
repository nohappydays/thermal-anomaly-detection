# Recorded sessions — TCView Plane exports

This folder contains the 20 thermal sessions used in the report's LOSO evaluation. Each session is a 25-minute recording of a Toshiba laptop under controlled thermal stress, exported from TCView in **Plane** mode at 1 Hz.

## Filename convention

```
S<NNN>_<YYYY-MM-DD>_toshiba_<normal|blocked|faulty>_<ambient>C_a.xlsx
```

- `<NNN>` — sequential session ID (zero-padded)
- `<YYYY-MM-DD>` — recording date
- `<normal|blocked|faulty>` — `faulty` is a legacy synonym for `blocked` (used in early recordings); the loader normalises both to `blocked`
- `<ambient>C` — measured ambient air temperature in °C at recording start
- `_a` — single-laptop ambient run (room baseline)

## Recording protocol

Each session is 25 minutes:

| Phase | Duration | Description |
|---|---|---|
| Baseline | 5 min | Laptop idle, no load |
| Stress | 15 min | `mprep.info/gpu` running (CPU+GPU load) |
| Cooldown | 5 min | Stressor stopped, fans run down |

For **blocked** sessions the rear vent and/or intake is physically obstructed with a folded cloth from t = 0. For **normal** sessions both vents are clear.

Stress onset is operator-triggered, so the actual `t_start_stress` varies by ±30 s per session. The loader auto-detects the per-session onset from the T_max curve (`run._detect_stress_onset_s`) and attaches it as a `stress_onset_s` column.

## TCView Plane column format

| Column | Type | Notes |
|---|---|---|
| `time` | datetime / string | 1 Hz timestamps |
| `Lowest Temperature` | string with °C suffix (e.g. `"23.4°C"`) | T_min |
| `Highest Temperature` | string with °C suffix | T_max |

`model/excel_loader.py` strips the unit, parses timestamps, and normalises to the project schema (`t_s`, `T_max`, `T_min`, `T_range`, `condition`, `session_id`, `laptop_id`).

## Sessions

| Sessions | Condition | Date |
|---|---|---|
| S001 (22 °C) | normal | 2026-06-01 |
| S001 (27 °C), S002–S008 | blocked / faulty | 2026-05-31 / 2026-06-01 |
| S009–S014, S016–S020 | normal | 2026-06-02 |

20 sessions total (12 normal + 8 blocked). S015 is **quarantined** — see below.

## Quarantine

`_QUARANTINE_S015_normal_24C.xlsx` is intentionally prefixed so it does *not* match the loader glob (`S*_*toshiba*.{csv,xlsx}`). S015's recording was interrupted partway through the stress phase and the resulting trace is not protocol-compliant. It is shipped here for transparency only; do not include it in evaluation runs.

## Note on labels

S001's "normal" recording exhibits unusually high baseline T_max (~28 °C) — likely a warm chassis from a prior unlogged run. This is discussed in the report as the dominant source of normal-class variance and motivates the "no-headline-PR-AUC" choice. See `outputs/ad/frame_scores.png` and the report §VI for details.
