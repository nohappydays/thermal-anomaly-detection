# Thermal Recording Protocol v3 — ASML Internship 5ARIP10

> **Why v3?** After the 2026-05-28 team meeting the scope changed: one laptop, binary fault, CSV-only, ~20 sessions. v2 was written for the 5-condition video-based plan and is now obsolete. The final real-data run used a 5 min baseline + 15 min stress + 5 min cooldown structure, so this protocol records the current 25-minute session target.

**Per-session deliverable:**
1. One TCView Plane export CSV/XLSX covering the full 25-min session.
2. One short session log (plain text) with timing + ambient + notes.
3. (Optional but recommended) the thermal video for backup and possible later post-hoc extraction.

---

## 0. Constants — set once, never change between sessions

| Constant | Value | Why |
|---|---|---|
| Laptop | Toshiba (`toshiba_a01`) | Single-laptop scope |
| Camera | TOPDON TC001 | Single sensor |
| Camera angle | 90° (top-down) | Removes geometric variability |
| Camera distance | **50 cm** above keyboard | Fixed framing across sessions |
| Stressor | mprep.info/gpu (simultaneous CPU + GPU) | Single stressor, single click |
| Stressor archive | `https://web.archive.org/web/<date>/https://mprep.info/gpu` | For methods-section citation |
| TCView mode | Plane measurement, palette **locked at 15 – 70 °C** | Avoids saturation observed in pilots |
| Plane region | Whole keyboard + vent strip | Captures heat field that drives T_max / T_min |
| Sample rate | 1 Hz (TCView default for Plane) | Matches what the export gives us |

---

## 1. Per-session steps

### 1.1 Equipment setup (5 min)
1. Verify camera is **50 cm above** the closed-lid plane of the laptop, lens centred on the keyboard, perpendicular to the surface.
2. Connect TC001 to phone, launch TCView, lock palette range to **15 – 70 °C** (manual mode).
3. Place the Plane overlay over the whole keyboard area + vent strip; do not move it once placed.
4. Start screen recording on the phone (backup video — useful even though primary analysis is CSV).
5. Note the **ambient temperature** with any thermometer or phone weather app (best-effort logging, even though the project treats ambient as noise).

### 1.2 Baseline phase (5 min, t = 0 – 300 s)
1. Laptop idle, browser open to the stressor page but **not started**.
2. Start TCView measurement export.
3. Note the wall-clock start time. The "session start" anchor is this moment, t = 0 s.
4. Do not touch the laptop. Let it sit for the full 5 minutes.

### 1.3 Stress phase (15 min, t = 300 – 1200 s)
1. Click the start button on mprep.info/gpu.
2. Note the wall-clock time of the click — this is **stress onset**.
3. Leave the stressor running for the full 15 minutes.
4. Click stop on the stressor page at exactly **t = 1200 s**.

### 1.4 Recorded cooldown phase (5 min, t = 1200 – 1500 s)
1. Keep the TCView measurement export running while the laptop cools.
2. Leave the laptop untouched and unobstructed except that blocked-session tape may be removed only after the export stops.
3. **Stop the measurement export at exactly t = 1500 s** (25 min total from session start).

### 1.5 Cool-between phase (~15 min, off-camera)
1. Move the laptop to a cool surface or open a window — the goal is to return to within 5 °C of ambient before the next session.
2. The next session must wait until the chassis is visibly back near ambient (touch test: not warm).
3. If you need to record back-to-back without waiting, mark the second session as "warm-start" in the session log so it can be excluded later.

---

## 2. Session log (write while recording)

Open a text file or paper notebook. Record (at minimum):

```
session_id:    S00N_2026-05-30_toshiba_<normal|blocked>_a
date_time:     2026-05-30 14:30
condition:     normal | blocked         <-- BINARY, must be filled
operator:      <name>
ambient_C:     22.5                     <-- best effort, room thermometer or weather app
stressor_link: https://mprep.info/gpu
TCView_range:  [15, 70]                 <-- confirm locked
camera_dist:   50 cm                    <-- confirm 50
laptop_idle_before: 15 min              <-- thermal pre-conditioning
notes:         <anything unusual>

Phase timestamps (HH:MM:SS, wall clock; also t in seconds from baseline start):
  00:00:00  t=0     baseline start, TCView export started
  00:05:00  t=300   stressor START (mprep.info/gpu click)
  00:20:00  t=1200  stressor STOP, cooldown START
  00:25:00  t=1500  TCView export STOP
```

These timestamps are pasted into the analysis script when labelling the phases.

---

## 3. Fault condition specifics

### Blocked sessions
- **Both intake AND exhaust vents fully covered.** Use the same tape across all blocked sessions for consistency (record the tape brand in the session log once).
- Apply tape **before** starting the recording — the laptop should be blocked for the *entire* session including baseline (so the model learns "starts cool, heats up faster" rather than "got blocked mid-session").
- After the stress phase, remove the tape during the cool-between window.

### Normal sessions
- No tape. Vents fully open. Keep the laptop on the same surface as blocked sessions to keep airflow conditions comparable.

---

## 4. Safety limits

Stop the stress phase early if:
- The Plane T_max reading exceeds **80 °C** (Toshiba thermal envelope ceiling).
- The laptop fan produces abnormal sounds, or smell/smoke appears.
- The phone running TCView is overheating.

Early-stop sessions go in the session log with an `early_stopped_at: <t_seconds>` field. The recorded data may still be usable up to the stop point.

---

## 5. Quality gate — what makes a session usable

Before adding a session to the dataset, check:

- [ ] Baseline = 5 min, stress = 15 min, cooldown = 5 min, total = 25 min.
- [ ] TCView palette stayed locked at 15 – 70 °C.
- [ ] Plane region didn't move mid-recording.
- [ ] Camera at 50 cm, top-down throughout.
- [ ] Session log filled in (especially `condition`, `ambient_C`, phase timestamps).
- [ ] Laptop started near ambient (chassis not warm to touch at t = 0).
- [ ] CSV export has 1500 rows (one per second) ± a few.

Sessions failing any of these go to `data/csv/quarantine/` and are excluded from the model.

---

## 6. Target dataset

20 sessions total, on the Toshiba only. Suggested split:

| Condition | Sessions | Notes |
|---|---|---|
| **Normal** | 10 | Spread across at least two different days to capture ambient variation |
| **Blocked** | 10 | Same spread |

This gives:
- Per-class training pool large enough for the unsupervised model.
- Each LOSO fold leaves out 1 session; ROC computed over 20 folds.
- Roughly 8–9 hours total including recorded sessions (20 × 25 min) and off-camera cooling between sessions.

---

## 7. File naming

```
S<NNN>_<YYYY-MM-DD>_toshiba_<condition>_<source>.<ext>
```

Where:
- `<NNN>` = three-digit session number (`001`, `002`, …)
- `<condition>` = `normal` or `blocked`
- `<source>` = `a` for authentic (real recording) or `s` for synthetic (model-generated)
- `<ext>` = `csv` for the Plane export, `mp4` for the backup video

Examples:
```
S001_2026-05-30_toshiba_normal_a.csv
S001_2026-05-30_toshiba_normal_a.mp4
S012_2026-06-02_toshiba_blocked_a.csv
SYN_001_2026-06-10_toshiba_blocked_s.csv
```

All authentic CSVs go to `data/csv/`; all synthetic CSVs go to `data/csv/synthetic/`.

---

## 8. Quick reference card (print this)

```
PRE                                  RECORD
□ Camera 50 cm top-down              □ TCView measurement START (t=0)
□ TCView range LOCKED 15-70°C        □ Baseline 5 min (idle)
□ Plane overlay placed               □ mprep.info/gpu CLICK (t=300)
□ Stressor page open, NOT started    □ Stress 15 min
□ Ambient temp noted                 □ Stop stressor (t=1200)
□ Tape applied IF blocked session    □ Save export, write session log
□ Session log file created
                                     POST
                                     □ Cooldown recorded until t=1500
                                     □ Add to dataset folder
                                     □ Confirm quality gate
                                     □ Cool laptop ~15 min before next session
```
