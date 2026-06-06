# Thermal Recording Protocol v2 — ASML Internship 5ARIP10

> **Why v2?** The May 12 sessions revealed protocol issues that compromise the data: the CPU-only recording started already hot from a previous run (negative ramp slope), neither session contains a labeled C0 baseline, and TCView's auto-scaling palette risks decoding drift in the video. This protocol fixes all three.

**Per-session deliverable:**
1. One thermal video (.mp4)
2. One Plane export (.xlsx) covering the same time window
3. One session config JSON for the video pipeline
4. One plain-text session log noting the start time of each phase + any anomalies

---

## 0. Before you arrive at the lab

- Charge laptop, phone, and tripod batteries
- Bring tape or a label maker — fiducial markers go on the laptop chassis
- Decide today's condition: **C0 / C1 / C2 / C3 / C4**
- Read [Section 8: Condition specifics](#8-condition-specifics) for what to run

---

## 1. Equipment setup (10 min)

1. Place laptop flat on a non-reflective surface, vents unobstructed unless C3
2. Mount TC001 on tripod or clamp arm, **30–35 cm above the laptop**, pointing straight down (90 ° to chassis)
3. Frame the shot so the laptop fills the central 70 % of the image — leave a margin so small camera shifts don't move features out of frame
4. Place 4 small reflective stickers at the laptop's corners (fiducials). These let the pipeline register the ROI across sessions even if the tripod shifts between recordings
5. Note ambient temperature with a separate thermometer; record in the session log

---

## 2. TCView configuration (5 min)

**Critical:** TCView's auto-ranging palette is the largest single source of decoding error. Lock it before recording.

1. Open TCView, connect TC001 via USB-C
2. **Lock the temperature range:**
   - Tap settings → measurement → choose "Manual range"
   - Set low = **15 °C**, high = **55 °C** (covers all expected conditions on the Toshiba)
   - Verify the scale bar legend now shows 15 / 55 — do **NOT** proceed if it still auto-ranges
3. Palette: keep **ironbow** (we have a calibrated LUT for it)
4. Measurement overlays:
   - Add a **Plane** covering the whole keyboard area + vents
   - Add a **Dot** at the expected hotspot (above the CPU)
   - Both must be inside the planned ROI — not over the scale bar
5. Start a **measurement export** in TCView (it will write the .xlsx in the background at 1 Hz)
6. Start the **screen recording** (phone) — show the full TCView interface including the scale bar
7. **Always start the screen recording before TCView measurement so the video covers a superset of the export.**

---

## 3. Thermal pre-conditioning (10 min)

The laptop must start each session from a known thermal state. Skipping this step is what made the May 12 CPU-only data unusable.

- Laptop must have been **powered off or idle (no foreground tasks) for ≥ 10 minutes** before the session begins
- T_max from the TCView preview should be within **+2 °C of ambient** before you start recording
- If above that, wait longer (or turn the laptop off entirely for 5 min)
- Record ambient + initial T_max in the session log

---

## 4. Mandatory recording phases

Every session — **regardless of condition** — must contain these phases, in order:

| Phase | Duration | What's happening |
|---|---|---|
| **C0 baseline** | 5 min | Laptop idle, no foreground tasks. Captures the "normal" reference for this session. |
| **Stress phase** | 10–20 min | Apply the condition's stressor (see §8). Stop only when T_max plateaus or reaches the safety limit. |
| **Cooldown** | 5 min | Stop the stressor, leave laptop running idle, watch the curve return toward baseline. |

Why: the unsupervised autoencoder track needs **normal data**, and the 5-min C0 segment from every session yields that without scheduling dedicated baseline-only sessions. The cooldown gives us a thermal-time-constant measurement and a second look at the baseline as the system relaxes.

---

## 5. Session log (write while recording)

Open a text file or a paper notebook. Record (at minimum):

```
session_id:    S001_C1_2026-05-15
laptop_id:     toshiba_a01
date_time:     2026-05-15 14:30
ambient_C:     22.5
initial_T_max_C: 24.1
camera_height_cm: 32
camera_angle_deg: 90
TCView_range_C: [15, 55]    # confirm you locked it
operator:      Jim
notes:         <anything unusual>

Phase log (HH:MM:SS, t in seconds from video start):
  00:00:00  recording start
  00:00:15  TCView measurement export started
  00:00:30  C0 baseline begin
  00:05:30  stress phase begin   <-- start your stressor command here
  00:18:30  stress phase end     <-- stop stressor
  00:18:35  cooldown begin
  00:23:35  cooldown end / stop recording
```

The phase timestamps go directly into the session config JSON's `phases` block.

---

## 6. Stop & file the data (5 min)

1. Stop TCView measurement export — the .xlsx is now in TCView's Files folder
2. Stop the phone screen recording
3. **Immediately copy both files to your laptop** with a consistent name: `S001_C1_video.mp4` and `S001_C1_plane.xlsx`
4. Open `model/configs/session_template.json`, save a copy as `model/configs/session_S001_C1.json`, fill in:
   - `session_id`, `video_path`, `output_dir`
   - `laptop` block (id, make, model, cpu, ram_gb, has_dedicated_gpu)
   - `condition` block (label, description, stressor, blockage_pct if applicable)
   - `phases` block (translate the times from your session log to seconds)
   - `thermal_scale.t_min_c` and `t_max_c` (= 15 and 55 if you locked TCView per §2)
   - `environment.ambient_temp_c` from your session log
5. Run the calibration check: `python pipeline_v2.py calibrate --config configs/session_S001_C1.json` and visually verify ROI + scale bar in `processed/S001_C1/calib_overlay.png`
6. If ROI is off, edit the JSON's `roi` block and re-run calibrate until it's right
7. Process: `python pipeline_v2.py process --config configs/session_S001_C1.json`

---

## 7. Safety limits

Stop the stress phase immediately if:
- T_max exceeds **75 °C** anywhere visible in the frame
- HWiNFO reports CPU > 95 °C internal
- The laptop fan starts producing abnormal sounds
- Any smell or visible smoke

The Toshiba's published thermal envelope tops out around 70 °C external for the AMD APU class. Going above is not informative for fault detection (we lose linearity in heat transfer) and risks the only laptop you have.

---

## 8. Condition specifics

### C0 — Baseline (idle)
- No additional stressor — the C0 segment of any session is the baseline
- Optionally: record a dedicated 30-min C0-only session per laptop, once

### C1 — CPU overheating
- Stressor: `stress-ng --cpu 0 --cpu-method matrixprod --timeout 15m` (Linux/WSL) or **Prime95** Small FFT mode (Windows)
- Run all available logical cores
- Note Prime95 / stress-ng command in the session log

### C2 — GPU overheating
- The Toshiba has no dedicated GPU — integrated AMD APU only
- Use **clpeak** or **Unigine Heaven** instead of FurMark (which won't load an integrated GPU meaningfully)
- Thermal effect will be milder than C1; budget 20 min stress phase
- For laptops *with* dedicated GPU: FurMark or `gpu-burn` with default settings

### C3 — Airflow blockage
- Pre-cut graduated tape masks: 25 %, 50 %, 75 %, 100 % vent coverage
- Sub-sessions: C3_25 / C3_50 / C3_75 / C3_100 (one per blockage level)
- Stressor underneath the blockage: same as C1 (so we isolate the airflow effect)
- Apply the mask **before** the stress phase begins, leave on for the entire stress + cooldown

### C4 — Combined
- Run C1's stressor with C3_50 blockage active
- Demonstrates fault stacking; the most realistic "multiple things going wrong" case

---

## 9. Quality gate — what makes a session usable

Before adding a session to the dataset, check:

- [ ] Initial T_max within +2 °C of ambient (pre-conditioning worked)
- [ ] TCView palette range stayed locked throughout (no auto-scale events)
- [ ] Video and Excel cover the same time window (Excel ⊆ Video)
- [ ] All three phases present (C0 baseline, stress, cooldown)
- [ ] Stress phase is at least 10 min long
- [ ] No camera movement mid-recording (fiducial markers visible throughout)
- [ ] Session log filled in with all required fields
- [ ] Pipeline calibrate step produced an overlay image where ROI fits the laptop

Sessions failing any of these go in a `quarantine/` folder, not the training set.

---

## 10. Target dataset (~32 sessions, 4–5 lab days)

| Condition | Sessions per laptop | Notes |
|---|---|---|
| C0 dedicated | 1 × 30 min | One per laptop, validates the baseline class |
| C1 | 4 × 25 min | Two morning + two afternoon to capture ambient drift |
| C2 | 4 × 25 min | Integrated GPU is mild — repeat to get separation |
| C3_25 / C3_50 / C3_75 / C3_100 | 3 each | Total 12 |
| C4 | 4 × 25 min | Combined |
| Quarantine / pilots | as needed | Don't count toward 32 |

With ≥ 2 laptops this doubles, giving cross-laptop generalization data.

---

## 11. Quick reference card (print this)

```
PRE                                  RECORD
□ Laptop off ≥10 min                 □ Phone screen-record START
□ Ambient T noted                    □ TCView measurement START
□ Tripod fixed 30–35 cm above        □ C0 baseline 5 min
□ TCView range LOCKED 15–55°C        □ Stressor START — note time
□ Plane + Dot overlays placed        □ Stressor 10–20 min
□ Fiducial stickers on chassis       □ Stressor STOP — note time
□ Session log file created           □ Cooldown 5 min
                                     □ TCView STOP, phone STOP, copy files

POST
□ Fill session config JSON
□ Calibrate: pipeline_v2.py calibrate
□ Verify ROI overlay
□ Process: pipeline_v2.py process
□ Check run_info.json, drift_events.json (should be empty)
□ Add to dataset OR quarantine
```
