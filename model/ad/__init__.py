"""Unsupervised anomaly detection for TCView Plane exports."""

__version__ = "0.1.0"
DEFAULT_SEED = 20260531

# Nominal protocol anchors. Real sessions can carry an auto-detected
# stress_onset_s value that overrides BASELINE_END_S during delay scoring.
SESSION_DURATION_S = 25 * 60          # 5 baseline + 15 stress + 5 cooldown
BASELINE_END_S = 5 * 60               # nominal stress onset at t=300s
STRESS_END_S = 20 * 60                # nominal cooldown onset at t=1200s
SAMPLE_RATE_HZ = 1.0
EXPECTED_LEN = int(SESSION_DURATION_S * SAMPLE_RATE_HZ)  # 1500 samples nominal

# Sessions are truncated or padded to EXPECTED_LEN before LSTM-VAE stacking.
# Raw recordings varied from about 20 to 31 minutes, so only large deviations
# should trigger warnings.
SESSION_LEN_TOLERANCE_S = 360
