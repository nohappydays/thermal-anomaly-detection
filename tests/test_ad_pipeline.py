import unittest

import numpy as np

from model.ad.evaluate import (
    compute_roc_pr,
    detection_delay_after_transition_s,
    detection_delay_s,
    detection_delay_sustained_s,
)
from model.ad.models import LSTMVAE, PeakTMaxRuleDetector
from model.ad.synthetic import generate_dataset


class MetricTests(unittest.TestCase):
    def test_compute_roc_pr_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])

        metrics = compute_roc_pr(scores, labels)

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)

    def test_detection_delay_variants(self):
        frame_scores = np.zeros(30)
        frame_scores[10:] = 2.0

        self.assertEqual(detection_delay_s(frame_scores, 1.0, stress_onset_s=5), 5.0)
        self.assertEqual(
            detection_delay_after_transition_s(
                frame_scores, 1.0, stress_onset_s=5, transition_duration_s=3
            ),
            5.0,
        )
        self.assertEqual(
            detection_delay_sustained_s(
                frame_scores, 1.0, stress_onset_s=5, sustained_frames=4
            ),
            5.0,
        )


class DetectorTests(unittest.TestCase):
    def test_peak_tmax_rule_scores_temperature_trace(self):
        sessions, _ = generate_dataset(n_normal=2, n_blocked=1, seed=123)
        normal = [s for s in sessions if s["condition"].iloc[0] == "normal"]
        blocked = [s for s in sessions if s["condition"].iloc[0] == "blocked"]

        detector = PeakTMaxRuleDetector().fit(normal)
        session = blocked[0]

        self.assertTrue(np.isfinite(detector.frame_threshold_))
        self.assertAlmostEqual(detector.score_session(session), session["T_max"].max())
        np.testing.assert_allclose(detector.score_frames(session), session["T_max"].to_numpy())

    def test_lstmvae_scoring_is_deterministic_in_eval(self):
        sessions, _ = generate_dataset(n_normal=3, n_blocked=1, seed=321)
        normal = [s for s in sessions if s["condition"].iloc[0] == "normal"]
        blocked = [s for s in sessions if s["condition"].iloc[0] == "blocked"]

        detector = LSTMVAE(epochs=2, random_state=321).fit(normal)
        first = detector.score_frames(blocked[0])
        second = detector.score_frames(blocked[0])

        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
        self.assertTrue(np.isfinite(detector.frame_threshold_))


if __name__ == "__main__":
    unittest.main()
