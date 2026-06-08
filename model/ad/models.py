"""Detector implementations for anomaly detection on TopInfrared Plane exports.

Every detector exposes fit(), score_session(), score_frames(), and name.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .features import session_features_matrix, time_series, window_features

log = logging.getLogger("ad.models")


@runtime_checkable
class Detector(Protocol):
    name: str
    def fit(self, normal_sessions: list[pd.DataFrame]) -> "Detector": ...
    def score_session(self, session: pd.DataFrame) -> float: ...
    def score_frames(self, session: pd.DataFrame) -> NDArray: ...


def _normal_frame_percentile_threshold(
    detector: Detector,
    normal_sessions: list[pd.DataFrame],
    percentile: float = 99.0,
) -> float:
    """Calibrate a robust per-frame alarm threshold from normal sessions.

    Use the median of per-session percentile scores instead of a pooled
    percentile. This keeps one abnormal normal-labelled session from setting
    the alarm threshold for every fold.
    """
    session_thresholds = []
    for s in normal_sessions:
        frame_scores = np.asarray(detector.score_frames(s), dtype=np.float64)
        frame_scores = frame_scores[np.isfinite(frame_scores)]
        if frame_scores.size:
            session_thresholds.append(float(np.percentile(frame_scores, percentile)))
    if not session_thresholds:
        return float("nan")
    return float(np.median(session_thresholds))


class IsolationForestDetector:
    """Train an IsolationForest on per-session scalar features.

    Per-frame scores are produced by sliding a window across the session and
    treating each window as a "mini-session" through the same feature pipeline.
    """
    name = "IsolationForest"

    def __init__(self, contamination: float | str = "auto", random_state: int = 20260531):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        self._model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()

    def fit(self, normal_sessions: list[pd.DataFrame]) -> "IsolationForestDetector":
        X, _ = session_features_matrix(normal_sessions)
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X))
        self.frame_threshold_ = _normal_frame_percentile_threshold(self, normal_sessions)
        log.info("IF fitted on %d normal sessions (%d features)", len(normal_sessions), X.shape[1])
        return self

    def score_session(self, session: pd.DataFrame) -> float:
        X, _ = session_features_matrix([session])
        Xs = self._scaler.transform(X)
        # Flip sign so higher scores always mean more anomalous.
        return float(-self._model.score_samples(Xs)[0])

    def score_frames(self, session: pd.DataFrame, window_s: int = 60, step_s: int = 5) -> NDArray:
        """Slide a window across the session and score each one independently."""
        feats, centres = window_features(session, window_s=window_s, step_s=step_s)
        if feats.shape[0] == 0:
            return np.zeros(len(session))
        # IF is a session-level model, so per-frame delay uses a rolling
        # temperature proxy rather than native IF scores.
        score_series = np.zeros(len(session), dtype=np.float64)
        for centre, f in zip(centres, feats):
            i = int(round(centre))
            if 0 <= i < len(score_series):
                score_series[i] = float(f[0])
        # Center against the session baseline so hotter windows score higher.
        baseline = np.mean(score_series[score_series > 0][:60]) if np.any(score_series > 0) else 0.0
        score_series = np.where(score_series > 0, score_series - baseline, 0.0)
        last = 0.0
        for i in range(len(score_series)):
            if score_series[i] == 0.0:
                score_series[i] = last
            else:
                last = score_series[i]
        return score_series


class OneClassSVMDetector:
    """One-Class SVM on per-session scalar features. RBF kernel."""
    name = "OneClassSVM"

    def __init__(self, nu: float = 0.1, gamma: float | str = "scale", random_state: int = 20260531):
        from sklearn.svm import OneClassSVM
        from sklearn.preprocessing import StandardScaler
        self._model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
        self._scaler = StandardScaler()
        self._rng = random_state

    def fit(self, normal_sessions: list[pd.DataFrame]) -> "OneClassSVMDetector":
        X, _ = session_features_matrix(normal_sessions)
        self._scaler.fit(X)
        self._model.fit(self._scaler.transform(X))
        self.frame_threshold_ = _normal_frame_percentile_threshold(self, normal_sessions)
        log.info("OC-SVM fitted on %d normal sessions", len(normal_sessions))
        return self

    def score_session(self, session: pd.DataFrame) -> float:
        X, _ = session_features_matrix([session])
        Xs = self._scaler.transform(X)
        return float(-self._model.decision_function(Xs)[0])

    def score_frames(self, session: pd.DataFrame) -> NDArray:
        """Same rolling-mean proxy as IsolationForestDetector.score_frames."""
        return IsolationForestDetector.score_frames(self, session)  # share logic


class PeakTMaxRuleDetector:
    """Simple physical baseline: session score = maximum observed T_max.

    It is not trained statistically, but it still participates in the same LOSO
    metric pipeline as the learned detectors.

    Per-frame scores are the instantaneous T_max values, making the delay metric
    interpretable as the first time the thermal trace crosses the selected
    temperature threshold.
    """
    name = "PeakTmaxRule"

    def fit(self, normal_sessions: list[pd.DataFrame]) -> "PeakTMaxRuleDetector":
        peaks = [float(np.nanmax(s["T_max"].to_numpy(dtype=np.float64))) for s in normal_sessions]
        self.normal_peak_mean_ = float(np.mean(peaks)) if peaks else float("nan")
        self.normal_peak_std_ = float(np.std(peaks)) if peaks else float("nan")
        self.frame_threshold_ = _normal_frame_percentile_threshold(self, normal_sessions)
        log.info("PeakTmaxRule ready (%d normal sessions; normal peak mean=%.2f C)",
                 len(normal_sessions), self.normal_peak_mean_)
        return self

    def score_session(self, session: pd.DataFrame) -> float:
        return float(np.nanmax(session["T_max"].to_numpy(dtype=np.float64)))

    def score_frames(self, session: pd.DataFrame) -> NDArray:
        return session["T_max"].to_numpy(dtype=np.float64)


class LSTMVAE:
    """Variational LSTM-AE trained per-frame on (T, C=3) time series.

    Architecture:
      Encoder: LSTM(C → H) → fc_mu, fc_logvar (H → latent_dim)
      Decoder: fc_decode (latent → H) → repeat T times → LSTM(H → C)
    Anomaly score per frame = reconstruction MSE on (T_max, T_min, T_range).

    Training uses whole sessions as 1500-step sequences so each fold preserves
    the original session boundary.
    """
    name = "LSTMVAE"

    def __init__(self, latent_dim: int = 16, hidden_size: int = 64, lr: float = 1e-3,
                 epochs: int = 100, beta: float = 0.5, random_state: int = 20260531):
        import torch
        self._torch = torch
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.beta = beta
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        self._model = self._build()

    def _build(self):
        torch = self._torch
        H, Z, C = self.hidden_size, self.latent_dim, 3

        class _Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = torch.nn.LSTM(input_size=C, hidden_size=H, batch_first=True)
                self.fc_mu = torch.nn.Linear(H, Z)
                self.fc_logvar = torch.nn.Linear(H, Z)
                self.fc_z = torch.nn.Linear(Z, H)
                self.dec = torch.nn.LSTM(input_size=H, hidden_size=H, batch_first=True)
                self.fc_out = torch.nn.Linear(H, C)
            def encode(self, x):
                _, (h, _) = self.enc(x)
                h = h.squeeze(0)
                return self.fc_mu(h), self.fc_logvar(h)
            def reparam(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                return mu + torch.randn_like(std) * std
            def decode(self, z, T):
                h = self.fc_z(z).unsqueeze(1).repeat(1, T, 1)
                y, _ = self.dec(h)
                return self.fc_out(y)
            def forward(self, x, sample: bool = True):
                mu, logvar = self.encode(x)
                z = self.reparam(mu, logvar) if sample else mu
                return self.decode(z, x.shape[1]), mu, logvar

        return _Net().to(self._device)

    def fit(self, normal_sessions: list[pd.DataFrame]) -> "LSTMVAE":
        torch = self._torch
        seqs = [time_series(s) for s in normal_sessions]
        X = torch.from_numpy(np.stack(seqs)).to(self._device)   # (N, T, C)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self._model.train()
        for epoch in range(self.epochs):
            opt.zero_grad()
            recon, mu, logvar = self._model(X, sample=True)
            recon_loss = torch.nn.functional.mse_loss(recon, X, reduction="none").mean(dim=(1, 2)).mean()
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean() / mu.shape[1]
            loss = recon_loss + self.beta * kl
            loss.backward()
            opt.step()
            if epoch == 0 or (epoch + 1) % 25 == 0:
                log.info("LSTM-VAE epoch %d/%d  loss=%.4f (recon=%.4f kl=%.4f)",
                         epoch + 1, self.epochs, loss.item(), recon_loss.item(), kl.item())

        # Training reconstruction stats define the fold-specific score scale.
        with torch.no_grad():
            self._model.eval()
            recon, _, _ = self._model(X, sample=False)
            err = ((recon - X) ** 2).mean(dim=2).cpu().numpy()  # (N, T)
        self._train_err_mean = float(err.mean())
        self._train_err_std = float(err.std() + 1e-9)
        train_scores = (err - self._train_err_mean) / self._train_err_std
        per_session_p99 = np.percentile(train_scores, 99.0, axis=1)
        self.frame_threshold_ = float(np.median(per_session_p99))
        log.info("LSTM-VAE: train recon mean=%.5f std=%.5f frame_p99=%.3f",
                 self._train_err_mean, self._train_err_std, self.frame_threshold_)
        return self

    def score_session(self, session: pd.DataFrame) -> float:
        scores = self.score_frames(session)
        return float(np.percentile(scores, 95))

    def score_frames(self, session: pd.DataFrame) -> NDArray:
        torch = self._torch
        x = torch.from_numpy(time_series(session)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            self._model.eval()
            recon, _, _ = self._model(x, sample=False)
            err = ((recon - x) ** 2).mean(dim=2).squeeze(0).cpu().numpy()  # (T,)
        return (err - self._train_err_mean) / self._train_err_std


def make_detector(kind: str, **kw) -> Detector:
    kind = kind.lower()
    if kind in ("peak", "peaktmax", "peaktmaxrule", "thermal", "thermalrule"):
        return PeakTMaxRuleDetector()
    if kind in ("if", "iforest", "isolationforest"):
        return IsolationForestDetector(**kw)
    if kind in ("ocsvm", "oneclasssvm"):
        return OneClassSVMDetector(**kw)
    if kind in ("vae", "lstmvae", "lstm-vae"):
        return LSTMVAE(**kw)
    raise ValueError(f"Unknown detector kind: {kind!r}")
