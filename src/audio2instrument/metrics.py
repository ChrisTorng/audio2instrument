from __future__ import annotations

import numpy as np
from scipy.signal import stft


def compare_audio(reference: np.ndarray, estimate: np.ndarray, sample_rate: int) -> dict[str, float]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    count = min(len(reference), len(estimate))
    if count == 0:
        raise ValueError("audio arrays must not be empty")
    ref = np.asarray(reference[:count], dtype=np.float64)
    est = np.asarray(estimate[:count], dtype=np.float64)
    ref_energy = float(np.sum(ref * ref) + 1e-15)
    error_energy = float(np.sum((ref - est) ** 2) + 1e-15)
    correlation = float(np.corrcoef(ref, est)[0, 1]) if np.std(ref) and np.std(est) else 0.0
    _, _, ref_stft = stft(ref, fs=sample_rate, nperseg=2048, noverlap=1536)
    _, _, est_stft = stft(est, fs=sample_rate, nperseg=2048, noverlap=1536)
    ref_mag = np.abs(ref_stft)
    est_mag = np.abs(est_stft)
    spectral_convergence = float(np.linalg.norm(ref_mag - est_mag) / (np.linalg.norm(ref_mag) + 1e-15))
    return {
        "correlation": correlation,
        "snr_db": float(10.0 * np.log10(ref_energy / error_energy)),
        "spectral_convergence": spectral_convergence,
        "reference_rms": float(np.sqrt(np.mean(ref * ref))),
        "estimate_rms": float(np.sqrt(np.mean(est * est))),
        "peak_error": float(np.max(np.abs(ref - est))),
    }
