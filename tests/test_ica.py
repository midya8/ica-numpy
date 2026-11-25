import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ica_numpy.data import sources, mix
from ica_numpy.ica import fastica, whiten
from ica_numpy.eval import align, snr_db


def test_whitening_makes_identity_covariance():
    rng = np.random.default_rng(0)
    X = np.array([[2.0, 0, 0.4], [0, 1.0, 0.3], [0.4, 0.3, 0.8]]) @ rng.normal(size=(3, 2000))
    Xw, mean, T = whiten(X)
    assert np.abs(np.cov(Xw) - np.eye(3)).max() < 1e-9
    # whiten is idempotent: the same mean and transform recover Xw again
    Xw2, _, _ = whiten(Xw)
    assert np.abs(np.cov(Xw2) - np.eye(3)).max() < 1e-9


def test_unmixing_rows_are_orthonormal():
    S = sources(2000)
    X, _ = mix(S, seed=1)
    S_hat, W, info = fastica(X)
    assert info["orthogonality_gap"] < 1e-8
    assert info["whitened_cov_max"] < 1e-6
    assert all(info["converged"])


def test_separation_beats_pca_on_mixtures():
    S = sources(4000)
    X, _ = mix(S, seed=1)
    S_hat, W, _ = fastica(X)
    _, _, corr = align(S, S_hat)
    # PCA projection with the same dimension, inverted, rescored
    Xc = X - X.mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(Xc))
    P = evecs[:, np.argsort(evals)[::-1][:4]]
    _, _, corr_pca = align(S, P @ (P.T @ Xc))
    assert corr.mean() > 0.99
    assert corr.mean() > corr_pca.mean() + 0.3


def test_recovery_is_stable_across_whitening_runs():
    S = sources(3000)
    X, _ = mix(S, seed=3)
    corr_runs = []
    for _ in range(2):
        S_hat, _, _ = fastica(X)
        _, _, corr = align(S, S_hat)
        corr_runs.append(corr.mean())
    assert abs(corr_runs[0] - corr_runs[1]) < 1e-9
    assert corr_runs[0] > 0.99


def test_alignment_resolves_sign_and_scale():
    rng = np.random.default_rng(5)
    S_true = rng.normal(size=(2, 5000))
    S_hat = np.array([-3.0 * S_true[1], 0.5 * S_true[0]])  # swapped and flipped
    S_al, perm, corr = align(S_true, S_hat)
    assert corr.min() > 0.9999
    assert list(perm) == [1, 0]
    snr = snr_db(S_true, S_al)
    assert snr.min() > 30  # exact copy up to gain, so SNR is huge
