"""Synthetic sources and mixtures for the cocktail-party setup."""

import numpy as np


def sources(n, seed=0):
    """Four independent sources, unit-ish scale, length n each."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    saw = 2.0 * ((t * 0.02) % 1.0) - 1.0
    sine = np.sin(2 * np.pi * t * 5.0 / n)
    chatter = rng.normal(0.0, 1.0, size=n)
    pulses = np.where((t % 97) < 5, 2.5, 0.0) + 0.3 * np.sin(2 * np.pi * t * 40.0 / n)
    S = np.vstack([saw, sine, chatter, pulses])
    # give each source a different amplitude so scaling has to be undone
    return S * np.array([[1.0], [2.0], [1.5], [0.8]])


def mix(S, seed=1, noise=0.0):
    rng = np.random.default_rng(seed)
    k = S.shape[0]
    A = rng.uniform(0.5, 1.5, size=(k, k))
    X = A @ S
    if noise:
        X = X + rng.normal(0.0, noise, size=X.shape)
    return X, A
