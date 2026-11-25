"""FastICA with a tanh contrast: centering, whitening, deflation."""

import numpy as np


def whiten(X):
    """Center, then whiten so the observed signals have identity
    covariance. Returns (Xw, mean, T) with Xw = (X - mean) @ T.

    The eigenvectors are sign-fixed (largest component positive) and
    sorted by eigenvalue descending, so the whitened basis does not
    depend on how the linear algebra library happens to order them.
    """
    mean = X.mean(axis=1, keepdims=True)
    Xc = X - mean
    C = np.cov(Xc)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    for j in range(evecs.shape[1]):
        if evecs[np.argmax(np.abs(evecs[:, j])), j] < 0:
            evecs[:, j] *= -1
    T = evecs @ np.diag(1.0 / np.sqrt(evals + 1e-12)) @ evecs.T
    return T @ Xc, mean, T


def _contrast(w, xw):
    """tanh-contrast value and its gradient pieces for direction w."""
    u = w @ xw
    g = np.tanh(u)
    gp = (1.0 - g ** 2).mean()
    # non-gaussianity proxy used to pick between restarts: for a
    # gaussian source E[tanh(u)] is zero, so its square grows with how
    # far the found direction is from gaussian
    score = float(g.mean() ** 2)
    return u, g, gp, score


def _one_unit(xw, w, W_prev, tol=1e-7, max_iter=300):
    """One fixed-point search for a single component, kept orthogonal
    to the rows already found."""
    w = w / np.linalg.norm(w)
    for _ in range(max_iter):
        u, g, gp, _ = _contrast(w, xw)
        w_new = xw @ g / xw.shape[1] - gp * w
        if W_prev is not None and len(W_prev):
            w_new -= W_prev.T @ (W_prev @ w_new)
        norm = np.linalg.norm(w_new)
        if norm < 1e-12:
            return None, False
        w_new /= norm
        if np.abs(np.abs(w_new @ w) - 1.0) < tol:
            return w_new, True
        w = w_new
    return w, False


def fastica(X, n_components=None, seed=2, max_iter=300, tol=1e-7, restarts=5):
    """Returns (S_hat, W, info). Rows of W unmix the whitened
    observations; S_hat = W @ whiten(X). Each component is searched
    from several random starts and the one with the largest
    non-gaussianity proxy is kept."""
    n_obs = X.shape[0]
    n_components = n_components or n_obs
    Xw, mean, T = whiten(X)
    rng = np.random.default_rng(seed)
    W = np.zeros((n_components, Xw.shape[0]))
    converged = []
    for c in range(n_components):
        best_w, best_score, any_converged = None, -1.0, False
        for _ in range(restarts):
            w0 = rng.normal(size=Xw.shape[0])
            w, ok = _one_unit(Xw, w0, W[:c], tol=tol, max_iter=max_iter)
            if w is None:
                continue
            _, _, _, score = _contrast(w, Xw)
            if score > best_score:
                best_w, best_score, any_converged = w, score, ok
        W[c] = best_w
        converged.append(any_converged)
    S_hat = W @ Xw
    return S_hat, W, {"whitened_cov_max": float(np.abs(np.cov(Xw) - np.eye(Xw.shape[0])).max()),
                      "orthogonality_gap": float(np.abs(W @ W.T - np.eye(n_components)).max()),
                      "converged": converged}
