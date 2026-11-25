"""Scoring separated sources against the truth, up to the usual
permutation, sign and scale freedom of ICA."""

import numpy as np


def align(S_true, S_hat):
    """Match each estimated row to a distinct true source by absolute
    correlation, then rescale it by the least-squares gain (which
    carries the sign), and return (S_aligned, perm, corr)."""
    k = S_true.shape[0]
    St = S_true / S_true.std(axis=1, keepdims=True)
    Sh = S_hat / S_hat.std(axis=1, keepdims=True)
    C = np.abs(np.corrcoef(St, Sh)[:k, k:])
    perm, used = [], set()
    for i in range(k):
        order = np.argsort(-C[i])
        j = next(j for j in order if j not in used)
        perm.append(j)
        used.add(j)
    perm = np.array(perm)
    # one gain per row; it is negative when the estimated copy is flipped
    scales = np.array([np.dot(S_true[i], S_hat[perm[i]]) /
                       np.dot(S_hat[perm[i]], S_hat[perm[i]]) for i in range(k)])
    S_aligned = np.array([scales[i] * S_hat[perm[i]] for i in range(k)])
    corr = np.array([abs(np.corrcoef(St[i], S_aligned[i])[0, 1]) for i in range(k)])
    return S_aligned, perm, corr


def snr_db(S_true, S_aligned):
    """Per-source reconstruction SNR after alignment."""
    out = []
    for i in range(S_true.shape[0]):
        err = S_true[i] - S_aligned[i]
        out.append(10 * np.log10(np.sum(S_true[i] ** 2) / np.sum(err ** 2)))
    return np.array(out)
