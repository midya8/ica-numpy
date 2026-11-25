"""Cocktail-party run: four sources, one random mixing matrix, FastICA
recovers them. Writes results.json and a figure."""

import json
import sys
sys.path.insert(0, 'src')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ica_numpy.data import sources, mix
from ica_numpy.ica import fastica
from ica_numpy.eval import align, snr_db

N = 4000
S = sources(N)
X, A = mix(S, seed=1)

S_hat, W, info = fastica(X)
S_aligned, perm, corr = align(S, S_hat)
snr = snr_db(S, S_aligned)

# what PCA does with the same observations: project onto its top
# components, invert the transform, rescore
from numpy.linalg import eigh
Xc = X - X.mean(axis=1, keepdims=True)
evals, evecs = np.linalg.eigh(np.cov(Xc))
order = np.argsort(evals)[::-1]
P = evecs[:, order[:4]]
X_pca = (P @ (P.T @ Xc)) + X.mean(axis=1, keepdims=True)
_, _, corr_pca = align(S, X_pca)

results = {
    "n": N, "sources": 4,
    "ica_corr": [round(float(c), 4) for c in corr],
    "ica_snr_db": [round(float(s), 2) for s in snr],
    "pca_corr": [round(float(c), 4) for c in corr_pca],
    "mean_abs_corr": round(float(corr.mean()), 4),
    "pca_mean_abs_corr": round(float(corr_pca.mean()), 4),
    "whitened_cov_max": info["whitened_cov_max"],
    "orthogonality_gap": info["orthogonality_gap"],
    "converged": info["converged"],
}
print(json.dumps(results, indent=2))
json.dump(results, open("results.json", "w"), indent=2)

fig, axes = plt.subplots(3, 4, figsize=(15, 7), sharex=True)
t = np.arange(N)
for i in range(4):
    axes[0, i].plot(t, S[i], lw=0.6)
    axes[1, i].plot(t, X[i], lw=0.6)
    axes[2, i].plot(t, S_aligned[i], lw=0.6)
axes[0, 0].set_ylabel("true source")
axes[1, 0].set_ylabel("observed mix")
axes[2, 0].set_ylabel("recovered")
fig.suptitle("FastICA from scratch: four sources from four mixtures")
fig.tight_layout()
fig.savefig("assets/fig_separation.png", dpi=120)
PY
python3 train_driver.py
