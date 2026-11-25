# ica-numpy

FastICA written from scratch in NumPy. Four synthetic sources (sawtooth,
sine, gaussian chatter, pulse train) are mixed by a random 4x4 matrix;
the unmixing recovers them from the mixtures alone, up to the usual
permutation, sign and scale freedom.

Measured on the 4000-sample run:

| metric | FastICA | PCA (same dimension, inverted) |
|---|---|---|
| mean absolute correlation with the true sources | 0.9997 | 0.53 |
| per-source SNR after alignment | 27.4 / 45.5 / 28.0 / 13.3 dB | n/a |

The pulse train scores lower on SNR because most of its energy sits in
5-sample bursts, where a small timing misalignment dominates the error.
sklearn's FastICA on the same data lands at 0.9998 mean correlation,
within a hair of this implementation.

## What is implemented

- `whiten`: centering plus a symmetric whitening transform. Eigenvectors
  are sign-fixed and sorted by eigenvalue so the basis does not depend
  on how the linear algebra library orders near-tied eigenvectors;
  without that, repeated runs could land on different fixed points.
- `fastica`: the tanh contrast with the classic fixed-point update,
  deflated one component at a time with Gram-Schmidt against the rows
  already found. Each component is searched from five random starts and
  the start with the largest non-gaussianity proxy is kept.
- `eval`: alignment by absolute correlation, a least-squares gain that
  carries the sign, and per-source SNR.

## Run

```
pip install numpy matplotlib
python train_driver.py   # writes results.json and assets/fig_separation.png
python -m pytest tests/ -q
```

`tests/test_ica.py` covers whitening to identity covariance, orthonormal
unmixing rows, beating PCA by a wide margin, run-to-run stability, and
alignment on a known swap-and-flip case.
