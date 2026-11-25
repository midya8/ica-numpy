"""Cross-check this FastICA against sklearn's on the same mixtures."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from sklearn.decomposition import FastICA as SkFastICA

from ica_numpy.data import sources, mix
from ica_numpy.eval import align, snr_db

S = sources(4000)
X, _ = mix(S, seed=1)

sk = SkFastICA(n_components=4, whiten='unit-variance', random_state=2, max_iter=1000)
S_sk = sk.fit_transform(X.T).T
S_al, _, corr_sk = align(S, S_sk)
print("sklearn FastICA corr:", np.round(corr_sk, 4),
      "snr:", np.round(snr_db(S, S_al), 1))
