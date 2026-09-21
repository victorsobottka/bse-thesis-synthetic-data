# TS2Vec (vendored, unmodified)

Source: https://github.com/yuezhihan/ts2vec  commit b0088e14a99706c05451316dc6db8d3da9351163
Licence: MIT (see LICENSE). Yue et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI 2022.

Copied verbatim: ts2vec.py, utils.py, models/. Nothing here is edited. The training loop with epoch
patience lives in embedding_distance.py (the upstream `fit` has no early stopping), and reuses this
package's `TSEncoder`, `hierarchical_contrastive_loss` and `take_per_row`.
