# Regenerated 2026-09-21 (report update)

Regenerated locally, fixed scaler (scaler fitted within each training fold):
- `<market>/auc_null.json` x5 through the notebook's own `precompute_auc_null` (the `auc_scaler` cache key forced it). Now also holds the approved fixed-length adjacent sliding null at both
  evaluation lengths (fixed-split and walk-forward), contiguous and purged, every position kept, plus the half-split legacy and 999-draw nulls. Previous files: `../backups/pre_S4_adoption_20260921_113011/auc_null/`.
- `stationary_reference_null.json`: stationary GARCH-t reference (300 pairs per length, both lengths; contiguous, purged, shuffled diagnostic).
- `nifty50_sub_0_5.json`: NIFTY50 draws below 0.5 under contiguous / purged / shuffled folds.
- 30 figures `<market>/seed<N>/<market>_metrics_heatmap.png` and `_rank_comparison.png` from the S4 metrics CSVs (the heatmap shows raw metric values, so its content is unchanged by S4; the rank comparison shows avg_rank over the twelve ranked metrics).

Not regenerated: walk-forward `discriminative_auc` (375 fold models never saved); published AUC columns of QuantGAN and CNN-WGAN-GP (CUDA RNG state never saved).
`earlier_experiment_provenance.json` records the figures quoted from the earlier experiment (0.506 +/- 0.084, 0.584); it is a citation, not a result.
