# PROJECT_STATE.md — local copy

**Provenance of this file.** The canonical PROJECT_STATE lives in the claude.ai
web project. This local copy was reconstructed on 2026-09-10 from
`.claude/CLAUDE.md` and from the production run artifacts, because the web
version could not be read from this session. Where the two disagree, the web
version was written first and this one was measured from disk — check both.

Run described: 5 models × 5 markets × 3 seeds × 5 folds, 9,000 generator
updates. Executed at commit `7352a02` (`run_config.json`,
`pipeline_run_metadata.json`), results committed as `33cc769`.
`SMOKE_TEST = False`.

---

## §4 Design decisions and the evidence behind them

| Decision | Evidence |
|---|---|
| Budgets in gradient steps, never epochs | Epoch-based budgets produced a 492× asymmetry in generator updates in one run; the affected model looked like an architectural failure until the budget was measured. Walk-forward folds differ in length ~5× (fold 0 trains on 810–837 points, fold 4 on 4,050–4,165). |
| Parity on generator updates, gradient family only | TimeGAN `ae_steps`/`sup_steps` are four-phase pre-training (Yoon et al. 2019), excluded. QuantGAN/FinGAN `n_critic=5` is intrinsic to WGAN-GP (Gulrajani et al. 2017). GARCH/GJR exempt: MLE has no gradient-step analogue. Equal updates ≠ equal compute: at 1,000 steps TimeGAN 20.5 s, FinGAN 42.4 s, QuantGAN 178 s. |
| z-score + tanh(z/3), not min-max | Min-max on MOEX to [−1,1]: 98% of data in 14.3% of the range, median +0.233 (measured, `report_data._build_minmax_evidence`). Switching moved QuantGAN output sd from 3.7× to 1.2× real at identical budget (prior measurement, no artifact in this run). |
| GARCH on raw returns ×100 | tanh squash compresses the variance dynamics GARCH models; `arch` converges poorly near 1e-2. |
| `kurtosis_diff`, `skewness_diff` descriptive-only | Hill α 2.54–3.21 ⇒ kurtosis infinite in 5/5 markets, skewness in 4/5. Sample kurtosis grows with block length in 5/5 markets; BOVESPA 2.00 (n=250) → 10.09 (n=2,000), disjoint blocks. |
| Fidelity/temporal split, equal-weighted composite | Unweighted mean won by shuffled control, 1.24 vs 2.47. 9 of 19 computed metrics permutation-invariant (control/best ratio < 3.4e-5; next metric 0.35). |
| Control excluded from rank | Its fidelity_rank is 1.000 by construction. |
| Both ARCH-LM variants descriptive-only | pvalue_diff 99% on simulated GARCH, saturates on real data (both p = 0.0). stat_diff 76%, null sd 46.4. |
| Conditional VaR, intended pooled | Unconditional VaR: Gaussian noise = real (0.0276). QLIKE inverts at n≈126, stable at n=623 pooled. **As run, files are per-market (n 486–501) — see Discrepancies.** |
| Walk-forward, unshuffled CV | Shuffling moves the AUC null 0.506 → 0.584 (20-day windows overlap by 19). |
| MOEX replaces MSCI World | MSCI World is not BRICS. −33.3% on 2022-02-24 + 27-day suspension kept as a gap, never interpolated. |
| Determinism not enforced | TimeGAN/control bit-identical over 5 runs; FinGAN Wasserstein CV 1.6%; QuantGAN CV 37%, tail_index_diff CV 79%. No single-run winner reported on tail_index_diff, hurst_diff, mean_diff. |
| Two GARCH variants only | GARCH(1,1)-t is the universal baseline; GJR adds leverage (Cont fact 5), which no GAN models explicitly. |
| Per-market training | Pooling adds ~4 cross-market transitions in ~3,900 windows but risks cross-market contamination. |

**Reversed:** TimeGAN collapse was attributed to tanh saturation in Recovery.
This run's `[DIAG]` shows latent saturation ≤ 3.1% (threshold 50%) and
reconstruction sd matching target to 3 decimals — explanation withdrawn,
mechanism open.

---

## §5 Reference values from the production run

All computed from `thesis_results/production/` and
`reports/production/run_log_20260908_222254.log`.

**Ranks (overall_performance.csv)**

| Model | composite | fidelity | temporal | params | mean fit s |
|---|---|---|---|---|---|
| GJR-GARCH | 2.543 | 2.705 | 2.381 | 6 | 0.017 |
| FinGAN | 2.714 | 2.486 | 2.943 | 55,401 | 172.9 |
| QuantGAN | 2.912 | 3.062 | 2.762 | 276,481 | 464.7 |
| GARCH | 2.950 | 3.138 | 2.762 | 5 | 0.018 |
| TimeGAN | 3.881 | 3.610 | 4.152 | 11,400 | 77.1 |

**Per-market-seed wins (of 15)** — composite: GARCH 5, QuantGAN 5, GJR 3,
FinGAN 2 (econometric 8). Temporal: GJR 5, GARCH 4, QuantGAN 4, FinGAN 2
(econometric 9). Fidelity: FinGAN 6, GJR 4, QuantGAN 3, GARCH 2 (gradient 9).
Among the three GANs only, by composite_rank within the five-model ranking:
FinGAN 8, QuantGAN 6, TimeGAN 1.

**Control audit (means over 15 market-seeds)**

| Metric | Control | FinGAN | GARCH | GJR | QuantGAN | TimeGAN |
|---|---|---|---|---|---|---|
| acf_returns_mae | 0.0517 | 0.0512 | 0.0514 | 0.0514 | 0.0535 | 0.0685 |
| acf_squared_mae | 0.0570 | 0.0576 | 0.0541 | 0.0544 | 0.0732 | 0.1002 |
| acf_absolute_mae | 0.0730 | 0.0636 | 0.0593 | 0.0610 | 0.0839 | 0.1232 |
| hurst_diff | 0.0759 | 0.0381 | 0.0404 | 0.0341 | 0.0410 | 0.0881 |

**Walk-forward AUC** — FinGAN 0.640±0.146, GJR 0.658±0.160, GARCH 0.699±0.168,
QuantGAN 0.737±0.193, TimeGAN 0.814±0.144. By fold 0→4: 0.818, 0.694, 0.743,
0.635, 0.658 (declining, **not monotonic**).

**tail_index_diff** — QuantGAN 1.098, GARCH 1.238, FinGAN 1.242, GJR 1.316,
TimeGAN 2.682 (sd 3.33, range 0.024–9.825).

**Guard** — 16 `[WARNING]` firings, all TimeGAN; mean |ACF(1)| 0.568, range
−0.622 to 0.942. TimeGAN worst on 7/7 temporal, 5/7 fidelity, 15/19 metrics.

**Known defects (reported, not fixed — CLAUDE.md §5)**
- Downstream GARCH fit degenerates in 4/90 cells: GARCH MOEX seed 42 (QLIKE
  33,081), control FTSE seeds 42/43/44 (16,975).
- Walk-forward GARCH SHANGHAI fold 3: Wasserstein 74.8 / 43.0 / 7.8 across
  seeds (median across all folds 0.0025), AUC 1.000.
- `pooled_downstream_utility.csv` is per-market (n 486–501), not pooled.

---

## Discrepancies with numbers previously stated

| Claim | Measured |
|---|---|
| TimeGAN guard ACF(1) = 0.429 | 16 firings, mean \|ACF(1)\| 0.568; 0.429 came from an older log (`run_log_20260907_221734.log`) |
| AUC falls monotonically with fold length | Declines overall; folds 2 and 4 rise |
| TimeGAN worst on every metric | 15/19; QuantGAN worse on std_diff, skewness_diff, quantile_mse |
| "Only GARCH clears" control on both volatility-clustering ACFs | True for acf_squared_mae; FinGAN also clears acf_absolute_mae |
| FinGAN won 11/15 at 9,000 updates (three-GAN run) | No three-GAN 9,000-update artifact on disk; this run gives 8/15 |
| Kurtosis 2.53 (n=250) → 11.84 (n=2,000), BOVESPA | 2.00 → 10.09 by disjoint-block mean; method difference likely |
| Downstream utility pooled at n≈2,480 | Per-market, n 486–501 |
| Every model distinguishable from the AUC null | 3/5 beyond 1.96 null sd (TimeGAN z 3.66, QuantGAN 2.75, GARCH 2.30); FinGAN (1.60) and GJR-GARCH (1.80) within |
| TimeGAN's non-worst metrics are all QuantGAN's | std_diff, skewness_diff, quantile_mse → QuantGAN; arch_pvalue_diff → GARCH |
| resid_kurtosis_diff favours the GARCH generators | GARCH 3.12, GJR 3.04 vs GANs 2.68–4.02 and control 0.94 — neither GARCH variant beats the control |
| Latent saturation "≤3.1%" | Confirmed: max 3.1% over 90 `[DIAG]` readings; reconstruction sd within 0.0014 of target over 90 |
