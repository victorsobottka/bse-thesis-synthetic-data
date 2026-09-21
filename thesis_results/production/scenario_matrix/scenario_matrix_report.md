# Task 11: purged and embargoed CV, and the scenario matrix — report and stop

Measurement only. No metric definition in the live path, no TEMPORAL_COLS, no composite, no report text was changed. One code change, an option: `FinancialMetrics.compute_discriminative_score(cv="contiguous"|"purged", embargo=None)` and the matching pass-through in `compute_auc_null`, notebook cell 16. The default is the published behaviour, bit for bit (checked below). Everything ran under `phase_b_venv/bin/python`. Scripts: `analysis/task11_2026-09-21/`; artifacts: `thesis_results/production/scenario_matrix/`.

## 0. Constraints applied, and two limits you should read first

- **No test-period null was attempted** (0 positions at the evaluation length inside the test block, 1 with validation). The null is the approved **full-series fixed-length adjacent sliding null**, computed exhaustively at both evaluation lengths (Track A: L = test block, 486–501; walk-forward: L = N//6 = 810–832). **Stated limitation: its population spans 20 years, while the observed comparison is the test period.** Nothing was done to engineer around it. It carries about 5 independent block pairs at Track A length and 3 at walk-forward length; positions overlap almost entirely.
- **The published GAN draws cannot be re-scored.** The series behind the recorded metrics of TimeGAN, QuantGAN and CNN-WGAN-GP were never stored (Part 4 of the overnight report: the `.npz` holds a second `generate()` call), so a purged AUC of the published GAN draws does not exist and cannot be computed without retraining. Two frames follow. **Frame P (published draw)** has S0 and S1 only, exact, from `{market}_metrics.csv`. **Frame F (fresh draws)** has all six scenarios: for each of the 20 Phase B draws per market-seed the metrics and both AUCs come from that same draw, the per-draw composite is computed, and a unit's composite is the mean over the 20 draws. Between F rows only the scenario differs. Frame F is therefore the only place the scenarios are comparable with each other; frame P is the anchor to what was published.
- Ranks are the pipeline's (five generators, `na_option="bottom"`, average ties). The control does not compete in the published ranks; where its composite is reported it is re-ranked as a **sixth competitor** (6-pool), which changes the generators' values slightly, so those are reported separately.

## Part 1. Purged and embargoed cross-validation

### 1.1 The option and what it costs

Folds stay contiguous: the test chunks are scikit-learn's own stratified, unshuffled chunks (one contiguous chunk per class per fold). For each class block, training rows within the embargo of the test chunk are dropped on both sides. **Embargo = `AUC_WINDOW` = 20 rows** (a window covers 20 observations, so any embargo ≥ 19 leaves no observation shared between a training row and a test row; measured minimum index gap 21). Shuffled folds are not offered by the option: they leak through the window overlap.

Cost per fold, 5 folds (rows of the stacked real|synthetic training design, both classes): **40, 80, 80, 80, 40 rows dropped**. Track A (L = 486–501, 466–481 windows per class): 5.2–5.4% of the training rows in the two edge folds, 10.4–10.7% in the three interior folds; smallest training set 665 rows. Walk-forward (L = 810–832): 3.1–6.3%; smallest training set 1184 rows. **The folds are not too small to fit** at either length (the option raises if one ever is, and it did not). It does become infeasible for short comparisons: on the test-block pairs `[0:n]` vs `[n:2n]` used for the numbers you quoted, purged CV cannot be fitted for n < 83.

Verification: (i) default path bit-identical to the pre-change notebook on 18 real/synthetic pairs and in `compute_auc_null`; (ii) `purged(embargo=0)` equals `contiguous` exactly; (iii) `shuffled` is refused; (iv) `verify_notebook.py` exits 0; (v) Phase B rerun for all 15 market-seeds with the changed notebook equals the earlier Phase B (before the option) on all 66,600 numeric cells, max |difference| 0; (vi) the contiguous sliding nulls reproduce Task 10's on all 36,320 positions (max |difference| 1.1e-16) and the contiguous observed AUCs reproduce Phase B's own `discriminative_auc_raw` (1.1e-16, 1,800 series).

### 1.2 NIFTY50 first: is the anti-predictive mass leakage-adjacent or drift?

| construction | draws with contiguous AUC < 0.5 | mean, contiguous | mean, shuffled folds | **mean, purged** | share > 0.5, shuffled | share > 0.5, purged |
|---|---|---|---|---|---|---|
| full-series sliding null, track_a | 828 of 3963 (20.9%) | 0.380 | 0.619 | **0.358** | 98.8% | 1.0% |
| full-series sliding null, walk_forward | 333 of 3305 (10.1%) | 0.355 | 0.609 | **0.326** | 90.1% | 0.3% |
| test-block pairs `[0:n]` vs `[n:2n]` (the 0.300 / 0.665 / 94% numbers), all 48 draws; purged fits 26 of them | 48 | 0.302 | 0.668 | not defined for 22 draws (n < 83) | 95.8% | — |
| same, the 26 draws where purged CV fits | 26 | 0.403 | 0.702 | **0.195** | 100.0% | 0.0% |

**Result: it stays near the contiguous value. The drift is real and the metric is measuring it.** Purged folds leave the anti-predictive draws where they were (0.380 → 0.358 at Track A, 0.355 → 0.326 at walk-forward; on the quoted test-block pairs, 0.403 → 0.195 on the draws where it fits) and only 1.0% / 0.3% of them cross 0.5, against 98.8% / 90.1% under shuffled folds. The share of the whole null below 0.5 goes **up** under purging (NIFTY50 Track A 20.9% → 25.6%). The shuffled-fold values near 0.62–0.67 are what leakage through overlapping windows produces, not what a corrected estimator recovers. I proceeded, as both outcomes were results.

### 1.3 All five markets: the null, at both evaluation lengths

Full-series sliding null (approved construction), exhaustive. Mean, sd, p97.5 and the share of draws below 0.5, under contiguous (published), purged (embargo 20) and shuffled (diagnostic only, leaks) folds, on the same pairs. Independent block pairs ≈ N/2L (5.0 at Track A, 3.0 at walk-forward).

| market | length (L, positions) | scheme | mean | sd | p97.5 | share < 0.5 |
|---|---|---|---|---|---|---|
| BOVESPA | Track A (496, 3962) | contiguous | 0.571 | 0.177 | 0.915 | 32.9% |
| BOVESPA | Track A (496, 3962) | purged | 0.556 | 0.179 | 0.911 | 39.7% |
| BOVESPA | Track A (496, 3962) | shuffled | 0.701 | 0.101 | 0.916 | 0.9% |
| BOVESPA | walk-forward (825, 3304) | contiguous | 0.580 | 0.132 | 0.821 | 26.8% |
| BOVESPA | walk-forward (825, 3304) | purged | 0.570 | 0.136 | 0.814 | 30.3% |
| BOVESPA | walk-forward (825, 3304) | shuffled | 0.679 | 0.069 | 0.840 | 0.0% |
| FTSE | Track A (501, 3996) | contiguous | 0.587 | 0.160 | 0.805 | 23.3% |
| FTSE | Track A (501, 3996) | purged | 0.572 | 0.164 | 0.805 | 28.8% |
| FTSE | Track A (501, 3996) | shuffled | 0.706 | 0.078 | 0.869 | 0.3% |
| FTSE | walk-forward (832, 3334) | contiguous | 0.535 | 0.186 | 0.827 | 48.0% |
| FTSE | walk-forward (832, 3334) | purged | 0.525 | 0.191 | 0.825 | 51.8% |
| FTSE | walk-forward (832, 3334) | shuffled | 0.666 | 0.099 | 0.844 | 0.4% |
| MOEX | Track A (500, 3995) | contiguous | 0.625 | 0.173 | 0.876 | 26.6% |
| MOEX | Track A (500, 3995) | purged | 0.610 | 0.185 | 0.878 | 28.5% |
| MOEX | Track A (500, 3995) | shuffled | 0.733 | 0.082 | 0.907 | 0.0% |
| MOEX | walk-forward (832, 3331) | contiguous | 0.602 | 0.165 | 0.846 | 28.9% |
| MOEX | walk-forward (832, 3331) | purged | 0.589 | 0.172 | 0.847 | 33.5% |
| MOEX | walk-forward (832, 3331) | shuffled | 0.704 | 0.085 | 0.854 | 0.0% |
| NIFTY50 | Track A (496, 3963) | contiguous | 0.638 | 0.170 | 0.906 | 20.9% |
| NIFTY50 | Track A (496, 3963) | purged | 0.621 | 0.179 | 0.904 | 25.6% |
| NIFTY50 | Track A (496, 3963) | shuffled | 0.745 | 0.096 | 0.938 | 0.3% |
| NIFTY50 | walk-forward (825, 3305) | contiguous | 0.663 | 0.136 | 0.852 | 10.1% |
| NIFTY50 | walk-forward (825, 3305) | purged | 0.652 | 0.145 | 0.855 | 13.4% |
| NIFTY50 | walk-forward (825, 3305) | shuffled | 0.738 | 0.070 | 0.855 | 1.0% |
| SHANGHAI | Track A (486, 3889) | contiguous | 0.693 | 0.157 | 0.976 | 8.3% |
| SHANGHAI | Track A (486, 3889) | purged | 0.675 | 0.168 | 0.975 | 14.9% |
| SHANGHAI | Track A (486, 3889) | shuffled | 0.773 | 0.106 | 0.968 | 0.0% |
| SHANGHAI | walk-forward (810, 3241) | contiguous | 0.663 | 0.144 | 0.950 | 12.4% |
| SHANGHAI | walk-forward (810, 3241) | purged | 0.638 | 0.161 | 0.950 | 17.2% |
| SHANGHAI | walk-forward (810, 3241) | shuffled | 0.758 | 0.093 | 0.946 | 0.0% |

The stationary GARCH-t reference (Task 10 recipe: 300 pairs of independent stationary paths from the seed-42 fit; SHANGHAI is the constrained boundary fit, 8 / 5 pairs rejected by the generation guard) under the same schemes:

| market | length | contiguous: mean (sd) | purged: mean (sd) | shuffled: mean (sd) | share < 0.5: contiguous / purged / shuffled |
|---|---|---|---|---|---|
| BOVESPA | Track A | 0.524 (0.135) | 0.503 (0.140) | 0.669 (0.066) | 40.3% / 47.7% / 0.7% |
| BOVESPA | walk-forward | 0.513 (0.114) | 0.503 (0.117) | 0.632 (0.058) | 44.0% / 48.3% / 0.3% |
| FTSE | Track A | 0.553 (0.159) | 0.529 (0.166) | 0.697 (0.077) | 35.3% / 44.3% / 0.7% |
| FTSE | walk-forward | 0.527 (0.137) | 0.517 (0.139) | 0.652 (0.071) | 40.7% / 44.7% / 0.3% |
| MOEX | Track A | 0.569 (0.175) | 0.543 (0.187) | 0.709 (0.088) | 38.3% / 44.3% / 0.0% |
| MOEX | walk-forward | 0.540 (0.146) | 0.528 (0.150) | 0.664 (0.078) | 40.0% / 43.0% / 0.7% |
| NIFTY50 | Track A | 0.573 (0.170) | 0.550 (0.180) | 0.710 (0.087) | 35.0% / 40.7% / 0.3% |
| NIFTY50 | walk-forward | 0.550 (0.140) | 0.537 (0.144) | 0.669 (0.076) | 34.0% / 39.0% / 0.0% |
| SHANGHAI | Track A | 0.704 (0.225) | 0.685 (0.240) | 0.801 (0.131) | 19.7% / 24.3% / 0.7% |
| SHANGHAI | walk-forward | 0.648 (0.199) | 0.634 (0.209) | 0.748 (0.118) | 23.3% / 26.0% / 0.3% |

Reading: on a stationary process the purged null centres on 0.50 (0.50–0.55; SHANGHAI 0.63–0.69 on the boundary fit), contiguous folds sit 0.01–0.03 higher, and shuffled folds sit at 0.63–0.71 (SHANGHAI 0.75–0.80) — the leak. On the real series the purged null is 0.52–0.68: the excess over the stationary reference is regime drift, and it is present at both lengths.

### 1.4 The observed discriminative AUC under each scheme

Track A length. The published GAN series are unavailable, so the observed values are the means over the 20 fresh Phase B draws × 3 seeds (60 series per cell; the contiguous value reproduces Phase B's own `discriminative_auc_raw` to 1.1e-16). The control is the pipeline's fixed `default_rng(42)` permutation, as published.

| model | mean over 5 markets: contiguous | purged | shuffled (diag.) |
|---|---|---|---|
| TimeGAN | 0.796 | 0.776 | 0.870 |
| QuantGAN | 0.639 | 0.615 | 0.742 |
| CNN-WGAN-GP | 0.633 | 0.612 | 0.739 |
| GARCH | 0.651 | 0.631 | 0.749 |
| GJR-GARCH | 0.633 | 0.608 | 0.734 |
| CONTROL: shuffled real | 0.465 | 0.430 | 0.639 |

**Market-model cells (25) above the same-scheme null:** contiguous 0/25 above p97.5 and 0/25 above mean + 1.96 sd; **purged 0/25 and 0/25**; shuffled 1/25 and 1/25. Walk-forward length, published contiguous AUCs (mean over folds and seeds; purged and shuffled cannot be recomputed because the fold series of the walk-forward generators were not stored): 0/25 above p97.5, 0/25 above mean + 1.96 sd. **Your zero-of-25 statement holds under purged CV as well.**

The control's AUC by market (contiguous / purged): BOVESPA 0.32 / 0.31; FTSE 0.70 / 0.68; MOEX 0.38 / 0.38; NIFTY50 0.51 / 0.39; SHANGHAI 0.41 / 0.39. Over 20 further random permutations of each market's test block (sensitivity, not used in the scenarios) the purged AUC is 0.440 on average with sd 0.129, range 0.21–0.70; contiguous 0.486. A discriminator that worked would put a series with no dynamics well above the real-versus-real null. Under both time-blocked schemes the control sits **below** the real-versus-real null in four of the five markets (null means 0.56–0.69) and **above** it in FTSE (0.70 / 0.68 against 0.59 / 0.57): its value is decided by which way the market drifts, not by the missing dynamics.

**Limitation recorded:** the sliding nulls above are populations of adjacent 20-year-series blocks, the observed comparison is the test period, and the block pairs overlap almost entirely (≈5 and ≈3 independent pairs). Residual, not changed: the `StandardScaler` in the metric is fit on all rows before the folds are cut, so scaling statistics from the test chunk enter the training features under every scheme, purged included.

## Part 2. The scenario matrix

| # | null | CV | ranked discriminative columns |
|---|---|---|---|
| S0 | legacy (n_rep = 20) | contiguous | dist + absz (published) |
| S1 | length-matched sliding | contiguous | dist + absz |
| S2 | length-matched sliding | purged | dist + absz |
| S3 | length-matched sliding | purged | dist only |
| S4 | — | purged | neither (both to DESCRIPTIVE_COLS) |
| S5 | stationary GARCH-t reference | purged | dist + absz |

**Validation.** Recomputing fidelity, temporal and composite ranks from the raw metric columns under S0 reproduces the published values for all 75 model-evaluations: max |difference| 8.9e-16 (fidelity), 8.9e-16 (temporal), 8.9e-16 (composite) — last-bit rounding of the mean — and every per-metric rank and every within-unit position identical. The same recomputation on each Phase B draw equals Phase B's own composite_rank (8.9e-16). For S1 the published-draw result reproduces Task 10: CNN-WGAN-GP 2.548 first, GJR-GARCH 2.567 second, margin 0.019.

Composite (overall = mean over the 15 units; lower is better), position in brackets, and the margin between first and second:

| frame | scenario | TimeGAN | QuantGAN | CNN-WGAN-GP | GARCH | GJR-GARCH | first | second | margin |
|---|---|---|---|---|---|---|---|---|---|
| P | S0 published | 3.8643 (5) | 3.0405 (4) | 2.6048 (2) | 2.9667 (3) | 2.5238 (1) | GJR-GARCH | CNN-WGAN-GP | 0.0810 |
| P | S1 (published draw) | 3.8548 (5) | 3.0405 (4) | 2.5476 (1) | 2.9905 (3) | 2.5667 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0190 |
| F | S0 (fresh draws) | 3.6931 (5) | 2.9590 (3) | 2.6589 (1) | 3.0100 (4) | 2.6789 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0200 |
| F | S1 (fresh draws) | 3.7131 (5) | 2.9600 (3) | 2.6432 (1) | 3.0048 (4) | 2.6789 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0357 |
| F | S2 (fresh draws) | 3.7021 (5) | 2.9636 (3) | 2.6339 (1) | 3.0074 (4) | 2.6930 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0590 |
| F | S3 (fresh draws) | 3.7236 (5) | 2.9726 (3) | 2.6412 (1) | 2.9894 (4) | 2.6733 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0321 |
| F | S4 (fresh draws) | 3.7159 (5) | 2.9952 (4) | 2.6526 (1) | 2.9739 (3) | 2.6624 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0098 |
| F | S5 (fresh draws) | 3.7112 (5) | 2.9645 (3) | 2.6282 (1) | 3.0043 (4) | 2.6918 (2) | CNN-WGAN-GP | GJR-GARCH | 0.0636 |

**How many of the 75 model-evaluations change rank relative to the published result.** "Position" = the model's 1–5 position inside its unit; "composite value" = the unit composite differs from the published one. Frame F compares a draw-averaged unit composite with a single published draw, so it carries draw noise as well as the scenario; the last columns separate the two.

| frame | scenario | position changed (of 75) | composite value changed (of 75) | mean abs composite change | units with any position change (of 15) | vs F-S0, same units: positions / values | one fresh draw at a time: position changes vs published, mean (range) | same draw, scenario vs S0: mean (range) |
|---|---|---|---|---|---|---|---|---|
| P | S0 | 0 | 0 | 0.000 | 0 | — | — | — |
| P | S1 | 2 | 32 | 0.061 | 2 | — | — | — |
| F | S0 | 44 | 75 | 0.496 | 14 | 0 / 0 | 53.0 (45–60) | 0.0 (0–0) |
| F | S1 | 44 | 75 | 0.499 | 14 | 3 / 54 | 52.3 (41–61) | 6.0 (1–15) |
| F | S2 | 44 | 75 | 0.497 | 14 | 7 / 69 | 52.6 (40–63) | 10.2 (5–21) |
| F | S3 | 44 | 75 | 0.494 | 14 | 5 / 75 | 52.0 (45–58) | 12.7 (6–20) |
| F | S4 | 46 | 75 | 0.492 | 14 | 18 / 75 | 52.8 (47–58) | 23.2 (13–34) |
| F | S5 | 44 | 75 | 0.499 | 14 | 5 / 70 | 52.7 (42–60) | 10.5 (4–20) |

**The noise floor is the finding here.** Two different fresh draws of the *same* scenario (S0) put 53 of the 75 model-evaluations in a different within-unit position (range 40–61); a fresh draw against the published result: 52–53. Replacing S0 by S1, S2, S3, S5 on the *same* draw moves 6, 10, 13 and 10 positions, and S4 moves 23. Every scenario effect on positions is between a tenth and a half of the draw-to-draw noise floor.

### The shuffled control through every scenario

Control re-ranked as a sixth competitor. "Beats best" = the control's composite is below the best generator's in the same 6-pool.

| frame | scenario | control fidelity rank | control temporal rank (generators' mean) | **control composite** | best generator | its composite | control beats best: units of 15 | draws of 300 |
|---|---|---|---|---|---|---|---|---|
| P | S0 published | 1.038 | 3.41 (3.52) | **2.224** | GJR-GARCH | 3.231 | 13 | — |
| P | S1 (published draw) | 1.038 | 3.62 (3.48) | **2.329** | CNN-WGAN-GP | 3.238 | 10 | — |
| F | S0 (fresh draws) | 1.025 | 3.30 (3.54) | **2.162** | CNN-WGAN-GP | 3.395 | 15 | 91% |
| F | S1 (fresh draws) | 1.025 | 3.51 (3.50) | **2.270** | CNN-WGAN-GP | 3.353 | 15 | 80% |
| F | S2 (fresh draws) | 1.025 | 3.58 (3.48) | **2.305** | CNN-WGAN-GP | 3.334 | 15 | 75% |
| F | S3 (fresh draws) | 1.025 | 3.41 (3.52) | **2.218** | CNN-WGAN-GP | 3.361 | 15 | 73% |
| F | S4 (fresh draws) | 1.025 | 3.49 (3.50) | **2.256** | CNN-WGAN-GP | 3.364 | 15 | 70% |
| F | S5 (fresh draws) | 1.025 | 3.51 (3.50) | **2.265** | CNN-WGAN-GP | 3.338 | 15 | 74% |

**Does demoting the discriminative columns let the control back into contention? No — it was never out of it, in any scenario, published included.** With the control competing, its composite is 2.16–2.33 against 3.23–3.40 for the best generator; it beats the best generator in 13 of 15 units under S0 as published and in 15 of 15 in every fresh-draw scenario. It does so because it scores 1.0 on six of the seven fidelity metrics by construction (fidelity rank 1.03) and the composite gives that family half the weight. Its temporal rank, the part the composite is meant to police it with, is 3.3–3.6 of 6 in every scenario, the same as the generators' average (3.5): the temporal family does not rank the shuffled series below a typical generator in S0 either. S4 moves the control from 2.16 to 2.26 (fresh draws), and the share of draws in which it beats the best generator falls from 91% (S0) to 70% (S4) — no gain for the control. So the test you set does not count against S4; what it shows is that the composite offers no protection against the control in any scenario, and that excluding the control from the ranks by construction (as the pipeline does) is what keeps it from winning, in S0 as in S4.

Control rank on each ranked metric (6-pool, mean over 15 units; 1 = closest to real): acf_returns_mae 3.20, acf_absolute_mae 4.27, acf_squared_mae 3.33, hurst_diff 4.60, resid_kurtosis_diff 2.73, dist 2.53, absz 3.20. On fresh draws the length-matched columns give dist_contiguous 2.60, dist_purged 3.03, absz_length-matched contiguous 4.57, purged 4.62. Only hurst_diff and acf_absolute_mae rank the control near the bottom; the discriminative columns place it mid-pack (dist) or worse (absz, because a two-sided z-score flags an AUC of 0.43 against a null of 0.6).

## Part 3. Is the margin real?

Bootstrap: the 15 market-seed units resampled with replacement, **B = 19,999**, each model's overall composite recomputed (mean over the resampled units), margin = composite(second) − composite(first) for the first and second of the point estimate, 95% percentile interval. "P(reversed)" is the share of resamples in which the order of the two is reversed or tied. Sensitivity, labelled as an addition: resampling the 5 markets (all 3 seeds of each), because seeds of one market share the real data.

| frame | scenario | first | second | margin | 95% interval (15 units) | excludes zero? | P(reversed) | P(first stays first) | 95% interval (5 markets) | excludes zero? |
|---|---|---|---|---|---|---|---|---|---|---|
| P | S0 published | GJR-GARCH | CNN-WGAN-GP | 0.0810 | [-0.388, +0.548] | no | 37% | 62% | [-0.067, +0.248] | no |
| P | S1 (published draw) | CNN-WGAN-GP | GJR-GARCH | 0.0190 | [-0.424, +0.467] | no | 48% | 52% | [-0.052, +0.086] | no |
| F | S0 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0200 | [-0.103, +0.148] | no | 38% | 62% | [-0.125, +0.147] | no |
| F | S1 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0357 | [-0.087, +0.162] | no | 30% | 70% | [-0.112, +0.164] | no |
| F | S2 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0590 | [-0.067, +0.190] | no | 19% | 81% | [-0.106, +0.194] | no |
| F | S3 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0321 | [-0.091, +0.160] | no | 31% | 69% | [-0.099, +0.162] | no |
| F | S4 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0098 | [-0.107, +0.130] | no | 44% | 56% | [-0.124, +0.134] | no |
| F | S5 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0636 | [-0.068, +0.196] | no | 17% | 83% | [-0.105, +0.201] | no |

**No scenario's 95% interval on the first-versus-second margin excludes zero** — not the published S0, not any length-matched, purged or stationary-reference variant, in either frame, and not under the market-level resampling. First and second are not separable by this benchmark. That is the answer, not a failure to find a winner. Two further readings of the same fact: across the 20 fresh draws, scored one draw at a time, the overall winner is CNN-WGAN-GP in 11, GJR-GARCH in 8, GARCH in 1 of 20 under S0, and the split is the same in every scenario (CNN-WGAN-GP 11–12, GJR-GARCH 6–8); and in the published draw the GJR-GARCH lead comes from one seed (margins by seed 42/43/44: 0.400;-0.071;-0.086; under S1: -0.229;0.200;0.086).

The two frames bracket the noise: frame P intervals (about ±0.45) contain the single-draw noise of the published run, frame F intervals (about ±0.13) come from draw-averaged composites and contain almost none of it. Neither excludes zero.

**Seed-to-seed spread.** Standard deviation of a model's unit composite over the 3 seeds inside a market (ddof = 1), averaged over the 5 markets (largest single market in brackets):

| frame | scenario | TimeGAN | QuantGAN | CNN-WGAN-GP | GARCH | GJR-GARCH | second − first margin, sd over seeds (values by seed) |
|---|---|---|---|---|---|---|---|
| P | S0 | 0.698 (1.19) | 0.651 (0.79) | 0.682 (1.01) | 0.805 (1.10) | 0.560 (0.75) | 0.276 (0.400;-0.071;-0.086) |
| P | S1 | 0.738 (1.38) | 0.702 (0.79) | 0.680 (0.92) | 0.841 (1.10) | 0.557 (0.83) | 0.222 (-0.229;0.200;0.086) |
| F | S0 | 0.455 (0.75) | 0.187 (0.28) | 0.193 (0.29) | 0.099 (0.21) | 0.123 (0.26) | 0.036 (0.049;0.030;-0.020) |
| F | S1 | 0.479 (0.85) | 0.188 (0.28) | 0.204 (0.33) | 0.104 (0.20) | 0.132 (0.26) | 0.038 (0.059;0.057;-0.008) |
| F | S2 | 0.479 (0.86) | 0.176 (0.28) | 0.200 (0.34) | 0.113 (0.21) | 0.128 (0.27) | 0.031 (0.091;0.056;0.030) |
| F | S3 | 0.459 (0.83) | 0.182 (0.27) | 0.206 (0.35) | 0.119 (0.22) | 0.119 (0.24) | 0.035 (0.061;0.043;-0.007) |
| F | S4 | 0.465 (0.79) | 0.184 (0.26) | 0.213 (0.38) | 0.132 (0.22) | 0.111 (0.20) | 0.058 (0.052;0.034;-0.056) |
| F | S5 | 0.475 (0.87) | 0.179 (0.29) | 0.204 (0.34) | 0.108 (0.21) | 0.130 (0.27) | 0.020 (0.082;0.067;0.042) |

Published draw: 0.56–0.84 per model (the composite of a model in one market moves by more than half a rank place between seeds), which is larger than the 0.08 published margin. Fresh draws remove the draw component and leave the training-run component: 0.10–0.48. In frame F the sign of the margin is the same in all three seeds under S2 and S5 (CNN-WGAN-GP ahead by 0.03–0.09), and differs by seed under S0, S1, S3 and S4.

## Part 4. Summary

| scenario | first | second | margin | 95% bootstrap interval on the margin (15 units) | excludes zero? | control's composite (6-pool) |
|---|---|---|---|---|---|---|
| S0 published | GJR-GARCH | CNN-WGAN-GP | 0.0810 | [-0.388, +0.548] | no | 2.224 (best generator 3.231) |
| S1 (published draw) | CNN-WGAN-GP | GJR-GARCH | 0.0190 | [-0.424, +0.467] | no | 2.329 (best generator 3.238) |
| S0 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0200 | [-0.103, +0.148] | no | 2.162 (best generator 3.395) |
| S1 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0357 | [-0.087, +0.162] | no | 2.270 (best generator 3.353) |
| S2 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0590 | [-0.067, +0.190] | no | 2.305 (best generator 3.334) |
| S3 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0321 | [-0.091, +0.160] | no | 2.218 (best generator 3.361) |
| S4 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0098 | [-0.107, +0.130] | no | 2.256 (best generator 3.364) |
| S5 (fresh draws) | CNN-WGAN-GP | GJR-GARCH | 0.0636 | [-0.068, +0.196] | no | 2.265 (best generator 3.338) |

Rows marked "published draw" use the published series (S0, S1 only). Rows marked "fresh draws" use 20 Phase B draws per market-seed and are the only ones in which S2–S5 exist.

### Recommendation (the decision is Victor's)

**Recommend S4: move `discriminative_auc_dist` and `discriminative_auc_absz` to DESCRIPTIVE_COLS, with purged CV as the way any AUC that is still reported is computed.** Reasoning, in the order the evidence came:

1. **The choice among scenarios does not decide who is first.** No interval excludes zero in any of the eight rows; the overall winner flips with the draw in about 40% of fresh draws in every scenario. Whatever is chosen, the honest statement is that GJR-GARCH and CNN-WGAN-GP are not separable here. The scenario effects on positions (6–23 of 75 on the same draw) are small next to the draw-to-draw floor (53 of 75).
2. **Purged CV is the right estimator but does not repair the metric.** On a stationary reference it recovers a null centred on 0.50 (contiguous is 0.01–0.03 high, shuffled 0.63–0.71 from leakage), so it removes the boundary leak. On the real series the anti-predictive draws stay (NIFTY50 0.380 → 0.358) because they are regime drift, and the purged null is broad (sd 0.14–0.19, p97.5 0.81–0.98). Against it 0 of 25 market-model cells rise above p97.5 or mean + 1.96 sd, under purged exactly as under contiguous, at Track A length; the same 0 of 25 at walk-forward length (contiguous only).
3. **Neither ranked column has a working reference under any scheme I tested.** `dist` measures distance from 0.5 where the real-versus-real level is 0.52–0.68; the shuffled control scores 0.43 purged (0.31–0.68 across markets; below the null in four markets, above it in FTSE) and is rated mid-pack. `absz` (S1, S2, S5) is a z-score against a null of 3 to 5 independent blocks that mixes 20 years of regimes, and its choice is the thing that moved first and second in Task 10. Retaining them (S1, S2, S5) changes the ranking by no more than the noise; retaining `dist` alone (S3) keeps the flaw you identified.
4. **S4 does not help the control.** With the control competing, its composite is 2.16 (S0) and 2.26 (S4) and its temporal rank is 3.3 and 3.5 against the generators' 3.5 either way. The composite does not police the control in any scenario, S0 included; the exclusion of the control from the ranks is doing that work, and S4 does not change it. Should you prefer a scenario that keeps an AUC column, **S2 is the cleanest retention** (purged, length-matched); S5, with the stationary reference null, gives the same ranking within the noise.
5. **Costs of S4, stated plainly.** The temporal family shrinks to five metrics (three ACF errors, Hurst, GARCH-residual kurtosis) against seven fidelity metrics, each temporal column now carrying 1/10 of the composite instead of 1/14; it moves 23 of 75 within-unit positions on the same draw against S0, the most of any scenario; its first-second margin is the smallest (0.010); and the benchmark loses its only supervised real-versus-synthetic test. That test is not doing measurable work as built: it does not place any generator outside real-versus-real, and it does not detect the control.

Not done, by instruction or by construction: no test-period null; no change to `FIDELITY_COLS`, `TEMPORAL_COLS`, the composite, `{market}_metrics.csv`, `generate_report.py`, `report_data.py` or any report text; no retraining; Phase A untouched. New tracked-file change: `3_4_integrated_pipeline.ipynb` (the purged option; default unchanged). New untracked outputs are under `analysis/task11_2026-09-21/`, `thesis_results/production/scenario_matrix/`, and `thesis_results/production/*/seed*/phase_b_COMMIT_DRIFT/` for the twelve market-seeds that did not yet have one. Stopping here.
