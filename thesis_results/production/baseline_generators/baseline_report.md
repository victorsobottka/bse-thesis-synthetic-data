# Baselines without oracle access: does the benchmark tell no dynamics from good dynamics?

Comparison entries only. Nothing here enters the published composite, `FIDELITY_COLS`, `TEMPORAL_COLS`, `{market}_metrics.csv`, or any report text. New files only, under `thesis_results/production/baseline_generators/` and `analysis/baselines_2026-09-21/`; run under `phase_b_venv/bin/python`. Stopping at the end: the framing decision is Victor's.

## Part 1. The baselines

- **iid** (historical simulation): for each market, seed and draw, a series of test-block length sampled with replacement from that market's training block.
- **sb_r** (the specified stationary block bootstrap, Politis & Romano 1994): geometric block lengths, circular wrap inside the training block, mean block length from **Politis–White** (`arch.bootstrap.optimal_block_length`, arch 8.0.0) applied to the training returns, floored at 1.
- **sb_abs** (added, *not* the specified baseline): same resampler, mean block length from Politis–White on |training returns|. The specified choice returns 0.3–5.4, i.e. almost no dependence, so the "modest short-range dependence" case is nearly iid; the volatility-persistence scale is 110–128, and the temporal metrics measure exactly that, so I ran it as a labelled sensitivity.

20 draws per market-seed (15 units, 300 series per baseline), draw *k* of seed *s* from `default_rng([s, k, code])`.

| market | n_train | target length | Politis–White b: returns / \|returns\| / returns² | mean block used: sb_r / sb_abs | empirical mean block, seed 42: sb_r / sb_abs |
|---|---|---|---|---|---|
| BOVESPA | 3962 | 496 | 5.43 / 111.4 / 94.3 | 5.43 / 111.4 | 5.36 / 105.5 |
| FTSE | 3997 | 501 | 0.27 / 122.9 / 110.6 | 1.00 / 122.9 | 1.00 / 113.0 |
| MOEX | 3995 | 500 | 2.19 / 125.7 / 76.7 | 2.19 / 125.7 | 2.18 / 118.6 |
| NIFTY50 | 3963 | 496 | 2.57 / 122.1 / 106.3 | 2.57 / 122.1 | 2.54 / 117.7 |
| SHANGHAI | 3888 | 486 | 1.69 / 127.9 / 124.6 | 1.69 / 127.9 | 1.68 / 109.9 |

Why these lengths: FTSE's Politis–White length on returns is 0.27, which is not a possible mean block length, so it is floored at 1 (iid). For the other markets the length is used as returned. The empirical mean block length of the generated series matches the target for sb_r and is 6–14% below it for the long blocks, which is the truncation of the last block in a series of about 500 observations.

**No observation after the end of the training block is read.** (i) `gen_baselines.py` opens `data/processed_files/train/<market>_train.parquet` and nothing else with data: from the test file it takes only the row count, from the parquet footer metadata, to set the target length; the validation file is not opened. (ii) The resamplers are `iid_bootstrap(train, n, rng)` and `stationary_bootstrap(train, n, mean_block, rng)`: the training array, a length and a generator are all they receive. (iii) Every generated index is below n_train (largest index used per market: BOVESPA 3961 of 3962, FTSE 3996 of 3997, MOEX 3994 of 3995, NIFTY50 3962 of 3963, SHANGHAI 3887 of 3888). (iv) All 900 stored draws were checked: values not found in the training block = 0; regenerating with the recorded seeds reproduces every draw exactly. Provenance: `baseline_provenance.json` (training-block SHA-256 per market).

## Part 2. Scoring

Every entry is scored per draw against the test block through the notebook's own `evaluate_multiple_models` → `FinancialMetrics` (same call as Phase B: `max_lag = min(50, n//4)`, the cached legacy AUC null; BLAS threading as in Phase B, because arch's GARCH residual-kurtosis fit differs at 1e-5–1e-3 under single-thread BLAS). **Validation:** the five generators' and the control's rows produced by this call equal Phase B's stored rows on all 34200 numeric cells, max |difference| 0. The two baselines (and sb_abs) go through the identical path.

Ranking: the five generators (their 20 fresh Phase B draws, draw *k* paired with baseline draw *k*), the shuffled control (the pipeline's fixed `default_rng(42)` permutation, the same in every draw) and the baselines compete in **one pool** per unit and draw, `na_option="bottom"`, average ties; composite = ½(fidelity + temporal). **S0** = the published columns (five temporal metrics + `discriminative_auc_dist` + `discriminative_auc_absz`); **S4** = both discriminative columns out. A unit's value is the mean over its 20 draws; "overall" is the mean over the 15 units; position = order of the overall composite. **Primary frame A: eight entries** (five generators, control, iid, sb_r). Frames B (sb_abs replaces sb_r), C (all nine) and P (the *published* generator draws and control against each baseline draw) are sensitivities below. Generators in frames A–C are fresh Phase B draws, not the published recorded draws (which cannot be re-scored; the recorded GAN series were never stored).

### 2.1 One ranking (frame A, eight entries)

| entry | fidelity rank | temporal rank S0 / S4 | composite S0 | composite S4 | position S0 / S4 |
|---|---|---|---|---|---|
| shuffled control | 1.032 | 4.121 / 4.447 | 2.577 | 2.739 | 1 / 1 |
| CNN-WGAN-GP | 4.337 | 3.930 / 3.930 | 4.133 | 4.133 | 2 / 2 |
| GJR-GARCH | 4.552 | 3.895 / 3.883 | 4.224 | 4.218 | 3 / 3 |
| QuantGAN | 4.928 | 4.280 / 4.411 | 4.604 | 4.669 | 4 / 5 |
| GARCH | 5.025 | 4.269 / 4.215 | 4.647 | 4.620 | 5 / 4 |
| stationary block bootstrap (Politis–White on returns) | 5.207 | 4.573 / 4.420 | 4.890 | 4.813 | 6 / 6 |
| iid bootstrap (historical simulation) | 5.280 | 5.137 / 4.885 | 5.209 | 5.082 | 7 / 7 |
| TimeGAN | 5.640 | 5.794 / 5.809 | 5.717 | 5.725 | 8 / 8 |

Sensitivities (composite S0 / S4, position S0 / S4):

| entry | frame B: sb_abs instead of sb_r | frame C: all nine entries | frame P: published generators + control, baseline draw *k* |
|---|---|---|---|
| shuffled control | 2.603 / 2.756; 1 / 1 of 8 | 2.827 / 3.016; 1 / 1 of 9 | 2.638 / 2.815; 1 / 1 of 8 |
| CNN-WGAN-GP | 4.190 / 4.186; 2 / 2 of 8 | 4.571 / 4.578; 2 / 2 of 9 | 4.044 / 4.049; 3 / 3 of 8 |
| GJR-GARCH | 4.257 / 4.235; 3 / 3 of 8 | 4.668 / 4.661; 3 / 3 of 9 | 3.982 / 3.935; 2 / 2 of 8 |
| QuantGAN | 4.636 / 4.695; 4 / 6 of 8 | 5.100 / 5.179; 4 / 6 of 9 | 4.619 / 4.707; 5 / 5 of 8 |
| GARCH | 4.679 / 4.636; 6 / 5 of 8 | 5.137 / 5.104; 6 / 4 of 9 | 4.571 / 4.490; 4 / 4 of 8 |
| TimeGAN | 5.751 / 5.763; 8 / 8 of 8 | 6.367 / 6.376; 9 / 9 of 9 | 5.948 / 5.925; 8 / 8 of 8 |
| iid bootstrap (historical simulation) | 5.227 / 5.095; 7 / 7 of 8 | 5.783 / 5.638; 8 / 8 of 9 | 5.269 / 5.189; 7 / 7 of 8 |
| stationary block bootstrap (Politis–White on returns) | — | 5.428 / 5.336; 7 / 7 of 9 | 4.929 / 4.890; 6 / 6 of 8 |
| stationary block bootstrap, long blocks (Politis–White on |returns|) | 4.657 / 4.634; 5 / 4 of 8 | 5.120 / 5.112; 5 / 5 of 9 | — |

Generators' mean over the five: fidelity 4.90, temporal 4.43 (S0) / 4.45 (S4).

### 2.2 Where the control's advantage comes from

Composite gap = ½ × fidelity gap + ½ × temporal gap (frame A):

| comparison | composite gap S0 | of which fidelity | of which temporal | composite gap S4 | of which fidelity | of which temporal |
|---|---|---|---|---|---|---|
| iid bootstrap (historical simulation) minus control | 2.632 | 2.124 (81%) | 0.508 (19%) | 2.343 | 2.124 (91%) | 0.219 (9%) |
| stationary block bootstrap (Politis–White on returns) minus control | 2.313 | 2.087 (90%) | 0.226 (10%) | 2.074 | 2.087 (101%) | -0.013 (-1%) |

### 2.3 Which temporal metrics fail to penalise missing dynamics

Mean rank among the eight (frame A, 300 unit-draws; 1 = closest to real). The last column is the mean over the five generators. Raw values are the mean over 15 units × 20 draws.

| metric | control | iid | sb_r | generators (mean) | best generator (rank) | control raw | iid raw | sb_r raw | generators raw (min–max) |
|---|---|---|---|---|---|---|---|---|---|
| acf_returns_mae | 4.05 | 2.82 | 3.19 | 5.19 | CNN-WGAN-GP (4.59) | 0.0517 | 0.0486 | 0.0495 | 0.0533–0.0813 |
| acf_absolute_mae | 5.12 | 4.92 | 4.71 | 4.25 | CNN-WGAN-GP (3.22) | 0.0730 | 0.0716 | 0.0706 | 0.0640–0.1177 |
| acf_squared_mae | 4.09 | 3.54 | 3.70 | 4.93 | CNN-WGAN-GP (4.29) | 0.0570 | 0.0544 | 0.0551 | 0.0633–0.0999 |
| hurst_diff | 5.90 | 6.95 | 5.16 | 3.60 | GJR-GARCH (2.90) | 0.0759 | 0.1079 | 0.0673 | 0.0315–0.0798 |
| resid_kurtosis_diff | 3.08 | 6.19 | 5.35 | 4.28 | GJR-GARCH (3.47) | 0.9361 | 10.7727 | 8.3694 | 2.8301–3.7924 |
| dist | 3.01 | 6.12 | 5.26 | 4.32 | GJR-GARCH (3.67) | 0.1179 | 0.2793 | 0.2476 | 0.1655–0.3028 |
| absz | 3.61 | 5.42 | 4.66 | 4.46 | QuantGAN (4.07) | 1.1361 | 1.7007 | 1.4976 | 1.3855–1.7045 |

The fidelity metrics for the same entries (mean rank; control / iid / sb_r / generators): mean_diff 1.00 / 4.72 / 4.68 / 5.12; std_diff 1.00 / 6.08 / 6.00 / 4.58; wasserstein 1.00 / 5.27 / 5.21 / 4.90; quantile_mse 1.00 / 5.76 / 5.65 / 4.72; tail_index_diff 1.00 / 5.08 / 4.90 / 5.00; extreme_events_diff 1.22 / 5.32 / 5.21 / 4.85; energy_distance 1.00 / 4.73 / 4.79 / 5.10.

**Does the theory hold?** *acf_returns_mae*: the control is at the noise floor (0.052 — real and shuffled ACFs of about 500 returns are both near zero, so their MAE is sampling noise), yes, but every entry is there (generators 0.053–0.055, iid 0.049; TimeGAN 0.081): the metric has no resolution, and it ranks iid *better* than the control. *hurst_diff*: holds strongly — control 5.90, iid 6.95 (worst), against 2.9–3.7 for GARCH, GJR-GARCH, CNN-WGAN-GP and QuantGAN. *acf_absolute_mae*: holds only weakly — control 0.0730, iid 0.0716, sb_r 0.0706 against GJR-GARCH 0.0707, GARCH 0.0744, QuantGAN 0.0791, TimeGAN 0.1177 and CNN-WGAN-GP 0.0640: only CNN-WGAN-GP is clearly better than no dynamics. *acf_squared_mae*: **does not hold** — the control (0.0570) and iid (0.0544) score *better* than all five generators (0.063–0.100). *resid_kurtosis_diff*: penalises iid and sb_r (6.19, 5.35), **not** the control (3.08, better than every one of the five generators: 3.47–5.62), because the control has the test block's exact marginal. So of the five temporal columns only `hurst_diff` (and, for iid, `resid_kurtosis_diff`) reliably penalise missing dynamics; `acf_returns_mae` and `acf_squared_mae` reward it or cannot tell. `dist` and `absz` (S0 only) carry marginal information: the control ranks 3.0 / 3.6 on them, iid 6.1 / 5.4.

**The PROJECT_STATE note (control 0.0730 beats QuantGAN 0.0981 and TimeGAN 0.1232 on acf_absolute_mae).** Those three numbers are exactly the means over the 15 published units (control 0.0730, QuantGAN 0.0981, TimeGAN 0.1232; verified from `{market}_metrics.csv`). They are mean effects of a heavy right tail, not typical units: medians are QuantGAN 0.0630 and TimeGAN 0.0798 against the control's 0.0760, and the worst units are 0.250 (QuantGAN) and 0.405 (TimeGAN). Published, QuantGAN beats the control in 9 of 15 units and TimeGAN in 7.

Does that mean those two generators produce worse volatility clustering than none at all? Partly, and for different reasons. On fresh draws the ACF of |r| over lags 1–50 (real: lag 1 0.159, lag 10 0.081, lag 50 −0.022, mean 0.051) looks like this:

| entry | mean ACF(\|r\|), lags 1–50 | signed bias vs real | MAE vs real (= acf_absolute_mae) | share of unit-draws with MAE above the control's | lag 1 / 10 / 50 (draw 0, mean over units) |
|---|---|---|---|---|---|
| shuffled control | -0.003 | -0.054 | 0.0730 | 0% | -0.010 / +0.011 / +0.014 |
| iid bootstrap (historical simulation) | -0.003 | -0.053 | 0.0716 | 44% | -0.014 / -0.007 / -0.013 |
| stationary block bootstrap (Politis–White on returns) | +0.000 | -0.048 | 0.0706 | 37% | +0.082 / -0.008 / -0.009 |
| stationary block bootstrap, long blocks (Politis–White on |returns|) | +0.073 | +0.028 | 0.0736 | 36% | +0.118 / +0.126 / +0.041 |
| TimeGAN | +0.065 | +0.038 | 0.1177 | 52% | +0.091 / +0.076 / +0.035 |
| QuantGAN | +0.143 | +0.035 | 0.0791 | 42% | +0.204 / +0.215 / +0.069 |
| CNN-WGAN-GP | +0.072 | +0.012 | 0.0640 | 29% | +0.102 / +0.097 / +0.028 |
| GARCH | +0.029 | +0.037 | 0.0744 | 45% | +0.079 / +0.080 / -0.007 |
| GJR-GARCH | +0.027 | +0.023 | 0.0707 | 39% | +0.085 / +0.095 / -0.014 |

The no-dynamics entries have a flat ACF near zero, so their error is simply the mean magnitude of the real ACF (about 0.07). **QuantGAN** clusters volatility too strongly and never lets it decay (mean ACF 0.143 against 0.051, lag 50 0.069 against −0.022): its error is a bias, and it is worse than nothing on the unit-draws where that bias is large — but in 58% of draws it beats the control, and its median published unit does. **TimeGAN**'s mean profile is roughly right (0.065) but erratic from draw to draw, which gives it the largest error (0.118) and a worse-than-control result in 52% of draws. **GARCH** and **GJR-GARCH** are too weak at lag 1 (0.079 / 0.085 against 0.159) and end up level with the control (0.074, 0.071 against 0.073). The larger point is the resolution of the metric: the gap between doing nothing (0.073) and the best generator (CNN-WGAN-GP 0.064) is 0.009, half of CNN-WGAN-GP's own draw-to-draw sd within a unit (0.019) and under a third of GARCH's (0.031; QuantGAN 0.035, TimeGAN 0.043).

## Part 3. Separation from noise

Bootstrap: the 15 units resampled with replacement, **B = 19,999**, each entry's overall composite recomputed; margin = composite(best generator) − composite(entry), so a **negative margin means the generator is ahead** of the baseline. The best generator is the one with the lowest overall composite in that frame (CNN-WGAN-GP in frames A–C, GJR-GARCH in frame P). "Units + draws" also resamples the 20 draws inside each resampled unit (paired across entries), which carries the baselines' own generation noise.

| frame | scenario | entry | best generator | margin (best gen − entry) | 95% interval, 15 units | excludes zero? | 95% interval, units + draws | excludes zero? |
|---|---|---|---|---|---|---|---|---|
| A | S0 | shuffled control | CNN-WGAN-GP | +1.557 | [+1.287, +1.844] | **yes** | [+1.274, +1.857] | **yes** |
| A | S0 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -1.075 | [-1.457, -0.703] | **yes** | [-1.480, -0.688] | **yes** |
| A | S0 | stationary block bootstrap (Politis–White on returns) | CNN-WGAN-GP | -0.757 | [-1.053, -0.470] | **yes** | [-1.082, -0.447] | **yes** |
| A | S4 | shuffled control | CNN-WGAN-GP | +1.394 | [+0.975, +1.804] | **yes** | [+0.967, +1.811] | **yes** |
| A | S4 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -0.949 | [-1.316, -0.586] | **yes** | [-1.332, -0.565] | **yes** |
| A | S4 | stationary block bootstrap (Politis–White on returns) | CNN-WGAN-GP | -0.680 | [-0.961, -0.385] | **yes** | [-0.993, -0.359] | **yes** |
| B | S0 | shuffled control | CNN-WGAN-GP | +1.587 | [+1.325, +1.870] | **yes** | [+1.311, +1.886] | **yes** |
| B | S0 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -1.037 | [-1.409, -0.675] | **yes** | [-1.435, -0.655] | **yes** |
| B | S0 | stationary block bootstrap, long blocks (Politis–White on |returns|) | CNN-WGAN-GP | -0.467 | [-0.659, -0.269] | **yes** | [-0.728, -0.208] | **yes** |
| B | S4 | shuffled control | CNN-WGAN-GP | +1.430 | [+1.033, +1.823] | **yes** | [+1.015, +1.840] | **yes** |
| B | S4 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -0.910 | [-1.263, -0.544] | **yes** | [-1.286, -0.530] | **yes** |
| B | S4 | stationary block bootstrap, long blocks (Politis–White on |returns|) | CNN-WGAN-GP | -0.449 | [-0.635, -0.258] | **yes** | [-0.708, -0.192] | **yes** |
| C | S0 | shuffled control | CNN-WGAN-GP | +1.744 | [+1.445, +2.056] | **yes** | [+1.430, +2.078] | **yes** |
| C | S0 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -1.212 | [-1.651, -0.790] | **yes** | [-1.669, -0.769] | **yes** |
| C | S0 | stationary block bootstrap (Politis–White on returns) | CNN-WGAN-GP | -0.857 | [-1.194, -0.537] | **yes** | [-1.225, -0.501] | **yes** |
| C | S0 | stationary block bootstrap, long blocks (Politis–White on |returns|) | CNN-WGAN-GP | -0.549 | [-0.771, -0.326] | **yes** | [-0.845, -0.250] | **yes** |
| C | S4 | shuffled control | CNN-WGAN-GP | +1.563 | [+1.101, +2.014] | **yes** | [+1.097, +2.026] | **yes** |
| C | S4 | iid bootstrap (historical simulation) | CNN-WGAN-GP | -1.059 | [-1.477, -0.641] | **yes** | [-1.495, -0.619] | **yes** |
| C | S4 | stationary block bootstrap (Politis–White on returns) | CNN-WGAN-GP | -0.758 | [-1.079, -0.418] | **yes** | [-1.111, -0.393] | **yes** |
| C | S4 | stationary block bootstrap, long blocks (Politis–White on |returns|) | CNN-WGAN-GP | -0.534 | [-0.750, -0.314] | **yes** | [-0.828, -0.240] | **yes** |
| P | S0 | shuffled control | GJR-GARCH | +1.344 | [+0.860, +1.878] | **yes** | [+0.858, +1.878] | **yes** |
| P | S0 | iid bootstrap (historical simulation) | GJR-GARCH | -1.287 | [-1.861, -0.641] | **yes** | [-1.878, -0.645] | **yes** |
| P | S0 | stationary block bootstrap (Politis–White on returns) | GJR-GARCH | -0.947 | [-1.494, -0.342] | **yes** | [-1.512, -0.332] | **yes** |
| P | S4 | shuffled control | GJR-GARCH | +1.119 | [+0.503, +1.746] | **yes** | [+0.498, +1.751] | **yes** |
| P | S4 | iid bootstrap (historical simulation) | GJR-GARCH | -1.254 | [-1.830, -0.636] | **yes** | [-1.831, -0.628] | **yes** |
| P | S4 | stationary block bootstrap (Politis–White on returns) | GJR-GARCH | -0.955 | [-1.507, -0.359] | **yes** | [-1.515, -0.354] | **yes** |

**Draw-to-draw spread** (frame A and the sb_abs frame; the overall composite of the entry in each of the 20 draws, then its spread across draws; the control is one fixed permutation, so its spread comes only from the generators it is ranked against):

| frame | scenario | entry | overall composite: mean (sd over draws) | range over draws | mean within-unit sd | position over the 20 draws: min – mode – max | draws ahead of every generator |
|---|---|---|---|---|---|---|---|
| A | S0 | shuffled control | 2.577 (0.055) | 2.469–2.683 | 0.211 | 1 – 1 – 1 | 20 of 20 |
| A | S0 | iid bootstrap (historical simulation) | 5.209 (0.163) | 4.864–5.410 | 0.641 | 4 – 7 – 7 | 0 of 20 |
| A | S0 | stationary block bootstrap (Politis–White on returns) | 4.890 (0.240) | 4.383–5.245 | 0.721 | 3 – 6 – 7 | 0 of 20 |
| A | S4 | shuffled control | 2.739 (0.081) | 2.577–2.870 | 0.265 | 1 – 1 – 1 | 20 of 20 |
| A | S4 | iid bootstrap (historical simulation) | 5.082 (0.150) | 4.806–5.313 | 0.635 | 5 – 6 – 7 | 0 of 20 |
| A | S4 | stationary block bootstrap (Politis–White on returns) | 4.813 (0.242) | 4.393–5.298 | 0.736 | 3 – 5 – 7 | 0 of 20 |
| B | S0 | shuffled control | 2.603 (0.062) | 2.514–2.767 | 0.219 | 1 – 1 – 1 | 20 of 20 |
| B | S0 | iid bootstrap (historical simulation) | 5.227 (0.186) | 4.888–5.493 | 0.632 | 5 – 7 – 7 | 0 of 20 |
| B | S0 | stationary block bootstrap, long blocks (Politis–White on |returns|) | 4.657 (0.340) | 4.167–5.457 | 1.097 | 3 – 4 – 7 | 0 of 20 |
| B | S4 | shuffled control | 2.756 (0.087) | 2.592–2.941 | 0.270 | 1 – 1 – 1 | 20 of 20 |
| B | S4 | iid bootstrap (historical simulation) | 5.095 (0.168) | 4.845–5.373 | 0.629 | 5 – 7 – 7 | 0 of 20 |
| B | S4 | stationary block bootstrap, long blocks (Politis–White on |returns|) | 4.634 (0.372) | 4.107–5.405 | 1.132 | 3 – 4 – 7 | 0 of 20 |
| P | S0 | shuffled control | 2.638 (0.023) | 2.583–2.686 | 0.096 | 1 – 1 – 1 | 20 of 20 |
| P | S0 | iid bootstrap (historical simulation) | 5.269 (0.126) | 5.024–5.498 | 0.532 | 6 – 7 – 7 | 0 of 20 |
| P | S0 | stationary block bootstrap (Politis–White on returns) | 4.929 (0.175) | 4.555–5.188 | 0.642 | 4 – 6 – 7 | 0 of 20 |
| P | S4 | shuffled control | 2.815 (0.029) | 2.735–2.869 | 0.130 | 1 – 1 – 1 | 20 of 20 |
| P | S4 | iid bootstrap (historical simulation) | 5.189 (0.128) | 4.991–5.418 | 0.532 | 6 – 7 – 7 | 0 of 20 |
| P | S4 | stationary block bootstrap (Politis–White on returns) | 4.890 (0.171) | 4.570–5.165 | 0.689 | 5 – 6 – 7 | 0 of 20 |

## Part 4. Summary

| entry | fidelity rank | temporal rank (S0 / S4) | composite S0 | composite S4 | position among eight (S0 / S4) |
|---|---|---|---|---|---|
| shuffled control | 1.032 | 4.121 / 4.447 | 2.577 | 2.739 | 1 / 1 |
| CNN-WGAN-GP | 4.337 | 3.930 / 3.930 | 4.133 | 4.133 | 2 / 2 |
| GJR-GARCH | 4.552 | 3.895 / 3.883 | 4.224 | 4.218 | 3 / 3 |
| QuantGAN | 4.928 | 4.280 / 4.411 | 4.604 | 4.669 | 4 / 5 |
| GARCH | 5.025 | 4.269 / 4.215 | 4.647 | 4.620 | 5 / 4 |
| stationary block bootstrap (Politis–White on returns) | 5.207 | 4.573 / 4.420 | 4.890 | 4.813 | 6 / 6 |
| iid bootstrap (historical simulation) | 5.280 | 5.137 / 4.885 | 5.209 | 5.082 | 7 / 7 |
| TimeGAN | 5.640 | 5.794 / 5.809 | 5.717 | 5.725 | 8 / 8 |

### Which explanation the evidence supports

**Both, unequally: the oracle marginal is the dominant cause of the control's composite win, strongly supported; the weak temporal penalty is real but partial and metric-specific.**

1. **Oracle marginal — strong.** The control wins (composite 2.58 S0 / 2.74 S4 against 4.13 / 4.13 for the best generator), and it wins on fidelity: fidelity rank 1.03. The two baselines that have no oracle access and no dynamics-fitting do **not** win: iid is seventh of eight in both scenarios (composite 5.21 / 5.08), sb_r sixth (4.89 / 4.81), and neither is ahead of every generator in a single one of the 20 draws. The generators are ahead of iid by 1.08 composite points (95% interval excludes zero; the same holds against the published generators, in S4, and with draws resampled). The fidelity family accounts for 81% of the composite gap between the control and iid under S0 and 91% under S4 (table 2.2). So the control's win is what an exact test-block marginal buys, which is a legitimate reason to keep it out of the competition.
2. **Weak temporal penalty — partial.** The temporal family alone ranks the control 4.12 (S0) / 4.45 (S4) of eight against the generators' mean 4.43 / 4.45: at their mean or slightly better, as before. That is not all oracle: with no oracle, iid ranks 5.14 / 4.88 (worse than the generators' mean by 0.7 / 0.4 places, better than TimeGAN), and sb_r 4.57 / 4.42 — level with the generators' mean. The long-block sb_abs, a pure resample of training history, is at 4.20 / 4.16 against 4.33 / 4.24 for GARCH and 4.33 / 4.44 for QuantGAN; it is fifth of eight under S0 and fourth under S4, level with GARCH and QuantGAN, and behind CNN-WGAN-GP by 0.47 composite points (interval excludes zero). By metric (2.3): `hurst_diff` penalises missing dynamics strongly; `acf_absolute_mae` only weakly (the gap to no dynamics is half the best generator's own draw-to-draw sd); `acf_returns_mae`, `acf_squared_mae` and, for the control, `resid_kurtosis_diff` do not penalise it or reward it.
3. **What the evidence does not support.** The strongest form of the concern — that the benchmark cannot tell no dynamics from good dynamics — is not supported at the composite level for a fair no-dynamics baseline: iid and sb_r rank sixth and seventh, and the separation from the best generator is outside the bootstrap noise. What it does support is narrower: the temporal family carries little information about dynamics per se (control and sb_r at the generators' mean; three of five columns unable to separate), so the composite's ordering among the *generators* leans on the fidelity family and on `hurst_diff`; and GARCH and QuantGAN are not distinguishable from a block bootstrap of the training data with blocks of about 120 observations (frame B, S0: composite 4.68 and 4.64 against 4.66 for sb_abs; margin sb_abs − GARCH −0.02 [−0.17, +0.12], sb_abs − QuantGAN +0.02 [−0.15, +0.18]; S4 likewise, all four intervals contain zero; `extra_checks.json`).

Caveats: generators in the pool are fresh Phase B draws (frame P repeats the comparison with the published generator rows and reaches the same reading, with GJR-GARCH as the best generator); fidelity ranks of the baselines are penalised by the shift between training-block and test-block marginals, which the oracle control does not suffer and the trained generators partly share; the baselines are compared on the same 15 units, whose seeds of one market share the same real data.

Stopping here. The framing decision is Victor's.
