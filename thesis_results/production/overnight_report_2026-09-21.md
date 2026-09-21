# Overnight bundle, 2026-09-21

Measurement only. Nothing here writes to FIDELITY_COLS, TEMPORAL_COLS, the composite, `{market}_metrics.csv`,
`generate_report.py`, `report_data.py`, or any `.tex` file. No generator was retrained and Phase A was not re-run.
Every script runs under `phase_b_venv/bin/python` (pandas 2.3.3; torch 2.13.0+cu130, arch 8.0.0, statsmodels 0.15.0,
scipy 1.18.1, scikit-learn 1.9.0, numpy 2.5.2 against 2.5.3 recorded). Scripts: `analysis/overnight_2026-09-21/`.
Outputs: `thesis_results/production/overnight_2026-09-21/`. Sections below are appended as each part finishes; the consolidated answers to the three questions are the last section ("Answers").

## Part 1: is TS2Vec permutation-sensitive?

Encoders loaded from `thesis_results/production/<market>/ts2vec/encoder.pt` (no retraining). Real test block versus the same block with its returns permuted, 20 independent permutations, same windowing (128, stride 1), z-scoring and max-pooling as the main run. MMD² is the unbiased statistic; p-values are against the existing exhaustive adjacent-sliding null.

| market | RBF MMD² (mean ± sd) | p vs null (mean) | draws p≤.05 | deg-3 MMD² (mean ± sd) | p (deg-3) | null median (RBF) | Task 5 v3 control (RBF) |
|---|---|---|---|---|---|---|---|
| BOVESPA | 0.1396 ± 0.0172 | 0.982 | 0% | 0.1550 ± 0.0265 | 0.992 | 0.2035 | 0.1388 |
| FTSE | 0.1702 ± 0.0180 | 0.709 | 0% | 0.2285 ± 0.0266 | 0.994 | 0.1896 | 0.1701 |
| MOEX | 0.1504 ± 0.0142 | 0.958 | 0% | 0.1609 ± 0.0181 | 1.000 | 0.2137 | 0.1472 |
| NIFTY50 | 0.1651 ± 0.0176 | 0.916 | 0% | 0.2054 ± 0.0275 | 0.981 | 0.2138 | 0.1676 |
| SHANGHAI | 0.1836 ± 0.0193 | 0.872 | 0% | 0.2437 ± 0.0275 | 0.972 | 0.2374 | 0.1823 |

**Linear probe** (logistic regression, real vs shuffled windows; AUC 0.5 = not separable). Primary protocol: fitted on the validation block, scored on the test block, so the encoder has seen neither. Reference: the same protocol on three volatility-clustering features (lag-1 and lag-5 autocorrelation of |r|, lag-1 of r²), which shows what separability is achievable in principle.

| market | probe AUC on TS2Vec embeddings (valid→test) | ACF-feature reference (valid→test) | diagnostic: train→test | diagnostic: in-period, train block |
|---|---|---|---|---|
| BOVESPA | **0.522** ± 0.153 | 0.501 | 0.431 | 1.000 |
| FTSE | **0.611** ± 0.138 | 0.642 | 0.737 | 1.000 |
| MOEX | **0.612** ± 0.135 | 0.760 | 0.751 | 1.000 |
| NIFTY50 | **0.605** ± 0.151 | 0.711 | 0.749 | 1.000 |
| SHANGHAI | **0.816** ± 0.103 | 0.720 | 0.759 | 1.000 |

The train-block diagnostics are not used for inference: the encoder was trained on the real train windows, so a probe on them cannot be read as out-of-sample sensitivity to ordering (in-period AUC is 1.000 in every market, a value I cannot attribute to ordering rather than to the encoder having seen those windows). The valid and test blocks were never used for encoder training or early stopping (patience is on the epoch-mean training loss).


**Reading (Part 1).** Two questions, two different answers.

1. *Does the MMD see a shuffle? No.* RBF MMD² between the real test block and its permutations averages 0.140–0.184 per market, which is below the null median in all five markets (0.19–0.24), with p = 0.71–0.98 and 0 of 100 permutation draws at p ≤ .05 (RBF and degree 3 alike; degree-3 p = 0.97–1.00). The values reproduce Task 5 v3's control to within 0.003 in every market, so this is a replication of that result, not independent evidence. That the permuted series sit *below* the null median is not itself puzzling: the null compares two adjacent blocks from *different periods*, the permutation keeps the period — the scale question Part 2 takes up.
2. *Is ordering information absent from the embedding? Not in general, but what is there is weak.* A logistic-regression probe fitted on validation-block windows separates real from permuted test-block windows at AUC 0.52 (BOVESPA), 0.61 (FTSE), 0.61 (MOEX), 0.61 (NIFTY50), 0.82 (SHANGHAI); mean 0.63. Three volatility-clustering features give 0.50 / 0.64 / 0.76 / 0.71 / 0.72 (mean 0.67). The embeddings therefore carry about as much ordering information as three ACF features and no more (better on SHANGHAI, worse on MOEX and NIFTY50). The share of the 20 permutations with probe AUC above 0.5 is 60% / 70% / 80% / 90% / 100%: for BOVESPA the probe is not distinguishable from chance, and the ACF reference is not either (0.501), so nothing separable was shown there even in principle.

**Plain statement.** The representation is not permutation-invariant — the probe beats chance in four of five markets — so the failure to detect the shuffled control in Task 5 v3 is a property of the statistic (unsupervised RBF/degree-3 MMD with an effective sample of about 4 independent observations (Task 5 v3), calibrated against a null that mixes periods), not a demonstration that the encoder throws ordering away. Equally it is not evidence that the encoder *uses* ordering well: AUC ≈ 0.6 on ~370 heavily overlapping windows per class, with no significance test that respects the overlap, supports "some", not "much".

Caveats: the probe's training and test windows are one realisation each (overlap 127/128), so the effective sample is far below 369; the ± in the table is permutation randomness only (0.10–0.15, individual permutations range from 0.15 to 0.96); a probe fitted on the validation period and scored on the test period also absorbs any train-to-test shift, which pushes AUC towards or below 0.5; the ACF reference is one hand-picked feature set, not a ceiling.

## Part 2: null-population mismatch (2a: test-period and recency-restricted nulls)

The mismatch: the exhaustive null has median σ 7.3–7.8 (median-heuristic bandwidth of the pooled pair of 20-year-sliding blocks), the real-versus-generator comparisons on the test block 2.7–4.8 (median 3.0–3.3). MMD² at different kernel bandwidths is not on one scale, so a p-value from that null answers a question about a different population.

**Feasibility of the requested null.** A null from test-period blocks only, at the same evaluation length, needs two adjacent blocks of L observations (L = 486–501 here) inside the test period. The test block holds L observations, so the number of positions is **0**. Extending to validation+test (2L observations) gives **1** position (the valid|test pair itself). Neither can form a null. What follows is the sensitivity the request asked for in that case: the same exhaustive adjacent-sliding null restricted to positions whose two blocks both lie in the last M observations (M = 2L, 3L, 4L, 6L, all ≈ 10L). "Independent pairs" = M/2L, the number of non-overlapping block pairs that fit; the positions themselves overlap almost completely, so positions ≫ independent pairs and the p-value floor 1/(positions+1) is not a resolution the data support.

| market | null (last M obs) | positions | independent pairs | mean MMD² | sd | p97.5 | median σ (min–max) |
|---|---|---|---|---|---|---|---|
| BOVESPA | L (test block only) | 0 | 0.5 | n/a | n/a | n/a | n/a |
| BOVESPA | 2L (valid+test) | 1 | 1.0 | 0.1383 | 0.0000 | 0.1383 | 2.82 (2.82–2.82) |
| BOVESPA | 3L | 497 | 1.5 | 0.2645 | 0.0875 | 0.4010 | 3.52 (2.82–6.77) |
| BOVESPA | 4L | 993 | 2.0 | 0.2684 | 0.0786 | 0.3996 | 6.77 (2.82–7.50) |
| BOVESPA | 6L | 1985 | 3.0 | 0.2306 | 0.0700 | 0.3950 | 7.43 (2.82–7.93) |
| BOVESPA | all (existing null) | 3962 | 4.99 | 0.2231 | 0.0575 | 0.3851 | 7.64 (2.82–7.97) |
| FTSE | L (test block only) | 0 | 0.5 | n/a | n/a | n/a | n/a |
| FTSE | 2L (valid+test) | 1 | 1.0 | 0.1513 | 0.0000 | 0.1513 | 3.20 (3.20–3.20) |
| FTSE | 3L | 502 | 1.5 | 0.2697 | 0.0463 | 0.3378 | 3.67 (3.15–6.92) |
| FTSE | 4L | 1003 | 2.0 | 0.2606 | 0.0609 | 0.3647 | 6.90 (3.15–7.90) |
| FTSE | 6L | 2005 | 3.0 | 0.2145 | 0.0651 | 0.3601 | 7.62 (3.15–7.95) |
| FTSE | all (existing null) | 3996 | 4.99 | 0.2052 | 0.0518 | 0.3467 | 7.76 (3.15–8.04) |
| MOEX | L (test block only) | 0 | 0.5 | n/a | n/a | n/a | n/a |
| MOEX | 2L (valid+test) | 1 | 1.0 | 0.3347 | 0.0000 | 0.3347 | 3.32 (3.32–3.32) |
| MOEX | 3L | 501 | 1.5 | 0.2875 | 0.0538 | 0.3738 | 3.73 (3.28–6.90) |
| MOEX | 4L | 1001 | 2.0 | 0.2707 | 0.0748 | 0.3806 | 6.89 (3.28–7.58) |
| MOEX | 6L | 2001 | 3.0 | 0.2298 | 0.0693 | 0.3793 | 7.43 (3.28–7.88) |
| MOEX | all (existing null) | 3995 | 4.99 | 0.2238 | 0.0545 | 0.3710 | 7.43 (3.28–7.88) |
| NIFTY50 | L (test block only) | 0 | 0.5 | n/a | n/a | n/a | n/a |
| NIFTY50 | 2L (valid+test) | 1 | 1.0 | 0.1885 | 0.0000 | 0.1885 | 3.12 (3.12–3.12) |
| NIFTY50 | 3L | 497 | 1.5 | 0.2365 | 0.0665 | 0.3608 | 3.17 (3.08–6.50) |
| NIFTY50 | 4L | 993 | 2.0 | 0.2553 | 0.0679 | 0.3668 | 6.49 (3.08–7.29) |
| NIFTY50 | 6L | 1985 | 3.0 | 0.2280 | 0.0578 | 0.3656 | 7.16 (3.08–7.44) |
| NIFTY50 | all (existing null) | 3963 | 4.99 | 0.2225 | 0.0467 | 0.3618 | 7.31 (3.08–7.65) |
| SHANGHAI | L (test block only) | 0 | 0.5 | n/a | n/a | n/a | n/a |
| SHANGHAI | 2L (valid+test) | 1 | 1.0 | 0.1347 | 0.0000 | 0.1347 | 3.02 (3.02–3.02) |
| SHANGHAI | 3L | 487 | 1.5 | 0.2508 | 0.0878 | 0.3954 | 3.50 (3.02–6.83) |
| SHANGHAI | 4L | 973 | 2.0 | 0.2800 | 0.0800 | 0.3971 | 6.83 (3.02–7.76) |
| SHANGHAI | 6L | 1945 | 3.0 | 0.2518 | 0.0686 | 0.3939 | 7.47 (3.02–7.92) |
| SHANGHAI | all (existing null) | 3889 | 5.0 | 0.2436 | 0.0543 | 0.3900 | 7.63 (3.02–8.11) |

RBF MMD² at the median-heuristic bandwidth (×1). The degree-3 columns are in `part2a_null_by_recency.csv`.

**Where σ changes.** Median σ is ≈3.0–3.7 in the last 3L observations (the roughly six most recent years) and ≈6.5–7.6 as soon as one block reaches further back (see the 3L → 4L rows): the ≈7.5 is a property of blocks that contain older periods, and the test-period comparisons sit in the regime where σ ≈ 3. The mismatch is therefore real, and a same-scale null is one drawn from the recent end.

**Recomputed p-values.** Share of the 300 draws (15 market-seeds × 20) per series with p ≤ .05, RBF at the median-heuristic bandwidth, under each null. "Existing" is the Task 5 v3 result.

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control | control units ≥ half of draws / any draw (of 15) |
|---|---|---|---|---|---|---|---|
| L (test block only) — not computable (0 positions) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 2L (valid+test) — floor p = 0.5, cannot reach .05 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 |
| 3L | 27.7% | 6.0% | 3.7% | 4.7% | 0.7% | 0.0% | 0 / 0 |
| 4L | 26.0% | 5.7% | 3.7% | 4.7% | 0.7% | 0.0% | 0 / 0 |
| 6L | 28.7% | 6.3% | 3.7% | 4.7% | 0.7% | 0.0% | 0 / 0 |
| all (existing null) | 37.0% | 8.0% | 5.7% | 5.3% | 2.0% | 0.0% | 0 / 0 |

Share of draws above the null's own p97.5 (same nulls, RBF):

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control |
|---|---|---|---|---|---|---|
| 2L (valid+test) | 76.7% | 58.7% | 59.3% | 55.3% | 57.0% | 48.7% |
| 3L | 26.0% | 6.0% | 3.3% | 4.7% | 0.7% | 0.0% |
| 4L | 25.3% | 5.3% | 3.0% | 4.7% | 0.7% | 0.0% |
| 6L | 26.3% | 5.7% | 3.7% | 4.7% | 0.7% | 0.0% |
| all (existing null) | 28.7% | 6.3% | 3.7% | 4.7% | 0.7% | 0.0% |

Degree-3 polynomial, share of draws with p ≤ .05:

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control |
|---|---|---|---|---|---|---|
| 2L (valid+test) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 3L | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 4L | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 6L | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| all (existing null) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |


### 2c: scale-matched null (bandwidth fixed at the observed σ)

Both the null and the observed statistic use one Gaussian kernel with σ* = the market's median observed real-vs-generator σ (below), instead of a per-pair median heuristic. The null is the exhaustive adjacent-sliding null over all positions; the recency restrictions are as in 2a. Observed statistics were recomputed from the cached Phase B draws; against the stored median-heuristic values the maximum absolute difference is 8.61e-06 (RBF), 6.09e-05 (σ), 2.34e-06 (degree 3), over 1800 draws.

| market | σ* | null | positions | independent pairs | mean MMD² | sd | p97.5 |
|---|---|---|---|---|---|---|---|
| BOVESPA | 3.02 | 2L (valid+test) | 1 | 1.0 | 0.1261 | 0.0000 | 0.1261 |
| BOVESPA | 3.02 | 3L | 497 | 1.5 | 0.3642 | 0.2055 | 0.7359 |
| BOVESPA | 3.02 | 4L | 993 | 2.0 | 0.4153 | 0.1967 | 0.7379 |
| BOVESPA | 3.02 | 6L | 1985 | 3.0 | 0.3466 | 0.1553 | 0.7351 |
| BOVESPA | 3.02 | all (exhaustive) | 3962 | 4.99 | 0.3170 | 0.1148 | 0.7246 |
| FTSE | 3.20 | 2L (valid+test) | 1 | 1.0 | 0.1511 | 0.0000 | 0.1511 |
| FTSE | 3.20 | 3L | 502 | 1.5 | 0.3659 | 0.1467 | 0.6562 |
| FTSE | 3.20 | 4L | 1003 | 2.0 | 0.4068 | 0.1609 | 0.7051 |
| FTSE | 3.20 | 6L | 2005 | 3.0 | 0.3400 | 0.1324 | 0.6956 |
| FTSE | 3.20 | all (exhaustive) | 3996 | 4.99 | 0.3147 | 0.0986 | 0.6702 |
| MOEX | 3.07 | 2L (valid+test) | 1 | 1.0 | 0.3597 | 0.0000 | 0.3597 |
| MOEX | 3.07 | 3L | 501 | 1.5 | 0.3794 | 0.1323 | 0.6862 |
| MOEX | 3.07 | 4L | 1001 | 2.0 | 0.4132 | 0.1523 | 0.6921 |
| MOEX | 3.07 | 6L | 2001 | 3.0 | 0.3445 | 0.1282 | 0.6903 |
| MOEX | 3.07 | all (exhaustive) | 3995 | 4.99 | 0.3243 | 0.0942 | 0.6756 |
| NIFTY50 | 3.17 | 2L (valid+test) | 1 | 1.0 | 0.1848 | 0.0000 | 0.1848 |
| NIFTY50 | 3.17 | 3L | 497 | 1.5 | 0.3217 | 0.1844 | 0.6996 |
| NIFTY50 | 3.17 | 4L | 993 | 2.0 | 0.3965 | 0.1850 | 0.7062 |
| NIFTY50 | 3.17 | 6L | 1985 | 3.0 | 0.3492 | 0.1396 | 0.7054 |
| NIFTY50 | 3.17 | all (exhaustive) | 3963 | 4.99 | 0.3358 | 0.1008 | 0.6917 |
| SHANGHAI | 3.29 | 2L (valid+test) | 1 | 1.0 | 0.1198 | 0.0000 | 0.1198 |
| SHANGHAI | 3.29 | 3L | 487 | 1.5 | 0.3335 | 0.2050 | 0.7403 |
| SHANGHAI | 3.29 | 4L | 973 | 2.0 | 0.4214 | 0.2018 | 0.7440 |
| SHANGHAI | 3.29 | 6L | 1945 | 3.0 | 0.3756 | 0.1508 | 0.7421 |
| SHANGHAI | 3.29 | all (exhaustive) | 3889 | 5.0 | 0.3582 | 0.1095 | 0.7340 |

Share of the 300 draws per series (15 market-seeds × 20) with p ≤ .05 under the scale-matched null:

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control | control units ≥ half / any draw (of 15) |
|---|---|---|---|---|---|---|---|
| 2L (valid+test) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 |
| 3L | 0.7% | 1.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0 / 0 |
| 4L | 0.7% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0 / 0 |
| 6L | 0.7% | 1.0% | 0.0% | 0.3% | 0.0% | 0.0% | 0 / 0 |
| all (exhaustive) | 0.7% | 2.0% | 0.0% | 1.3% | 0.0% | 0.0% | 0 / 0 |

Share above the null's own p97.5:

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control |
|---|---|---|---|---|---|---|
| 2L (valid+test) | 76.3% | 58.7% | 61.7% | 57.0% | 58.3% | 47.3% |
| 3L | 0.7% | 0.3% | 0.0% | 0.3% | 0.0% | 0.0% |
| 4L | 0.7% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| 6L | 0.7% | 0.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| all (exhaustive) | 0.7% | 1.0% | 0.0% | 0.3% | 0.0% | 0.0% |

Mean observed MMD² at σ*, per series (mean over 15 market-seeds and 20 draws): TimeGAN 0.3109, GARCH 0.2201, QuantGAN 0.2099, GJR-GARCH 0.2058, CNN-WGAN-GP 0.1910, control 0.1497.


### Part 2: reading

Significance rate (share of the 300 draws per series with p ≤ .05, RBF) under each null:

| null | TimeGAN | GARCH | QuantGAN | GJR-GARCH | CNN-WGAN-GP | control | control units (of 15) |
|---|---|---|---|---|---|---|---|
| existing exhaustive null (per-pair σ, ≈7.5) | 37.0% | 8.0% | 5.7% | 5.3% | 2.0% | 0.0% | 0 |
| last 3L observations (per-pair σ, ≈3.2–3.7; 1.5 independent pairs) | 27.7% | 6.0% | 3.7% | 4.7% | 0.7% | 0.0% | 0 |
| last 4L | 26.0% | 5.7% | 3.7% | 4.7% | 0.7% | 0.0% | 0 |
| last 6L | 28.7% | 6.3% | 3.7% | 4.7% | 0.7% | 0.0% | 0 |
| scale-matched, all positions (σ* fixed, ≈3.0–3.3; ≈5 independent pairs) | 0.7% | 2.0% | 0.0% | 1.3% | 0.0% | 0.0% | 0 |
| scale-matched, last 3L | 0.7% | 1.0% | 0.0% | 0.7% | 0.0% | 0.0% | 0 |

1. **A test-period-only null at the same length does not exist** (0 positions; 1 with validation included). What can be said comes from the sensitivity nulls, none of which has more than about 1.5 independent block pairs at the σ ≈ 3 end and about 5 at the full end. The p-values in the 3L/4L/6L rows are formal (positions overlap almost entirely); the rates are useful as a comparison across nulls, not as calibrated test sizes.
2. **The σ mismatch is real** — the median σ is ≈3.2–3.7 for the last 3L observations and ≈6.5–7.6 as soon as one block reaches into older data — **but correcting it does not move the control**: 0 of 15 units and 0 of 300 draws are significant under every null above. The control's MMD² (0.14–0.18 at the median heuristic) sits at or below every null's median. Part 1 is the consistent finding: the statistic does not see the shuffle, whichever null it is compared with.
3. **What the null does change is how many generator draws look distinguishable.** Restricting to the recent end (matched σ, per-pair median heuristic) lowers TimeGAN from 37% to 26–29%, GARCH from 8% to about 6%, QuantGAN from 6% to 4%, GJR-GARCH 5% → 5%, CNN-WGAN-GP 2% → 1%. Fixing the kernel bandwidth at the observed σ* for both null and observed statistic (so both are on one scale) collapses every series to ≤ 2%, TimeGAN included: at the kernel scale of the test comparisons, two adjacent real blocks from different periods are farther apart (null mean 0.31–0.36, p97.5 0.67–0.73) than any generator is from the test block (mean 0.19–0.31; control 0.15).
4. **Ranking of the 37%.** Under the existing null only TimeGAN is well above the 5% a size-α test would give for indistinguishable series (GARCH 8%, QuantGAN 5.7%, GJR-GARCH 5.3%, CNN-WGAN-GP 2%, and draws within a generator share weights). Under the recent-end nulls TimeGAN is still the only series clearly above 5% (26–29%); under the scale-matched null none is. So "TimeGAN is farther from the real test block than the others" survives one kind of null and not the other; "the control is not detected" survives all of them.
5. **The null itself answers a different question** than the one asked of the generator. A block-pair null measures how much two adjacent real blocks differ, which mixes sampling noise with period-to-period non-stationarity; a generator is asked to reproduce one specific period. A wide null therefore lowers power for every series, control included. A within-period null (independent real replicates of the same period) is not available from one realisation of each market.

Sensitivity to the reproduction: cached draws re-encoded on the GPU reproduce the stored median-heuristic statistics to at most 8.6e-06 (RBF), so the 2c rates are on the same draws as Task 5 v3.

## Part 3: trivial fixes

**CLAUDE.md §6, "15 real-versus-real splits".** Already corrected earlier in this session (your mid-turn message); verified present now: `.claude/CLAUDE.md` lines 170–183 state `n_rep=20`, "it has used 20 since 2026-08-29 and no version has ever run 15". That edit went further than substituting the count: the paragraph now also records that 0.506 ± 0.084 is not the output of `compute_auc_null` (first appears 2026-08-30 as "15 seeds on identical distributions", independent samples, script not in the repository), that the per-market null means run 0.48–0.79, that the legacy cut scheme is unrepaired, and that repointing `discriminative_auc_absz` at another null is a §5 decision. If you wanted the count alone changed, say so and I will narrow it; I did not revert. Report text, `report_data.py` and `PROJECT_STATE.md` were not touched.

**sklearn in the recorded library versions.** `3_4_integrated_pipeline.ipynb`, cell 20, two places (4 lines): the `run_metadata` dict written to `pipeline_run_metadata.json` by `run_complete_pipeline()` now has `'sklearn'`, and `_library_versions()` (which Phase B's pre-flight compares against the recorded metadata) has it too. Verified: `_library_versions()` returns `sklearn 1.9.0`; `verify_notebook.py` exits 0; the notebook still has 22 cells; `git diff --stat` on the notebook is 5 insertions, 3 deletions (exactly these lines). Nothing else in the notebook changed; no metric, rank or column list was touched.

Consequence to know about: the existing production `pipeline_run_metadata.json` has no sklearn entry, so Phase B's pre-flight now adds one warning against it (`library sklearn: not recorded in pipeline_run_metadata.json`). It is a warning, not a failure; it becomes a stop only when the pre-flight is called with `strict=True`. Pre-existing warning (unchanged): numpy recorded 2.5.3, current 2.5.2 in `phase_b_venv`. The embedding-distance JSONs already written record sklearn 1.9.0 in their own `libraries` block.

## Part 4: is the |z|>3 cluster an artifact of RNG history?

**Data.** Two further market-seed pairs, chosen before looking: FTSE/43 and NIFTY50/44. Phase B, 20 fresh draws each (`_draw_seed(seed, k)`), run under `phase_b_venv` (pandas 2.3.3) into the pipeline's own `<seed dir>/phase_b_COMMIT_DRIFT/` (drift accepted and stamped, as for BOVESPA/42; 14 s each). Nothing else was written. Same recipe as the BOVESPA/42 count: for each GAN and each of the 19 metrics in the comparison CSV (composite_rank excluded), z = (recorded − mean of the 20 fresh draws) / their sd (ddof 0). The BOVESPA/42 result reproduces: 11 of 57 (TimeGAN 1, QuantGAN 8, CNN-WGAN-GP 2), every one of them with the recorded value above all 20 fresh draws.

**Yardstick.** "~0.15 expected" is the Gaussian figure (57 × 0.0027). It is not the right reference here: the metrics are heavy-tailed and a sd from 20 draws is a noisy scale, so fresh draws scored against each other produce far more |z|>3 cells. The reference used below is empirical: each fresh draw is treated in turn as "the recorded one" and scored against the other 19 (same recipe, 20 replicates per unit).

| unit | \|z\|>3 of 57 (TimeGAN / QuantGAN / CNN-WGAN-GP) | recorded above all 20 / below all 20 | fresh-vs-fresh reference: mean (max) count | share of fresh draws scoring ≥ this count |
|---|---|---|---|---|
| BOVESPA/42 (reference) | **11** (1 / 8 / 2) | 20 / 0 | 1.10 (7) | 0 of 20 (p ≈ 1/21) |
| FTSE/43 | **2** (0 / 0 / 2) | 1 / 1 | 1.85 (8) | 30% |
| NIFTY50/44 | **1** (0 / 0 / 1) | 1 / 0 | 1.45 (5) | 55% |

The cluster does not repeat: 2 and 1 of 57, both ordinary against the fresh-draw reference.

**Supplementary, all 15 market-seeds** (from the earlier scratch Phase B run under pandas 3.0.5, which reproduced the pinned environment's results for BOVESPA/42 exactly; FTSE/43 and NIFTY50/44 above agree with it cell for cell): 33 of 855 cells at |z|>3 (3.9%) against 20.6 expected from the fresh-vs-fresh reference; without BOVESPA/42, 22 against 19.5. Cells outside the 20-draw range: 74 of 855 (8.7%) against 9.5% expected from exchangeability (2/21). Above-all / below-all: 46 / 28 with BOVESPA/42, **26 / 28 without it** — the whole asymmetry is that one unit. Median percentile of the recorded draw within its 20 fresh draws (per unit and model, over 19 metrics; 50 = typical): TimeGAN 54 (median 50), QuantGAN 56 (median 50), CNN-WGAN-GP 51.7 (median 45); sign test over 45 unit-model pairs p = 0.64 (0.32 without BOVESPA/42). Unit-level p-values (1 + #fresh ≥ recorded)/21 combined by Fisher: χ²(30) = 26.4, p = 0.65 (approximate: discrete p-values, correlated metrics within a unit). Two of 15 units sit at p ≤ 0.10 (BOVESPA/42 0.048, MOEX/43 0.095); the chance that at least one of 15 exchangeable units reaches p ≤ 1/21 is 0.52.

**Reading.**
1. *No systematic offset between the recorded draws and fresh draws.* Across 14 units the recorded values are indistinguishable from fresh ones on every summary I can construct (outside-range rate, above/below balance, median percentile, sign test, Fisher). That includes TimeGAN, whose noise is drawn on the CPU, and QuantGAN / CNN-WGAN-GP, whose noise is drawn on the GPU, so a device-path difference producing a common tilt is not visible either.
2. *BOVESPA/42 is one unit in 15 at p ≈ 0.05*, which is what exchangeable draws produce about half the time somewhere in 15 units. Its eight QuantGAN cells are not eight events: they are the scale/tail block of the metrics responding together to one large-variance series (recorded std_diff 0.0235, above all 20 fresh draws, z = 3.6; a generated sd of about 0.034 against the real block's 0.0103, if the difference is upward). Counting cells overstates the evidence by the number of correlated metrics.
3. *Your suspected explanation is consistent with this and not contradicted, but it is not verified for that draw.* A valid RNG stream, clean or accumulated, gives exchangeable noise, so "the recorded draw is an ordinary draw from an unrecorded state" predicts exactly what is seen: no aggregate tilt and an occasional tail unit. What it cannot be given is a direct test for QuantGAN and CNN-WGAN-GP: the CUDA RNG state at the time of the recorded draw was not saved and cannot be restored, so that draw cannot be replayed. The claim rests on aggregate consistency, not on reproducing the draw.

**Incidental, for completeness (not acted on; §5 territory).** The series saved in `downstream_utility_inputs.npz` is **not** the series the recorded metrics were computed on, for the three GAN models. This was established for TimeGAN earlier in the session by the CPU-RNG replay (first `generate()` call → the metrics, never stored; second call → the npz); here it is checked for all three GANs across all 15 market-seeds. `comprehensive_evaluation` (notebook cell 20) calls `model.generate(len(real_data))` for the metrics (line 71) and a second, separate `model.generate(...)` for the downstream-utility inputs (line 186). GARCH and GJR-GARCH re-seed on every call, so both calls return one series; the GANs do not. Check: |skew(real) − skew(series in the npz)| against the recorded `skewness_diff`: GARCH 15/15 and GJR-GARCH 15/15 agree (relative gap ~1e-7, float32 storage); TimeGAN, QuantGAN and CNN-WGAN-GP 0/15. Example, BOVESPA/42 QuantGAN: 0.0014 from the saved series, 1.684 recorded. Consequences: (i) the series behind the recorded GAN metrics was never stored, only its metrics (TimeGAN's can be regenerated by the CPU-RNG replay; QuantGAN's and CNN-WGAN-GP's, drawn on the CUDA generator, cannot); (ii) for the GANs, the pooled downstream-utility backtest and the ranked metrics use different draws. For the econometric pair neither applies. No code or result was changed.

## Part 5: encoder robustness (same 15 market-seeds, same 20 cached draws, same statistic; only the encoder changes)

Two further encoder configurations were trained per market (5 markets; seed 42; GPU training is not bit-reproducible, so a retrain will not reproduce these encoders to the last digit — loss trajectories and encoder files are in `thesis_results/production/<market>/ts2vec_<config>/`). The effective-sample-size diagnostic is not repeated. Everything else is as in Task 5 v3: unbiased MMD², RBF at the per-pair median-heuristic bandwidth (primary), degree-3 polynomial (secondary), exhaustive adjacent-sliding null computed **with the encoder under test**, p = (1 + #{null ≥ obs})/(B + 1). Verified draw identity: the cached draws reproduce the main run's stored statistics to at most 8.6e-06 (Part 2c).

**Encoders.**

| config | market | embedding dim | training | seconds | final loss | stop |
|---|---|---|---|---|---|---|
| main | BOVESPA | 100 | 88 epochs × 479 | 1760 | 0.3378 | patience=10 (no new best for 10 epochs) |
| main | FTSE | 100 | 104 epochs × 483 | 1998 | 0.3313 | patience=10 (no new best for 10 epochs) |
| main | MOEX | 100 | 98 epochs × 483 | 1932 | 0.3820 | patience=10 (no new best for 10 epochs) |
| main | NIFTY50 | 100 | 77 epochs × 479 | 1553 | 0.3557 | patience=10 (no new best for 10 epochs) |
| main | SHANGHAI | 100 | 100 epochs × 470 | 1921 | 0.3449 | patience=10 (no new best for 10 epochs) |
| d320 | BOVESPA | 320 | 88 epochs × 479 | 2180 | 0.2621 | patience=10 (no new best for 10 epochs) |
| d320 | FTSE | 320 | 78 epochs × 483 | 2072 | 0.2919 | patience=10 (no new best for 10 epochs) |
| d320 | MOEX | 320 | 78 epochs × 483 | 2072 | 0.2947 | patience=10 (no new best for 10 epochs) |
| d320 | NIFTY50 | 320 | 88 epochs × 479 | 2181 | 0.2711 | patience=10 (no new best for 10 epochs) |
| d320 | SHANGHAI | 320 | 74 epochs × 470 | 1936 | 0.2832 | patience=10 (no new best for 10 epochs) |
| tsgbench | BOVESPA | 100 | 600 iterations (fit default 600) | 26 | 2.2554 | n_iters reached (default); no early stopping |
| tsgbench | FTSE | 100 | 600 iterations (fit default 600) | 26 | 2.1240 | n_iters reached (default); no early stopping |
| tsgbench | MOEX | 100 | 600 iterations (fit default 600) | 26 | 2.2722 | n_iters reached (default); no early stopping |
| tsgbench | NIFTY50 | 100 | 600 iterations (fit default 600) | 26 | 2.1823 | n_iters reached (default); no early stopping |
| tsgbench | SHANGHAI | 100 | 600 iterations (fit default 600) | 26 | 2.1720 | n_iters reached (default); no early stopping |

Note (tsgbench): the logged loss is the first-epoch mean; the run stops 121 iterations into the second epoch and that partial epoch is not logged by upstream `fit()`.

**main: main run (d = 100, up to 300 epochs, patience 10).** Mean over 15 market-seeds; MMD² is the per-unit mean over 20 draws, "draw sd" the per-unit sd over draws, averaged.

| series | MMD² (RBF) | draw sd | draws p ≤ .05 | draws above null p97.5 | MMD² (deg 3) | draws p ≤ .05 (deg 3) | mean rank (1 = closest) |
|---|---|---|---|---|---|---|---|
| TimeGAN | 0.2952 | 0.0543 | 37.0% | 28.7% | 0.4657 | 0.0% | 5.80 |
| GARCH | 0.2133 | 0.0735 | 8.0% | 6.3% | 0.3267 | 0.0% | 4.00 |
| QuantGAN | 0.2041 | 0.0623 | 5.7% | 3.7% | 0.3107 | 0.0% | 3.73 |
| GJR-GARCH | 0.2016 | 0.0619 | 5.3% | 4.7% | 0.3009 | 0.0% | 3.00 |
| CNN-WGAN-GP | 0.1916 | 0.0478 | 2.0% | 0.7% | 0.2725 | 0.0% | 2.73 |
| control | 0.1612 | 0.0175 | 0.0% | 0.0% | 0.1969 | 0.0% | 1.73 |

Control has the lowest MMD² of the six series in **11 of 15** units (RBF, mean rank 1.73); degree 3: 12 of 15 (mean rank 1.33); scale-matched RBF (σ* fixed, as Part 2c): 12 of 15 (mean rank 1.40).

**d320: d = 320 (otherwise as main).** Mean over 15 market-seeds; MMD² is the per-unit mean over 20 draws, "draw sd" the per-unit sd over draws, averaged.

| series | MMD² (RBF) | draw sd | draws p ≤ .05 | draws above null p97.5 | MMD² (deg 3) | draws p ≤ .05 (deg 3) | mean rank (1 = closest) |
|---|---|---|---|---|---|---|---|
| TimeGAN | 0.2788 | 0.0450 | 31.7% | 23.7% | 0.0652 | 0.0% | 5.73 |
| GARCH | 0.2187 | 0.0692 | 9.3% | 8.0% | 0.0516 | 0.3% | 4.00 |
| QuantGAN | 0.2091 | 0.0572 | 6.3% | 6.0% | 0.0475 | 0.0% | 4.13 |
| GJR-GARCH | 0.2080 | 0.0584 | 7.7% | 7.0% | 0.0469 | 0.0% | 3.27 |
| CNN-WGAN-GP | 0.1919 | 0.0394 | 2.3% | 1.7% | 0.0403 | 0.0% | 2.60 |
| control | 0.1606 | 0.0102 | 0.0% | 0.0% | 0.0294 | 0.0% | 1.27 |

Control has the lowest MMD² of the six series in **12 of 15** units (RBF, mean rank 1.27); degree 3: 15 of 15 (mean rank 1.00); scale-matched RBF (σ* fixed, as Part 2c): 15 of 15 (mean rank 1.00).

**tsgbench: TSGBench literal (d = 100, batch 8, fit() default 600 iterations, no early stopping).** Mean over 15 market-seeds; MMD² is the per-unit mean over 20 draws, "draw sd" the per-unit sd over draws, averaged.

| series | MMD² (RBF) | draw sd | draws p ≤ .05 | draws above null p97.5 | MMD² (deg 3) | draws p ≤ .05 (deg 3) | mean rank (1 = closest) |
|---|---|---|---|---|---|---|---|
| TimeGAN | 0.3823 | 0.0975 | 10.3% | 6.3% | 0.0601 | 2.7% | 5.60 |
| GARCH | 0.2629 | 0.1294 | 5.3% | 2.3% | 0.0968 | 5.0% | 4.20 |
| QuantGAN | 0.2501 | 0.1225 | 4.7% | 2.0% | 0.0429 | 4.0% | 3.87 |
| GJR-GARCH | 0.2440 | 0.1108 | 3.7% | 1.0% | 0.0719 | 4.3% | 3.13 |
| CNN-WGAN-GP | 0.2146 | 0.0916 | 1.3% | 1.0% | 0.0288 | 0.0% | 2.53 |
| control | 0.1435 | 0.0202 | 0.0% | 0.0% | 0.0120 | 0.0% | 1.67 |

Control has the lowest MMD² of the six series in **11 of 15** units (RBF, mean rank 1.67); degree 3: 14 of 15 (mean rank 1.07); scale-matched RBF (σ* fixed, as Part 2c): 13 of 15 (mean rank 1.13).

**Null geometry per encoder** (RBF, per-pair median heuristic): the mismatch that motivated Part 2 is a property of the encoder too.

| config | market | null mean | null sd | null p97.5 | median σ, null | median σ, real-vs-generator |
|---|---|---|---|---|---|---|
| main | BOVESPA | 0.2231 | 0.0575 | 0.3851 | 7.64 | 3.02 |
| main | FTSE | 0.2052 | 0.0518 | 0.3467 | 7.76 | 3.20 |
| main | MOEX | 0.2238 | 0.0545 | 0.3710 | 7.43 | 3.07 |
| main | NIFTY50 | 0.2225 | 0.0467 | 0.3618 | 7.31 | 3.17 |
| main | SHANGHAI | 0.2436 | 0.0543 | 0.3900 | 7.63 | 3.29 |
| d320 | BOVESPA | 0.2217 | 0.0485 | 0.3535 | 8.43 | 3.13 |
| d320 | FTSE | 0.2196 | 0.0467 | 0.3587 | 8.12 | 3.08 |
| d320 | MOEX | 0.2239 | 0.0412 | 0.3430 | 7.81 | 2.98 |
| d320 | NIFTY50 | 0.2342 | 0.0438 | 0.3677 | 8.13 | 3.14 |
| d320 | SHANGHAI | 0.2442 | 0.0472 | 0.3654 | 8.24 | 3.20 |
| tsgbench | BOVESPA | 0.2368 | 0.1338 | 0.5981 | 1.24 | 1.16 |
| tsgbench | FTSE | 0.2268 | 0.1060 | 0.5507 | 1.45 | 1.42 |
| tsgbench | MOEX | 0.2222 | 0.1335 | 0.5520 | 1.12 | 1.15 |
| tsgbench | NIFTY50 | 0.2778 | 0.1570 | 0.7067 | 1.34 | 1.32 |
| tsgbench | SHANGHAI | 0.3287 | 0.1991 | 0.8812 | 1.52 | 1.61 |


### Part 5: reading

1. **The control still has the lowest mean MMD² of the six series under both new encoders.** RBF, mean over 15 market-seeds: 0.161 (main), 0.161 (d = 320), 0.144 (TSGBench-literal); the next-lowest series, CNN-WGAN-GP, is at 0.192 / 0.192 / 0.215. Unit by unit it is the lowest in 11 / 12 / 11 of 15 (RBF), 12 / 15 / 14 (degree 3) and 12 / 15 / 13 (scale-matched RBF). The exceptions are six distinct units (3–4 per configuration), concentrated in FTSE/42–44, MOEX/42–43 and NIFTY50/42, with margins of 0.001–0.025 against a control MMD² of about 0.16; only FTSE/44 is an exception in all three. On FTSE/44 the TSGBench-literal encoder puts TimeGAN, the series that is farthest on average, below the control.
2. **The control is never detected.** 0.0% of its 300 draws reach p ≤ .05 under every encoder (RBF and degree 3), and 0.0% exceed the null's p97.5. Its MMD² is below the median of every null in every market (existing, 3L, 4L, 6L, scale-matched; Part 2), so this holds for the encoder axis and the null axis together.
3. **What the encoder does change.** (a) d = 320 leaves the picture essentially as it was: TimeGAN 31.7% of draws at p ≤ .05 (37.0% in the main run), GARCH 9.3% (8.0%), QuantGAN 6.3% (5.7%), GJR-GARCH 7.7% (5.3%), CNN-WGAN-GP 2.3% (2.0%). (b) The TSGBench-literal encoder has a much wider null (p97.5 0.55–0.88 against 0.35–0.39), and every rate falls: TimeGAN 10.3%, GARCH 5.3%, QuantGAN 4.7%, GJR-GARCH 3.7%, CNN-WGAN-GP 1.3%. The ordering of the six series is otherwise stable: TimeGAN is farthest in all three (mean rank 5.60–5.80 of 6), then GARCH/QuantGAN (which swap places at d = 320), then GJR-GARCH, CNN-WGAN-GP, control.
4. **The σ mismatch is a property of the trained encoder, not of the null construction.** At d = 320 it is as in the main run (median σ 7.8–8.4 in the null against 3.0–3.2 real-vs-generator). With the TSGBench-literal encoder (600 iterations, about 1.25 epochs; epoch-mean loss 2.1–2.3 against 0.33–0.38 for the fully trained main encoders) the two agree (1.1–1.5 against 1.2–1.6), so nothing needs recalibrating there — and the control is still not detected and still has the lowest mean MMD².
5. **The two configurations do not prove much on their own.** One encoder per configuration and market (seed 42, GPU training not bit-reproducible), so encoder-training variance is not separated from the configuration effect; the 0.001-scale per-unit exceptions are inside that noise. The degree-3 MMD² values are on different scales across encoders (0.20–0.47 main, 0.03–0.07 d = 320, 0.01–0.10 TSGBench-literal) because that kernel is not scale-normalised; only within-encoder comparisons and p-values are meaningful. Window length, max-pooling, z-scoring and the 20 draws are shared by all three configurations; a different pooling or window length is a different question.

---

## Answers

**1. Does TS2Vec see a shuffle?** Partly, and not through the MMD. The embeddings are not permutation-invariant: a logistic-regression probe fitted on validation-block windows separates real from permuted test-block windows at AUC 0.52 (BOVESPA), 0.61 (FTSE), 0.61 (MOEX), 0.61 (NIFTY50) and 0.82 (SHANGHAI), mean 0.63, against 0.67 for a probe on three volatility-clustering features (0.50–0.76). That is about the ordering information a windowed ACF summary carries and no more, and for BOVESPA it is not distinguishable from chance. The MMD does not use it: RBF and degree-3 MMD² between the real test block and 20 permutations of it is 0.14–0.18 (below every null median), p = 0.71–1.00, 0 of 100 permuted draws at p ≤ .05 — the same as Task 5 v3's control. So the failure to detect the shuffle belongs to the statistic at this effective sample size, not to the representation discarding order; but the ordering signal that exists is weak (AUC ≈ 0.6 in three markets, with no significance test that respects the 127/128 window overlap).

**2. Does the control still rank lowest under a test-period null and two other encoder configurations?** Yes on average; not in every unit; and never detected. *Test-period null:* it cannot be built (0 positions at the evaluation length inside the test block; 1 with the validation block). The null does not affect the ranking of MMD² (only p-values), so the sensitivity nulls (last 3L/4L/6L observations; bandwidth fixed at the observed σ*) test whether the control becomes detectable: it does not (0 of 15 units, 0 of 300 draws, under every variant that can resolve p ≤ .05; the single-position 2L null has a p-value floor of 0.5), and its MMD² is below every null's median in every market. The generators' significance rates do move: TimeGAN 37% → 26–29% with a recent-end null and 0.7% with a scale-matched one; the other four fall to ≤ 6.3% with a recent-end null and ≤ 2% with a scale-matched one. *Encoders:* the control has the lowest mean MMD² of the six series under all three (0.161 main, 0.161 d = 320, 0.144 TSGBench-literal; next 0.192 / 0.192 / 0.215) and is the lowest series in 11 / 12 / 11 of 15 units by RBF (12 / 15 / 14 by degree 3), with the exceptions on FTSE and MOEX at margins of 0.001–0.025. The consequence for the metric's validity is the same in every configuration tried: a statistic that ranks the order-destroying control as the closest of the six series to real data does not check temporal structure.

**3. Is the |z|>3 cluster an artifact of RNG history?** Consistent with it, not verified, and not needed to explain it. The cluster does not repeat: FTSE/43 has 2 of 57 and NIFTY50/44 has 1, both ordinary against a fresh-vs-fresh reference (the Gaussian 0.15 is the wrong yardstick; fresh draws scored against each other give 1.0–1.9 per unit). Across all 15 market-seeds the recorded draws show no systematic offset from fresh ones (outside-range rate 8.7% against 9.5% expected, above/below balance 26/28 without BOVESPA/42, median percentile 50 / 50 / 45 for TimeGAN / QuantGAN / CNN-WGAN-GP (means 54 / 56 / 52), Fisher p = 0.65). BOVESPA/42 is one unit at p ≈ 0.05 out of 15, mostly one large-variance QuantGAN draw counted eight times through correlated metrics. That fits "a recorded draw is one ordinary draw from an unrecorded RNG state" — but the CUDA RNG state behind the recorded QuantGAN and CNN-WGAN-GP draws was never saved, so that draw cannot be replayed and the explanation cannot be confirmed for it. Related and newly quantified: for the three GANs the series in `downstream_utility_inputs.npz` is a second `generate()` call, not the series the recorded metrics were computed on (0 of 15 agree for each GAN; 15 of 15 for GARCH and GJR-GARCH).

---

**What changed on disk.** Modified, tracked: `3_4_integrated_pipeline.ipynb` (sklearn added to `run_metadata` and `_library_versions()`, 4 lines; `verify_notebook.py` exits 0; 22 cells), `.claude/CLAUDE.md` (§6 AUC-null paragraph, edited earlier in this session), `.gitignore` (encoder files under `thesis_results/production/*/ts2vec*/` ignored). Not touched: FIDELITY_COLS, TEMPORAL_COLS, the composite, `{market}_metrics.csv`, `generate_report.py`, `report_data.py`, any `.tex`. New (untracked): `analysis/overnight_2026-09-21/` (scripts), `thesis_results/production/overnight_2026-09-21/` (JSON/CSV, cached draws, logs), `thesis_results/production/{FTSE/seed43,NIFTY50/seed44}/phase_b_COMMIT_DRIFT/`, `thesis_results/production/{market}/ts2vec_d320/` and `ts2vec_tsgbench/` (encoders and training records). No generator retrained; Phase A not re-run.

**Open, waiting on you.** (i) Your Task 11 (purged/embargoed CV option, scenario matrix S0–S5, covariance gate) is untouched since it was received. (ii) Part 3's CLAUDE.md edit is broader than the one-word substitution requested; say if you want it narrowed. (iii) The metrics-draw vs downstream-draw split (Part 4) is a §5 item: reported, not changed.
