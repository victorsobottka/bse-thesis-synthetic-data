PROJECT_STATE — synthetic financial time series benchmark

Last updated: after the five-model production run (commit f03ab89), report restructured to Nature-style, knowledge base partially updated.

1. What this is

Research project with UPC, co-author Ariel Duarte López. Began as a BSE master's thesis, now beyond that scope. Target venue: Quantitative Finance or ACM ICAIF (NMI assessed and rejected — domain benchmark, small dataset, niche occupied by CTBench/TSGBench/SFAG).

Repo: github.com/victorsobottka/bse-thesis-synthetic-data (public, contains unpublished research) Main artifact: 3_4_integrated_pipeline.ipynb Standing constraints: .claude/CLAUDE.md (262 lines, tracked) Knowledge base: 191+ pages, 16 chapters — gitignored, sources local only

2. Data

Five BRICS indices, 20 years, 24,758 observations (2006-08-30 → 2026-08-28):

Market	n	vol %	excess kurtosis	Hill α
BOVESPA	4,953	1.63	10.36	2.89
FTSE_JSE	4,997	1.20	5.72	3.21
MOEX	4,994	1.87	65.63	2.54
NIFTY50	4,954	1.29	14.52	2.71
SHANGHAI	4,860	1.48	5.77	2.77

Splits 80/10/10 (~3,970 / ~496 / ~496). Pooled test = 2,479. MOEX has a −33.3% day (2022-02-24) and 27-day suspension — real, never interpolated.

3. Current results (5 models × 5 markets × 3 seeds × 5 folds, 9,000 generator updates)
Model	Family	Composite	Fidelity	Temporal	Wins	Fit (s)
GJR-GARCH	econometric	2.524	2.667	2.381	2/15	<1
CNN-WGAN-GP	gradient	2.605	2.429	2.781	6/15	180
GARCH	econometric	2.967	3.152	2.781	3/15	<1
QuantGAN	gradient	3.041	3.138	2.943	2/15	481
TimeGAN	gradient	3.864	3.614	4.114	2/15	81

Headline: a six-parameter GJR-GARCH fitted in under a second ranks first overall and leads the temporal family, against a 276,481-parameter QuantGAN trained for 481 s per market-seed. Deep models lead fidelity; econometric models lead dynamics.

Gradient family takes 10 of 15 composite wins; econometric 5. The econometric edge is in mean rank, not a sweep.

4. Architecture

Two families, parity scoped to the gradient family:

Gradient: TimeGAN (GRU autoencoder, four-phase), QuantGAN (TCN-WGAN-GP), CNN-WGAN-GP (CNN-WGAN-GP)
Econometric: GARCH(1,1)-t, GJR-GARCH(1,1)-t, MLE

Two evaluation tracks:

Track A — fixed 80/10/10 split, models trained once, produces the 19 metrics and all rankings (75 trainings)
Track B — walk-forward, reassembles the full series, retrains from scratch each fold, produces per-fold diagnostics only (375 trainings, ~83% of wall-clock)
5. Key decisions and evidence

Budgets in gradient steps, never epochs — epochs gave a 492× asymmetry. Parity on generator updates; TimeGAN pre-training excluded (Yoon 2019); n_critic=5 (Gulrajani 2017).

z-score + tanh(z/3), not min-max — at Hill α 2.54–3.21 one extreme day sets the scale; 98% of MOEX occupied 15% of [−1,1]. Changing only this moved QuantGAN std from 3.7× to 1.2× real.

GARCH fits raw returns ×100, no normalisation — tanh squash would destroy the variance dynamics it models.

Two-stage GARCH fit at the integrated boundary — fit unconstrained, record persistence, refit at α+β ≤ 1−δ (δ=1e-4) only when within 1e-6 of 1. Justified by bimodality: 6 of 110 refits within 1.7e-13 of 1, all others ≥1.23e-4 below. Cost 0.0006–0.0197 log-likelihood. (Lamoureux & Lastrapes 1990; Mikosch & Stărică 2004.)

kurtosis/skewness descriptive-only — population kurtosis infinite at α<4. tail_index_diff (Hill 1975) is the ranked tail metric.

Fidelity/temporal rank split — unweighted mean won by the shuffled control, 1.24 vs 2.47.

Control excluded from rank competition — its fidelity_rank is 1.000 by construction.

Downstream utility pooled — n=2,479. QLIKE needs n≳600; per-market at ~496 the control wins qlike_rank in 33 of 75 comparisons (44%).

Walk-forward over random CV — shuffling moves the AUC null 0.506 → 0.584.

Determinism deliberately not enforced, cost measured.

6. Verified evidence
Claim	Value
Permutation-invariant metrics	9 of the 19 per-market metrics (7 fidelity + skewness_diff + kurtosis_diff); DESCRIPTIVE_COLS has 7 members, 5 of them in {market}_metrics.csv
Discriminative AUC null	0.506 ± 0.084 (15 seeds)
Wasserstein floor = real MAD	BOVESPA 0.00770; collapsed model 0.0077 beat working model 0.0083
Integrated-boundary hits	12 of 180 windows — SHANGHAI main (3,888) + fold 3 (3,240); NIFTY50 fold 0 (829); identical across all 3 seeds
Pre-constraint explosions	Wasserstein 74.795 / 7.814 / 43.006 vs median 0.0025, AUC exactly 1.000
Nondeterminism at fixed seed	TimeGAN + control bit-identical; QuantGAN Wasserstein CV 37%, tail_index_diff CV 79%
Ranking is budget-dependent	QuantGAN won 5/5 at 1,000 updates; loses at 9,000
Control on acf_absolute_mae	0.0730 — beats QuantGAN (0.0981) and TimeGAN (0.1232)
Walk-forward AUC fold effect	QuantGAN 0.921 (fold 0, ~830 pts) → 0.647 (fold 4, ~4,050 pts)
Kupiec power	n=125: 17%; n=623: 56%; n=2500: 99%
7. Gotchas

Persistent volume state. Three production launches were lost to stale state: old thesis_results/ inherited by a new pod, git reset --hard restoring deleted results, and once running executed_run.ipynb (the previous run's output) instead of the source notebook. Always verify before launching: git log --oneline -1 plus a check that FinGAN is absent and constraint_applied present in the notebook source. run_production.sh now does this.

VS Code / Claude Code write conflict — close the notebook tab first. verify_notebook.py must exit 0 before every commit.

run_config.json now records the model roster — a roster mismatch forces re-run.

Library versions recorded: Python 3.12.3, torch 2.13.0+cu130, arch 8.0.0, statsmodels 0.15.0, scipy 1.18.1, numpy 2.5.3, pandas 2.3.3.

8. Contribution
9 of 19 standard metrics are permutation-invariant; a shuffled control wins an unweighted composite
Wasserstein has a hard floor at the real data's MAD
Kurtosis is undefined for this data (α < 4)
Normalisation can matter more than architecture
Nominal budget parity is not parity
A collapsed generator wins ACF volatility-clustering metrics, beating the shuffled control
Run-to-run nondeterminism can exceed between-model differences
Model ranking reverses between 1,000 and 9,000 generator updates — benchmark conclusions may be budget artifacts
A six-parameter econometric model beats three deep generators at ~1/1,200 the compute
GARCH can converge to non-stationary parameters and generate explosive paths, undetected by convergence diagnostics
9. Open items
Item	Priority
Knowledge base: §8.9 says "GARCH is never a generative model" (now false); no IGARCH section; no GJR section; 25 FinGAN references	high
Diffusion arm — DDPM first to de-risk integration, then Diffusion-TS	highest research
LLM arm (DeepSeek, LoRA)	medium
Chapter 14 of KB frames field as GAN-vs-GAN	medium
Report at 47 pages; Results has 12 subsections, some single-finding	low
Repo public with unpublished research	open
10. Workflow

Tasks go to Claude Code as a 🧑 YOU framing block (problem, scope, standing constraints) plus a 🤖 CLAUDE CODE numbered block. Model/effort stated at the end. .claude/CLAUDE.md §5 = report-and-stop rather than fix if a change would touch metric definitions or ranking logic.

RunPod: RTX 4090 at $0.74/hr, ~18 h per full run (~$14). Clone to /workspace, venv there, register the thesis kernel pointing into it, verify argv[0], launch via ./run_production.sh inside tmux. ~$40 of $70 budget spent.

Next decision needs: score-based generative modelling (Ho 2020; Song 2021) for diffusion; two-sample testing for dependent data (Gretton 2012, MMD) to make the metric audit theoretical rather than empirical; EVT (Embrechts et al. 1997) for tail-estimator stability.