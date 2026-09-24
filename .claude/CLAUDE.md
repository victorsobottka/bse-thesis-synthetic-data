# CLAUDE.md — standing constraints for this repository

BSE Master Thesis: benchmarking GANs for synthetic financial time series.
Read this before editing anything.

---

## 1. Workflow

### `verify_notebook.py` must exit 0 immediately before every commit

```bash
python verify_notebook.py   # must exit 0
```

A FAIL is a regression, not a warning. Do not commit through one.

Run it immediately before `git add`, not several steps earlier. `SMOKE_TEST =
True` was carried through eight commits despite verify having passed during the
session that produced them — the check had been run, then the file changed, then
the commit was made. Verification is only evidence about the state it actually
read.

As of 2026-09-05, `verify_notebook.py` no longer checks `SMOKE_TEST` at all — see
§4. That flag is now unguarded: nothing here will stop a commit that claims real
results while `SMOKE_TEST = True`. Read the assignment yourself before any commit
that is meant to report real results, not just before a checkpoint commit.

---

## 2. Do not remove

### `[DIAG]` and `[WARNING]` output

This instrumentation is not clutter. It located three separate bugs:

| Instrument | Found |
|---|---|
| `[DIAG]` latent statistics | latent collapse — mean 0.5069, sd 0.04, autoencoder R² ≈ 0 |
| `[WARNING]` post-generation guard | mode collapse — generated sd 0.0002 vs real 0.0103, ACF(1) = 0.831 |
| step-count printout | the 492× training-budget gap between models |

Each fires rarely and matters enormously when it does.

### `composite_rank`

```
composite_rank = ½ (fidelity_rank + temporal_rank)
```

It is the equal-weighted mean of the two metric families, and it exists because
an unweighted mean over all metrics **was won by the shuffled-real control**:
1.24 against 2.47 for a genuine generator.

Nine of the nineteen per-market metrics are permutation-invariant, so they score
a shuffled deck perfectly by construction. Model selection uses `composite_rank`;
`avg_rank` is retained only for comparability and must never be the selection
criterion.

**The split did not solve the control problem.** With the control allowed to
compete it still wins the composite, 2.224 against 3.231 for the best generator
(S0, published draw). Only its exclusion from the ranks keeps it out. That
exclusion is legitimate — the control is the test block's own returns, so it has
oracle access to the test marginal, and fidelity accounts for 81–91% of its gap
to a fair i.i.d. baseline — but the exclusion, not the split, is what does the
work. Say so wherever the split is justified.

**As of 2026-09-21 (S4), `TEMPORAL_COLS` holds five metrics:** `acf_returns_mae`,
`acf_absolute_mae`, `acf_squared_mae`, `hurst_diff`, `resid_kurtosis_diff`.
`discriminative_auc_dist` and `discriminative_auc_absz` moved to
`DESCRIPTIVE_COLS` — computed and reported, never ranked. Twelve metrics are
ranked in total. Grounds in §6.

**The composite cannot resolve small differences.** The 95% bootstrap interval on
the first-second margin spans ~0.9 rank points over 15 market-seed units
(19,999 resamples), so a gap below ~0.45 is not a result. Two fresh generation
draws disagree on 53 of 75 within-unit positions; a change of ranking scheme
moves 6–13. Never report a winner without its interval.

---

## 3. Training budgets

**Budgets are specified in gradient steps, never epochs.**

An epoch is `floor(n_windows / batch_size)` steps — a data-dependent unit
disguised as a fixed one. Equal epoch counts across models or folds do not mean
equal training.

**Parity is on generator updates:**

```
TimeGAN.joint_steps == QuantGAN.train_steps == CNN-WGAN-GP.train_steps
```

The pipeline asserts this and refuses to run otherwise.

- TimeGAN's `ae_steps` and `sup_steps` are **pre-training** required by the
  four-phase algorithm (Yoon et al. 2019). They are reported separately and
  **excluded from parity**.
- QuantGAN and CNN-WGAN-GP take `n_critic = 5` discriminator updates per generator
  update. That is intrinsic to WGAN-GP (Gulrajani et al. 2017), not an extra
  budget.
- Equal generator updates is **not** equal compute. Measured wall-clock at 1,000
  generator steps, averaged over five runs: TimeGAN 20.5 s, CNN-WGAN-GP 42.4 s,
  QuantGAN 178.0 s — roughly a 9× spread. Record both budget and wall-clock;
  they are different claims. Run-to-run spread itself is uneven across models —
  see §6, Run-to-run nondeterminism.

All budgets, seeds, markets and fold counts live in the **EXPERIMENT
CONFIGURATION** cell. Do not set them anywhere else. Scattered hyperparameters
are what caused the 492× gap: a smoke-test override of `epochs` reached two
models and not the third.

**Per-market training, not cross-market pooling.** Each model trains on one
market's own data; the pipeline never concatenates markets into a shared
training series. Pooling was considered and rejected: concatenating all five
markets end-to-end adds only ~4 cross-market transitions across ~3,900
possible windows, which trades a small artefact for more training volume —
but a single generator fit across five independent markets risks each
market's dynamics contaminating the others' learned distribution, which
per-market training avoids entirely.

---

## 4. `SMOKE_TEST` must be `False` in any commit that reports real results — unguarded, check manually

The flag is in the EXPERIMENT CONFIGURATION cell. `verify_notebook.py` used to
check the **assignment**, line-anchored — an earlier substring check was
satisfied by the neighbouring comment text and passed while the flag was `True`.

As of 2026-09-05 that check has been deliberately removed, to allow committing a
smoke-test run as a checkpoint. There is now **no automated gate at all**:
`verify_notebook.py` passing says nothing about the value of `SMOKE_TEST`. Before
any commit meant to report real results, open the EXPERIMENT CONFIGURATION cell
and confirm `SMOKE_TEST = False` yourself.

Setting it `True` also changes seeds, markets, folds and step budgets, so an
unnoticed `True` silently records a configuration nobody intended to publish —
this is exactly how it was carried through eight commits previously (§1).

---

## 5. Report bugs, do not fix them silently

If a defect is found in any of:

- data, preprocessing, or the train/validation/test splits
- fold construction or walk-forward methodology
- model architectures, loss functions, or optimisers
- evaluation metrics or the generator-update definition
- normalisation or sequence construction

then **report it and stop**. Do not fix it as part of another task. These are
the scientific content of the thesis; a silent change to any of them invalidates
results that have already been reported.

Fixing is appropriate only when the defect blocks the requested work outright
(for example, a crash that prevents the pipeline running at all), and then it
must be stated prominently, not buried in a summary.

---

## 6. Project facts

**Data.** 5 BRICS markets (BOVESPA, FTSE, MOEX, NIFTY50, SHANGHAI), 20 years
(2006-08-30 → 2026-08-28), ~24,758 observations. Splits are temporal 80/10/10,
never shuffled. Test set ~496 per market, ~2,480 pooled.

**MOEX 2022.** Single-day return of −33.3% on 2022-02-24, then a 27-trading-day
suspension (2022-02-25 → 2022-03-24). Recorded as a known gap, not interpolated.

**Heavy tails.** Hill α = 2.54–3.21 across the five markets. Therefore
`E|X|^k < ∞` only for `k < α`:

- variance exists everywhere;
- skewness is infinite in 4 of 5 markets;
- **kurtosis is infinite everywhere.**

`kurtosis_diff` and `skewness_diff` are **descriptive-only** — computed and
displayed, never ranked. `tail_index_diff` is the ranked heavy-tail metric. Hill
returns `NaN` above α = 20, because a degenerate series drove it to 31,581.

**Metric taxonomy.** 7 fidelity (permutation-invariant, ranked) + 5 temporal
(ordering-sensitive, ranked) + 7 descriptive (computed, never ranked). Changed
2026-09-21 by S4 — see §2. Nineteen metrics appear in each per-market CSV;
`var_coverage_error` and `garch_persistence_diff` are computed elsewhere. NaN ranks last via
`na_option='bottom'`, so a failed metric counts as the worst outcome rather than
being silently dropped.

**The discriminative AUC null is not 0.5, is not one number, and no longer feeds
the ranking.** Per-market B=999 nulls: BOVESPA 0.549, FTSE 0.755, MOEX 0.616,
NIFTY50 0.557, SHANGHAI 0.569, with 97.5th percentiles 0.808 / 0.960 / 0.837 /
0.996 / 0.863. A stationary GARCH-t reference gives means 0.51–0.65 with spread
0.11–0.20 and p97.5 0.71–0.99 — the null is not 0.5 even under ideal conditions.

**0.506 ± 0.084 came from a different experiment**: independent stationary
samples, first recorded on 2026-08-30 as "15 seeds on identical distributions"
and relabelled as real-versus-real half-splits in `064f008` with the value
untouched. Its script is not in the repository. The estimator never changed and
never ran 15 replicates — `compute_auc_null` has used `n_rep=20` since
2026-08-29. Treat any surviving 0.506 as a quotation of that other experiment and
label it as such.

The legacy half-split scheme has a **left-anchored truncation defect**: past the
midpoint it compares the first n observations against the last n, leaving a gap
of 2·cut − N, in 121–125 of about 250 cuts per market. The approved construction
is the fixed-length adjacent sliding null, enumerated at every start position, at
both evaluation lengths. Note that a test-period null cannot be built: at the
evaluation length the test block holds 0 positions, 1 including validation.

Under contiguous folds and drift the classifier measures regime change rather
than real-versus-synthetic difference. Purged, embargoed CV (embargo 20 windows,
5–11% of training rows) confirms the drift is real rather than leakage: NIFTY50
0.375 contiguous → 0.352 purged, against 0.619 shuffled.

**0 of 25** market-model cells clear any calibrated threshold on fresh draws or at
mean + 1.96 sd; **1 of 25** clears p97.5 on the published draw (FTSE TimeGAN,
0.832 against 0.805). State both. Rank on `|AUC − 0.5|`, never on raw AUC:
ranking ascending on the raw value treats anti-predictive (0.30) as better than
indistinguishable (0.50). Cross-validation must not be shuffled; 20-day windows
overlap by 19 observations, and shuffling moves an independent-samples null from
about 0.50 to 0.58–0.60.

**Statistical power.** Kupiec at true p = 7% against a claimed 5%: 17% power at
n = 125, ~56% at n = 496, ~99% at n = 2,480. Per-market non-rejections are
uninformative; downstream utility is computed pooled (genuinely so since
2026-09-10 — see *Downstream-utility pooling* below). QLIKE inverts at
n ≈ 126 — Gaussian noise scored −6.888 against real data's −6.876 — and is stable
only at pooled n.

**Reference values.** TRTR QLIKE ≈ −8.134. Real-GARCH VaR coverage error 0.0259.
Wasserstein has a floor at the real data's MAD (0.00770 for BOVESPA), so a
collapsed constant-output model can beat a working one on that metric alone.

**Run-to-run nondeterminism.** At fixed seed and identical code, TimeGAN and the
shuffled control are bit-identical across five runs (sd = 0 on every metric,
including every walk-forward fold). CNN-WGAN-GP drifts slightly: Wasserstein CV 1.6%,
WF AUC sd 0.009. QuantGAN drifts substantially: Wasserstein CV 37%,
`tail_index_diff` CV 79%, raw AUC 0.551–0.734.

Those figures are retrains with the generation stream held fixed. Phase B's own
20-draw spread, same weights and fresh noise, is CV 0.68 for QuantGAN Wasserstein
and 0.79 for its `tail_index_diff` — use the Phase B figures when judging whether
a difference between fresh draws is meaningful.

Not a seeding bug — loss traces agree to four significant figures at epoch 1 and
separate by epoch 5. Consistent with QuantGAN being the only model combining
WGAN-GP double-backward with dilated convolutions.
`torch.use_deterministic_algorithms(True)` and `cudnn.deterministic` remain
deliberately unset; the cost is now measured rather than assumed.

`composite_rank` is stable (QuantGAN wins 5/5, 1.429–1.500; CNN-WGAN-GP 1.786–1.893;
TimeGAN 2.643–2.714). But `tail_index_diff`, `hurst_diff` and `mean_diff` change
their winner between runs. Never report a single-run winner on
`tail_index_diff` — QuantGAN spans 0.065–1.707 against CNN-WGAN-GP's 0.357–0.865.

That stability was measured at a 1,000-update smoke budget with the generation
stream fixed. It does not generalise: at 9,000 updates the ranking reverses, and
across fresh generation draws 53 of 75 within-unit positions move (§2).

**Non-stationary GARCH fits.** Maximum-likelihood GARCH can converge — `convergence_flag
= 0`, so the non-convergence guard never fires — to parameters on the
covariance-stationarity boundary, and then simulate explosive synthetic paths. Standard
convergence diagnostics do not detect this.

Found on SHANGHAI walk-forward fold 3 (`train[:3240]`, 3,240 points): the production
GARCH walk-forward CSVs give Wasserstein 74.795 / 7.814 / 43.006 for seeds 42 / 43 / 44,
against a run-wide median of 0.0025, with discriminative AUC 1.000 in all three. Refit on
that window (arch 8.0.0, local):

| | μ | ω | α | γ | β | ν | persistence |
|---|---|---|---|---|---|---|---|
| GARCH | 0.05389748 | 0.00674063 | 0.05453920 | — | 0.94546080 | 4.89916394 | 1 + 5.9e-14 |
| GJR-GARCH | 0.05391927 | 0.00673637 | 0.05461722 | −0.00020376 | 0.94548466 | 4.90039858 | 1 − 1.7e-13 |

Persistence is α + β, plus γ/2 for GJR (the negative-innovation term is active half the
time under a symmetric distribution).

- **A boundary effect, not a high-persistence effect.** Across 110 refits (5 markets ×
  full-train, 5-fold and 2-fold walk-forward windows × both models) 6 sit within 1.7e-13
  of 1; every other fit is at least 1.23e-4 below it (max 0.99987662, SHANGHAI full-train
  GJR). At persistence 0.9972 and 0.9995 (SHANGHAI folds 2 and 4), 0 of 1,200 simulated
  paths exceed 10× the training sd (max 7.76×).
- **At the boundary, explosion is a knife-edge.** In production, GARCH exploded on all
  three seeds while GJR stayed sane (Wasserstein 0.0011–0.0080). The local refit reverses
  it: GJR exceeds 10× the training sd on 200 of 200 seeds, GARCH on 0 of 200.
  Production's arch version was not recorded (`requirements.txt` says only `arch>=7.0`).
- **Not only fold 3.** SHANGHAI full-train GARCH is on the boundary too — the production
  `weights/GARCH_params.json` records α + β = 1.0000000000, matching the local refit to
  ~7 significant figures — and, locally, so is NIFTY50 5-fold walk-forward fold 0 GARCH.
- **Decision (2026-09-10): two-stage fitting, not raise-and-stop.** A first guard raised on
  boundary fits; it surfaced the problem but would have stopped every run at SHANGHAI
  Step 1 (confirmed on a smoke run). Now stage 1 fits unconstrained and *always* records
  persistence — a reported result, not a diagnostic. Stage 2 runs only when stage 1 is
  within `STATIONARITY_TOL = 1e-6` of 1: it refits under persistence ≤ 1 −
  `STATIONARITY_DELTA` (1e-4), and output is generated from the constrained parameters.
  Why: near-unit persistence in daily equity returns is structural breaks in the
  unconditional variance absorbed as persistence (Lamoureux & Lastrapes 1990; Mikosch &
  Stărică 2004) — SHANGHAI 2006–2026 spans the 2007 bubble and crash, 2015 and COVID — and
  at α + β = 1 the variance is a martingale with nothing to revert to, so paths
  random-walk without an anchor. Whether a boundary fit explodes is arbitrary (knife-edge
  above), so it is fixed at fit time rather than by rejecting draws after generation.
- **Why δ = 1e-4.** The bimodality above: boundary fits within 1.7e-13 of 1, every other
  fit at least 1.23e-4 below. 1 − 1e-4 is inside that empty gap, so the bound binds only on
  fits that were at the boundary.
- **How the constraint is imposed.** arch exposes no persistence bound, arch 8.0.0 has no
  variance-targeting GARCH, and rescaling parameters into `fix()` is not an MLE.
  `_StationaryGARCH` subclasses arch's `GARCH` and changes only the right-hand side of
  arch's own stationarity row (−Σα − ½Σγ − Σβ ≥ −1 becomes ≥ −(1 − δ)); arch's likelihood,
  SLSQP optimiser and `simulate()` are untouched, and it refuses to run if arch's
  constraint layout ever differs. The γ/2 weight is arch's own: both `constraints()` and
  `simulate()` use 0.5. Local refits: every boundary fit converges to persistence
  0.9999000000 at a log-likelihood change of −0.0006 to −0.0197; on stationary controls
  (BOVESPA full-train, SHANGHAI fold 4) the refit reproduces the unconstrained fit
  (ΔLL 0.0000, max parameter change 4.5e-4).
- **The generation guard can still fire.** `generate()` raises `GenerationSanityError`
  outside 1/10–10× the training sd, and the three generation call sites re-raise it. At
  persistence 0.9999 the pipeline's seeds 42–44 stay within 1.7× on every boundary window,
  but over 200 seeds one path per boundary window exceeded 10× (max 20.8×, SHANGHAI
  full-train GARCH). If it fires in a run, look before loosening it.
- **Recorded.** `<MODEL>_params.json` holds both parameter sets, both persistences and
  `generated_from`; `<MARKET>/seed<N>/garch_persistence.csv` has one row per econometric
  fit (main and every fold). `pipeline_run_metadata.json` records arch, statsmodels, scipy,
  numpy and pandas versions, because the knife-edge flipped between environments.
- **Pre-rerun production weights.** `weights/` was never tracked in git. Before
  `thesis_results/production/` was deleted on 2026-09-10, its 150 untracked files (22 MB:
  generator checkpoints and fitted-parameter JSONs, including the SHANGHAI file cited
  above) were archived to `../backups/thesis_results_production_untracked_33cc769.tar.gz`,
  outside the repository.

**Downstream-utility pooling.** `comprehensive_evaluation()` sees one market at a time,
so the per-seed `pooled_downstream_utility.csv` it used to write held that market's test
set alone (n 486–501) — below QLIKE's n ≳ 600 stability threshold, under a name
promising the pooled set. It now writes only `downstream_utility_inputs.npz` (the real
test series and one synthetic draw per model), and `compute_pooled_downstream_utility()`
concatenates those across every market for each seed, calls `compute_downstream_utility()`
once, and writes a single `pooled_downstream_utility.csv` at the results root. The metric
is unchanged. Concatenating five markets means one GARCH filter crosses four market boundaries within
~2,480 pooled observations — a disclosed artefact of pooling. Before pooling, the
shuffled control's QLIKE beat a genuine generator's in 33 of 75 control-versus-generator
comparisons (44%). QLIKE's
mean across cells is uninformative whenever one variance forecast is near zero — report
the median beside it.

**Known defect, not yet fixed in code (2026-09-21).** For the three GANs the series
in `downstream_utility_inputs.npz` is a second `generate()` call, not the draw the
19 metrics were computed on: 0 of 15 market-seed units agree per GAN, against 15 of
15 for GARCH and GJR-GARCH, which re-seed on every call. Fidelity and VaR/QLIKE
therefore describe different samples for the GANs. Published values are unchanged;
the correction is to feed the metrics draw to the downstream backtest.

**Two execution phases (2026-09-20).** Phase A trains and fits, writing weights and
`_meta.json` (RunPod, ~18 h, ~$14). Phase B loads weights, regenerates series and
recomputes metrics (local, CPU, free). Run Phase B with `phase_b_venv/bin/python`,
where pandas is pinned to 2.3.3, matching `requirements.txt`'s `>=2.0,<3.0`. A
pandas 2.3.3-versus-3.0.5 A/B over 20 draws × 6 series gave a maximum difference of
0.0 on every metric and rank column, so the pin is hygiene rather than a known risk.
`_meta.json` now carries `data_mean` and `data_std`; without them a loaded model
cannot denormalise, because `__init__` leaves them `None`.

**`compute_discriminative_score` scales inside each fold (2026-09-21).** It uses a
`Pipeline` of `StandardScaler` and `LogisticRegression`. The previous version fitted
the scaler on all rows before cutting folds, leaking test-fold statistics under every
CV scheme including purged. The fix moved null means by ≤0.008 but single AUC values
by up to 0.176 in strongly drifting stretches; no audit conclusion changed. All three
callers — the null, Track A and Track B — go through this one function.

**The AUC-null cache key includes `auc_scaler`.** Without it, nulls computed before
the scaler fix would be reused silently against post-fix AUCs.

**Not all published draws are reproducible.** GARCH, GJR-GARCH and the control
reproduce exactly; TimeGAN reproduces via CPU RNG replay (correlation 0.999998),
because it draws noise on the CPU; QuantGAN and CNN-WGAN-GP cannot, because their
CUDA RNG state was never saved. Walk-forward fold models were never saved at all, so
walk-forward AUC cannot be refreshed without retraining all 375 (~15 h, ~$11.50).

**Fair baselines read the training block only.** i.i.d. historical simulation and a
stationary block bootstrap, both scored by the same pipeline. Tune the block length
by Politis–White on **|r|**, not raw returns — raw returns lack linear
autocorrelation, so tuning on them collapses the block length to 0.3–5.4 and
degenerates to i.i.d. The correct version (~110–128 days) is indistinguishable from
GARCH and QuantGAN; TimeGAN ranks below plain i.i.d. resampling.

**Learned metrics do not rescue the audit.** C-FID is not estimable here: stride-1
windows share 127 of 128 observations, so 369 windows carry 3.6–4.4 independent
observations of a 100–320-dimensional distribution — a property of the windowing, not
the encoder. MMD on TS2Vec embeddings ranks the shuffled control lowest of six series
under all three encoder configurations and detects it in 0 of 15 units. Calibrated
against a scale-matched null, collapsed TimeGAN is detected in 0.7% of draws, down
from 37%. TS2Vec does carry ordering information — a linear probe separates real from
permuted at mean AUC 0.63, comparable to 0.67 from three ACF features — so the failure
belongs to the statistic at this sample size, not to the representation.

**Model naming.** The CNN-deconvolution WGAN-GP generator is **CNN-WGAN-GP** (class
`CNN_WGAN_GP`). Until 2026-09-10 it was called `FinGAN`, which collides with the published
Fin-GAN (Vuletić, Prenzel & Cucuringu 2024, *Quantitative Finance* 24(2), 175–199) — a
different model that forecasts and classifies returns with an economics-driven loss.
Result files written before the rename (all of `thesis_results/production/` at commit
`33cc769`) carry the old name; `report_data.py` discovers model names from the
filesystem, so it reads either.

---

## 7. Layout of the repository

| Path | Contents |
|---|---|
| `3_4_integrated_pipeline.ipynb` | the pipeline: models, metrics, plots, run loop |
| `verify_notebook.py` | regression guard; must exit 0 before any commit |
| `knowledge_base/` | 16-chapter LaTeX book; `make` builds `knowledge_base.pdf` |
| `thesis_results/production/` | real-run (`SMOKE_TEST=False`) data artifacts — **tracked** |
| `thesis_results/smoke/` | structural-check output — **gitignored, never tracked** |
| `report_data.py` | data layer for `generate_report.py`; reads `thesis_results/production/` by default |
| `generate_report.py` | builds the main PDF report; `--results-dir`/`--reports-dir` to point elsewhere |
| `Makefile` | `make report` / `report-smoke` / `verify` |
| `analysis/` | one directory per analysis round; scripts and outputs that are not pipeline artifacts |
| `vendor/ts2vec/` | official TS2Vec, unmodified, MIT licence, pinned commit — used by the embedding-distance audit |
| `embedding_distance.py`, `embedding_distance_summary.py` | MMD and C-FID on TS2Vec embeddings; writes outside the ranked metric set |
| `reports/production/` | shareable reports; main PDFs, metric diagnostics HTML, run metadata, run logs |
| `reports/smoke/` | same, for a smoke run — **gitignored**, split for the same reason as `thesis_results/smoke/` |

`thesis_results/<production\|smoke>/` both keep the same internal shape:
`<MARKET>/seed<N>/` (metrics,
`downstream_utility_inputs.npz`, `garch_persistence.csv`, `run_config.json`, `weights/` when not smoke-testing), `walk_forward/<MARKET>/` —
plus a single `pooled_downstream_utility.csv` at the root, pooled across markets
per seed.
Never mix the two roots by hand — `RESULTS`/`REPORTS` in the notebook and
`--results-dir`/`--reports-dir` in `generate_report.py`/`report_data.py` pick
one based on `SMOKE_TEST`, and nothing reads across the split.

**Resume.** `run_complete_pipeline()` skips a `(market, seed)` whose output
under the current `RESULTS` is already complete (metrics, downstream-utility
inputs, GARCH persistence,
every model's walk-forward CSV, and — outside smoke tests — every model's
weights, or for GARCH/GJR-GARCH the fitted-parameter JSON in the same
`weights/` directory) **and** whose `run_config.json` matches the run about to start
(`smoke_test`, `generator_updates`, `n_folds`, `seq_len`, TimeGAN pre-training
steps, the registered **model roster**, git commit). Any mismatch, or an absent `run_config.json`, forces a
re-run rather than a skip. `FORCE_RERUN = True` in EXPERIMENT CONFIGURATION
ignores all of this and redoes everything. The roster is recorded because ranks are
computed jointly across the models present: a five-model run and a six-model run
are not interchangeable at identical budgets, and output written before the field
existed (everything up to 2026-09-12) therefore reads as a config change. Aggregation (`overall_performance.csv`,
`per_seed_market_performance.csv`) is read back from the per-market-seed CSVs
on disk, not from in-memory state, so a run that skips everything still
produces complete aggregates. The pooled downstream-utility backtest is
recomputed the same way, from every market's `downstream_utility_inputs.npz`, and
refuses to run if any market for a seed is missing.

**Leftover files.** A rerun overwrites its own outputs but deletes nothing, so a results
tree can hold files from models that no longer exist (a renamed model's
`walk_forward_<OLD>_seed<N>.csv`, pre-2026-09-10 per-seed `pooled_downstream_utility.csv`).
`report_data.py` reads walk-forward files only for models present in that market-seed's
own metrics CSV, raises if one of those is missing, and lists every skipped file in the
`generate_report.py` output; it prefers the root pooled file over per-seed ones. Nothing
is deleted automatically.

**Notebook cell map** (indices shift when cells are inserted — re-check before
editing by index):

| Cell | Contents |
|---|---|
| 0 | imports, shared return bounds |
| 1 | paths (`ROOT`, `DATA`, `REPORTS`) — `RESULTS` and `REPORTS` both start here but are re-pointed to `smoke/`/`production/` in cell 3, once `SMOKE_TEST` is known |
| 3 | **EXPERIMENT CONFIGURATION** — seeds, markets, folds, step budgets, `SMOKE_TEST`, `FORCE_RERUN`, the `RESULTS` and `REPORTS` split |
| 4 | GPU Device Check — raises if CUDA is unavailable, does not fall back to CPU silently |
| 8 / 10 / 12 | TimeGAN / QuantGAN / CNN-WGAN-GP — gradient-trained (`family = 'gradient'`) |
| 14 | `GARCHModel` — GARCH(1,1) / GJR-GARCH(1,1), Student-t, fit by MLE (`family = 'econometric'`, exempt from the parity assertion); two-stage fit (constrained refit on boundary windows) and a raising post-generation scale guard (§6) |
| 16 | `FinancialMetrics` |
| 18 | plotting |
| 20 | pipeline, resume/skip logic, and `generate_metric_diagnostics_report` |
| 21 | main execution |

Main reports use LaTeX (`pdflatex`), never matplotlib, and are written to
`reports/report_YYYY-MM-DD.pdf` (or `report_YYYY-MM-DD_SMOKE.pdf`, stamped
with a banner and watermark, when `--allow-smoke` overrides the refusal that
`smoke_test: true` otherwise triggers). Pipeline-generated metric diagnostics
are written to `reports/metric_diagnostics/`, reproducibility metadata to
`reports/pipeline_run_metadata.json`, and full stdout+stderr to
`reports/run_log_<timestamp>.log`.