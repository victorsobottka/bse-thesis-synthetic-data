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

Nine of the nineteen originally ranked metrics were permutation-invariant, so
they scored a shuffled deck perfectly by construction. No weighting of the
remaining ten could overcome that. Model selection uses `composite_rank`;
`avg_rank` is retained only for comparability and must never be the selection
criterion.

---

## 3. Training budgets

**Budgets are specified in gradient steps, never epochs.**

An epoch is `floor(n_windows / batch_size)` steps — a data-dependent unit
disguised as a fixed one. Equal epoch counts across models or folds do not mean
equal training.

**Parity is on generator updates:**

```
TimeGAN.joint_steps == QuantGAN.train_steps == FinGAN.train_steps
```

The pipeline asserts this and refuses to run otherwise.

- TimeGAN's `ae_steps` and `sup_steps` are **pre-training** required by the
  four-phase algorithm (Yoon et al. 2019). They are reported separately and
  **excluded from parity**.
- QuantGAN and FinGAN take `n_critic = 5` discriminator updates per generator
  update. That is intrinsic to WGAN-GP (Gulrajani et al. 2017), not an extra
  budget.
- Equal generator updates is **not** equal compute. Measured wall-clock at 1,000
  generator steps, averaged over five runs: TimeGAN 20.5 s, FinGAN 42.4 s,
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

**Metric taxonomy.** 7 fidelity (permutation-invariant) + 7 temporal
(ordering-sensitive) + 7 descriptive (excluded from ranking). NaN ranks last via
`na_option='bottom'`, so a failed metric counts as the worst outcome rather than
being silently dropped.

**Discriminative AUC null is 0.506 ± 0.084**, measured over 15 real-vs-real
splits — not 0.5. Rank on `|AUC − 0.5|`, never on raw AUC: ranking ascending on
the raw value treats anti-predictive (0.30) as better than indistinguishable
(0.50). Cross-validation must not be shuffled; 20-day windows overlap by 19
observations, and shuffling moves the null from 0.506 to 0.584.

**Statistical power.** Kupiec at true p = 7% against a claimed 5%: 17% power at
n = 125, ~56% at n = 496, ~99% at n = 2,480. Per-market non-rejections are
uninformative; downstream utility is computed pooled. QLIKE inverts at
n ≈ 126 — Gaussian noise scored −6.888 against real data's −6.876 — and is stable
only at pooled n.

**Reference values.** TRTR QLIKE ≈ −8.134. Real-GARCH VaR coverage error 0.0259.
Wasserstein has a floor at the real data's MAD (0.00770 for BOVESPA), so a
collapsed constant-output model can beat a working one on that metric alone.

**Run-to-run nondeterminism.** At fixed seed and identical code, TimeGAN and the
shuffled control are bit-identical across five runs (sd = 0 on every metric,
including every walk-forward fold). FinGAN drifts slightly: Wasserstein CV 1.6%,
WF AUC sd 0.009. QuantGAN drifts substantially: Wasserstein CV 37%,
`tail_index_diff` CV 79%, raw AUC 0.551–0.734.

Not a seeding bug — loss traces agree to four significant figures at epoch 1 and
separate by epoch 5. Consistent with QuantGAN being the only model combining
WGAN-GP double-backward with dilated convolutions.
`torch.use_deterministic_algorithms(True)` and `cudnn.deterministic` remain
deliberately unset; the cost is now measured rather than assumed.

`composite_rank` is stable (QuantGAN wins 5/5, 1.429–1.500; FinGAN 1.786–1.893;
TimeGAN 2.643–2.714). But `tail_index_diff`, `hurst_diff` and `mean_diff` change
their winner between runs. Never report a single-run winner on
`tail_index_diff` — QuantGAN spans 0.065–1.707 against FinGAN's 0.357–0.865.

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
| `reports/production/` | shareable reports; main PDFs, metric diagnostics HTML, run metadata, run logs |
| `reports/smoke/` | same, for a smoke run — **gitignored**, split for the same reason as `thesis_results/smoke/` |

`thesis_results/<production\|smoke>/` both keep the same internal shape:
`<MARKET>/seed<N>/` (metrics, `pooled_downstream_utility.csv`,
`run_config.json`, `weights/` when not smoke-testing), `walk_forward/<MARKET>/`.
Never mix the two roots by hand — `RESULTS`/`REPORTS` in the notebook and
`--results-dir`/`--reports-dir` in `generate_report.py`/`report_data.py` pick
one based on `SMOKE_TEST`, and nothing reads across the split.

**Resume.** `run_complete_pipeline()` skips a `(market, seed)` whose output
under the current `RESULTS` is already complete (metrics, downstream utility,
every model's walk-forward CSV, and — outside smoke tests — every model's
weights, or for GARCH/GJR-GARCH the fitted-parameter JSON in the same
`weights/` directory) **and** whose `run_config.json` matches the run about to start
(`smoke_test`, `generator_updates`, `n_folds`, `seq_len`, TimeGAN pre-training
steps, git commit). Any mismatch, or an absent `run_config.json`, forces a
re-run rather than a skip. `FORCE_RERUN = True` in EXPERIMENT CONFIGURATION
ignores all of this and redoes everything. Aggregation (`overall_performance.csv`,
`per_seed_market_performance.csv`) is read back from the per-market-seed CSVs
on disk, not from in-memory state, so a run that skips everything still
produces complete aggregates.

**Notebook cell map** (indices shift when cells are inserted — re-check before
editing by index):

| Cell | Contents |
|---|---|
| 0 | imports, shared return bounds |
| 1 | paths (`ROOT`, `DATA`, `REPORTS`) — `RESULTS` and `REPORTS` both start here but are re-pointed to `smoke/`/`production/` in cell 3, once `SMOKE_TEST` is known |
| 3 | **EXPERIMENT CONFIGURATION** — seeds, markets, folds, step budgets, `SMOKE_TEST`, `FORCE_RERUN`, the `RESULTS` and `REPORTS` split |
| 4 | GPU Device Check — raises if CUDA is unavailable, does not fall back to CPU silently |
| 8 / 10 / 12 | TimeGAN / QuantGAN / FinGAN — gradient-trained (`family = 'gradient'`) |
| 14 | `GARCHModel` — GARCH(1,1) / GJR-GARCH(1,1), Student-t, fit by MLE (`family = 'econometric'`, exempt from the parity assertion) |
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
