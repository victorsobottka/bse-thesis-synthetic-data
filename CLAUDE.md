# CLAUDE.md — standing constraints for this repository

BSE Master Thesis: benchmarking GANs for synthetic financial time series.
Read this before editing anything.

---

## 1. Workflow

### Close the notebook tab in VS Code before editing `.ipynb`

VS Code holds an in-memory copy of an open notebook. Its next save writes that
copy to disk, silently discarding any edit made outside the editor. **This has
destroyed work three times.**

The failure is quiet and it survives verification. Commit `ab37aad` contains a
hardened `SMOKE_TEST` check *and* `SMOKE_TEST = True`, so `verify_notebook.py`
fails on that commit — even though it passed 34/34 in the moments before the
`git add`. The file changed in between. Nothing errored.

Before any programmatic edit to `3_4_integrated_pipeline.ipynb`:

1. Close the notebook tab in VS Code.
2. Make the edit.
3. Run `python verify_notebook.py`.
4. `git add` and commit.
5. Re-open the tab afterwards if needed.

If the tab cannot be closed, say so and stop rather than editing anyway.

### `verify_notebook.py` must exit 0 before every commit

```bash
python verify_notebook.py   # must exit 0
```

A FAIL is a regression, not a warning. Do not commit through one. The script
guards against exactly the silent-overwrite failure above, and it only helps if
it is run immediately before staging — not several steps earlier.

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
  generator steps: TimeGAN 19.7 s, FinGAN 39.3 s, QuantGAN 180.1 s — a 9× spread.
  Record both budget and wall-clock; they are different claims.

All budgets, seeds, markets and fold counts live in the **EXPERIMENT
CONFIGURATION** cell. Do not set them anywhere else. Scattered hyperparameters
are what caused the 492× gap: a smoke-test override of `epochs` reached two
models and not the third.

---

## 4. `SMOKE_TEST` must be `False` in any commit

The flag is in the EXPERIMENT CONFIGURATION cell. `verify_notebook.py` checks the
**assignment**, line-anchored — an earlier substring check was satisfied by the
neighbouring comment text and passed while the flag was `True`.

Setting it `True` also changes seeds, markets, folds and step budgets, so a
commit carrying `True` silently records a configuration nobody intended to
publish.

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

---

## 7. Layout of the repository

| Path | Contents |
|---|---|
| `3_4_integrated_pipeline.ipynb` | the pipeline: models, metrics, plots, run loop |
| `verify_notebook.py` | regression guard; must exit 0 before any commit |
| `knowledge_base/` | 16-chapter LaTeX book; `make` builds `knowledge_base.pdf` |
| `thesis_results/` | outputs; `manifest.json` records commit, seed, device |

**Notebook cell map** (indices shift when cells are inserted — re-check before
editing by index):

| Cell | Contents |
|---|---|
| 0 | imports, shared return bounds |
| 1 | paths (`ROOT`, `DATA`, `RESULTS`) |
| 2 | **EXPERIMENT CONFIGURATION** — seeds, markets, folds, step budgets, `SMOKE_TEST` |
| 7 / 9 / 11 | TimeGAN / QuantGAN / FinGAN |
| 13 | `FinancialMetrics` |
| 15 | plotting |
| 17 | pipeline and `generate_results_report` |
| 18 | main execution |

Reports use LaTeX (`pdflatex`), never matplotlib, and are written to
`reports/report_YYYY-MM-DD.pdf`.
