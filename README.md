# GAN Benchmarking for BRICS Emerging-Market Log Returns

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active%20research-brightgreen)

Peer-reviewed paper benchmarking five generators of daily returns -- three GANs
(**TimeGAN**, **QuantGAN**, **CNN-WGAN-GP**) and two econometric baselines
(**GARCH(1,1)-t**, **GJR-GARCH(1,1)-t**) -- on five BRICS emerging-market log-return
series (Bovespa, FTSE JSE, MOEX, Nifty 50, Shanghai Composite), 2006-08-30 to
2026-08-28. Models are ranked on a metric suite that is audited against a shuffled
copy of the real data and against fair resampling baselines, and checked in
walk-forward validation.

The audit is a result in its own right: a large share of standard evaluation metrics
are permutation-invariant, the discriminative AUC null is per-market and far from 0.5,
and C-FID is not estimable on stride-1 financial windows. See the latest report in
`reports/production/` for the current findings and the "Known issues and what is not
resolved" section for what the benchmark does not yet establish.

Collaboration between **Barcelona School of Economics (BSE)** and **UPC**.

---

## Repository layout

```
.
├── 0_3_Optiver_datapreprocessing_BRICS_v0.ipynb   # Raw CSV → processed parquets (80/10/10 split)
├── 3_4_integrated_pipeline.ipynb                  # Main pipeline: train all GANs, evaluate, rank
├── generate_report.py                             # LaTeX report generator → reports/production/report_YYYY-MM-DD.pdf
├── report_data.py                                 # Data layer for generate_report.py (reads run artifacts)
├── verify_notebook.py                             # Regression guard; must exit 0 before every commit
├── Makefile                                       # make report / report-smoke / verify
├── 5_Paper_Calculate_LogReturns.py                # Standalone log-return computation utility
│
├── data/
│   ├── 20years/
│   │   ├── Bovespa Historical Data.csv            # Brazil (B3)
│   │   ├── FTSE_JSE All Share Historical Data.csv # South Africa (JSE)
│   │   ├── MOEX Russia Index Historical Data.csv  # Russia (MOEX)
│   │   ├── Nifty 50 Historical Data.csv           # India (NSE)
│   │   └── Shanghai Composite Historical Data.csv # China (SSE)
│   └── processed_files/
│       ├── BOVESPA_processed.csv                  # Log-return series (no clipping)
│       ├── FTSE_JSE_processed.csv
│       ├── MOEX_processed.csv
│       ├── NIFTY50_processed.csv
│       ├── SHANGHAI_processed.csv
│       ├── train/ valid/ test/                    # Parquet splits per market
│       └── preliminary_results/                   # Early-run BRICS metrics & plots
│
├── reports/                                       # Shareable reports and report inputs
│   ├── production/                                # Real-run reports -- tracked
│   │   ├── report_YYYY-MM-DD.pdf                  # Main date-stamped PDF report
│   │   ├── metric_diagnostics/                    # HTML metric/plot diagnostics from pipeline
│   │   │   └── index.html                         # Cross-market diagnostics entry point
│   │   ├── pipeline_run_metadata.json             # Run config, commit, seeds, library versions
│   │   └── run_log_<commit>.txt                   # Stdout of the production run
│   └── smoke/                                     # Smoke-run reports -- gitignored, not tracked
│
├── thesis_results/                                # Pipeline data artifacts: CSVs, metrics, plots
│   ├── production/                                # Real runs (SMOKE_TEST=False) -- tracked
│   └── smoke/                                     # Structural-check runs -- gitignored, not tracked
│
├── analysis/                                      # Per-round analysis scripts and outputs (not pipeline artifacts)
├── vendor/ts2vec/                                 # Official TS2Vec, unmodified, MIT, pinned commit
├── embedding_distance.py                          # MMD and C-FID on TS2Vec embeddings (audit, outside the ranked set)
├── embedding_distance_summary.py                  # Aggregates the embedding-distance outputs
│
├── papers/                                        # Reference papers for the BRICS paper (gitignored)
├── knowledge_base/                                # 16-chapter LaTeX book; `make` builds knowledge_base.pdf (gitignored)
│
└── thesis_archive/                                # Original BSE master thesis (NVIDIA, 2025)
    └── README.md                                  # See this file for thesis layout
```

---

## Models compared

| Model         | Architecture                 | Hyperparameters recorded for the production run |
|---------------|------------------------------|--------------------------------------------------|
| TimeGAN       | GRU autoencoder + GAN        | `hidden_dim=24`, `num_layers=3`, `lr=1e-3`, `batch_size=128`; 3,000 autoencoder + 3,000 supervisor pre-training steps, then 9,000 joint generator updates (Yoon 2019) |
| QuantGAN      | TCN WGAN-GP                  | `lr_g=lr_d=1e-4`, `batch_size=64`, `noise_dim=100`, `n_critic=5`, `λ_gp=10` (Gulrajani 2017); 9,000 generator updates |
| CNN-WGAN-GP   | CNN deconv WGAN-GP           | as QuantGAN; 9,000 generator updates |
| GARCH         | GARCH(1,1), Student-t        | maximum likelihood, `burn_in=500`; refit under a stationarity bound if the fit reaches persistence 1 |
| GJR-GARCH     | GJR-GARCH(1,1), Student-t    | as GARCH, with one leverage term (`o=1`) |

Values are those in `thesis_results/production/*/seed*/run_config.json`
(`generator_updates=9000`, `timegan_ae_steps=3000`, `timegan_sup_steps=3000`,
`n_folds=5`, `seq_len=128`) and `reports/production/pipeline_run_metadata.json`
(`models_config`). Training budgets are in **generator updates, never epochs** (an
epoch is a data-dependent number of steps). `TimeGAN.joint_steps ==
QuantGAN.train_steps == CNN-WGAN-GP.train_steps` is asserted; the value lives in the
EXPERIMENT CONFIGURATION cell of the pipeline notebook and nowhere else. The GARCH
models are exempt: maximum likelihood has no gradient-step analogue.

---

## Evaluation metrics

| Family      | Metrics | Role |
|-------------|---------|------|
| Fidelity (7)   | mean, sd, Wasserstein-1, energy distance, quantile MSE, tail index (Hill), extreme-event frequency | ranked; permutation-invariant |
| Temporal (5)   | ACF of returns / absolute returns / squared returns, Hurst exponent, residual kurtosis | ranked; order-sensitive |
| Descriptive (7)| skewness, kurtosis, ARCH-LM p-value and statistic, discriminative AUC (`\|AUC − 0.5\|` and its z-score against the empirical null), raw AUC, VaR coverage error, GARCH persistence difference | reported, never ranked |

Models are ranked on `composite_rank = ½ (fidelity_rank + temporal_rank)`. A shuffled
copy of the real data is scored by the same code as a control and is not ranked: it has
oracle access to the test marginal, and would win the composite if allowed to compete.
Fair baselines that read the **training block only** — i.i.d. historical simulation and a
stationary block bootstrap — are scored by the same pipeline and do not win.

The two discriminative AUC columns left the ranked set on 2026-09-21. No reference for
them works under any calibration tested, and their value is determined by which way the
market drifted rather than by generator quality. They are still computed and reported.

**The composite cannot resolve small differences.** The 95% bootstrap interval on the
first-second margin spans about 0.9 rank points over 15 market-seed units, and two fresh
generation draws disagree on 53 of 75 within-unit positions. Report intervals, not winners.

Walk-forward validation is a separate track: 5 expanding folds on the **full** series
(fold *k* trains on the first n − L·(5 − *k*) observations and is tested on the next
L = n // 6), every model retrained from fresh initialisation in each fold. It reports
per-fold diagnostics and never enters the ranking.

---

## Quickstart

```bash
# 1. Pre-process raw CSVs (once)
jupyter nbconvert --to notebook --execute 0_3_Optiver_datapreprocessing_BRICS_v0.ipynb

# 2. Run the full pipeline (trains all five models, evaluates, ranks)
jupyter nbconvert --to notebook --execute 3_4_integrated_pipeline.ipynb
#    → thesis_results/{smoke,production}/*.csv, thesis_results/{smoke,production}/*/*.png
#      (which one depends on SMOKE_TEST in the EXPERIMENT CONFIGURATION cell)
#    → reports/{smoke,production}/metric_diagnostics/index.html
#    → reports/{smoke,production}/pipeline_run_metadata.json

# 3. Generate the PDF report (make report also works)
python generate_report.py        # → reports/production/report_YYYY-MM-DD.pdf

# Before any commit
python verify_notebook.py        # must exit 0
```

Requires: Python 3.10+, PyTorch (CUDA -- the pipeline refuses to fall back to CPU), arch, statsmodels, scikit-learn, pdflatex (TeX Live). See `requirements.txt`.

---

## Data notes

- **No winsorization/clipping** applied to any return series — deliberate methodological
  decision (Adams et al. 2019) to preserve EVT-relevant tail behaviour.
- Data sources: Investing.com historical downloads (daily closing prices).
- Log-return formula: `r_t = ln(P_t / P_{t-1})`.
- Static 80 / 10 / 10 train-validation-test split applied during preprocessing.

---

## Related work (thesis)

The original BSE master thesis (2025) used **NVIDIA (NVDA)** daily log-returns and
additionally explored LLM-based generation (zero-shot, few-shot, DeepSeek fine-tuning).
All thesis artefacts are preserved in [`thesis_archive/`](thesis_archive/README.md).

---

## License

MIT — see [LICENSE](LICENSE).