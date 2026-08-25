# GAN Benchmarking for BRICS Emerging-Market Log Returns

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active%20research-brightgreen)

Peer-reviewed paper benchmarking **TimeGAN**, **QuantGAN**, and **FinGAN** on five
BRICS emerging-market log-return series (Bovespa, FTSE JSE, MSCI, Nifty 50,
Shanghai Composite) using stylized-fact preservation, walk-forward validation, and
distributional metrics (Wasserstein distance, energy distance, discriminative AUC,
ARCH-LM, Hurst exponent).

Collaboration between **Barcelona School of Economics (BSE)** and **UPC**.

---

## Repository layout

```
.
├── 0_3_Optiver_datapreprocessing_BRICS_v0.ipynb   # Raw CSV → processed parquets (80/10/10 split)
├── 3_4_integrated_pipeline.ipynb                  # Main pipeline: train all GANs, evaluate, rank
├── generate_report.py                             # LaTeX report generator → reports/report_YYYY-MM-DD.pdf
├── 5_Paper_Calculate_LogReturns.py                # Standalone log-return computation utility
│
├── data/
│   ├── Bovespa Historical Data.csv                # Brazil (B3)
│   ├── FTSE_JSE All Share Historical Data.csv     # South Africa (JSE)
│   ├── MSCI Stock Price History.csv               # MSCI EM index
│   ├── Nifty 50 Historical Data.csv               # India (NSE)
│   ├── Shanghai Composite Historical Data.csv     # China (SSE)
│   └── processed_files/
│       ├── BOVESPA_processed.csv                  # Log-return series (no clipping)
│       ├── FTSE_JSE_processed.csv
│       ├── MSCI_processed.csv
│       ├── NIFTY50_processed.csv
│       ├── SHANGHAI_processed.csv
│       ├── train/ valid/ test/                    # Parquet splits per market
│       └── preliminary_results/                   # Early-run BRICS metrics & plots
│
├── reports/                                       # Generated PDF reports (pdflatex)
│   └── report_2026-08-20.pdf                      # Latest — 30 pages, clickable references
│
├── papers/                                        # Reference papers for the BRICS paper
│
└── thesis_archive/                                # Original BSE master thesis (NVIDIA, 2025)
    └── README.md                                  # See this file for thesis layout
```

---

## Models compared

| Model    | Architecture          | Key hyperparameters (literature-anchored) |
|----------|-----------------------|-------------------------------------------|
| TimeGAN  | GRU autoencoder + GAN | `n_critic=1` (standard GAN), `epochs=200` |
| QuantGAN | TCN WGAN-GP           | `n_critic=5`, `λ_gp=10` (Gulrajani 2017) |
| FinGAN   | CNN deconv WGAN-GP    | `n_critic=5`, `λ_gp=10` (Gulrajani 2017) |

---

## Evaluation metrics

| Category          | Metric                             |
|-------------------|------------------------------------|
| Distribution      | Wasserstein-1, energy distance, quantile MSE |
| Moments           | Kurtosis diff, skewness diff       |
| Temporal          | ACF (raw + absolute), partial ACF  |
| Heavy tails       | Tail index (Hill estimator)        |
| Volatility        | ARCH-LM p-value difference         |
| Long memory       | Hurst exponent on `\|returns\|`    |
| Discriminability  | Discriminative AUC (CTBench protocol) |

Walk-forward validation: 5 rolling folds on the test set, model retrained each fold,
reporting mean ± std across folds.

---

## Quickstart

```bash
# 1. Pre-process raw CSVs (once)
jupyter nbconvert --to notebook --execute 0_3_Optiver_datapreprocessing_BRICS_v0.ipynb

# 2. Run the full pipeline (trains all three GANs, evaluates, ranks)
jupyter nbconvert --to notebook --execute 3_4_integrated_pipeline.ipynb

# 3. Generate the PDF report
python generate_report.py        # → reports/report_YYYY-MM-DD.pdf
```

Requires: Python 3.10+, PyTorch, statsmodels, scikit-learn, pdflatex (TeX Live).

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
