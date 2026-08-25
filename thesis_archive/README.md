# Thesis Archive — Original BSE Master Thesis (NVIDIA, 2025)

This directory preserves the artefacts from the original **BSE Master Thesis** (2025),
which benchmarked TimeGAN, QuantGAN, FinGAN, and LLM-based approaches on
**NVIDIA (NVDA) daily log-returns** (2020–2021).

The thesis subsequently evolved into a peer-reviewed paper using **BRICS
emerging-market data** (Bovespa, FTSE JSE, MSCI, Nifty 50, Shanghai Composite).
All current/active work lives in the project root.

---

## Directory layout

```
thesis_archive/
├── data/                  Raw NVIDIA price and return CSVs
├── notebooks/             All original Jupyter notebooks
│   ├── 0_2_*              NVIDIA data exploration
│   ├── 2_1_* – 2_3_*     LLM zero-shot, few-shot, fine-tuning experiments
│   ├── 3_1_* – 3_4_*     TimeGAN, QuantGAN, FinGAN, combined GANs notebook
│   └── 4_*                Original model comparison
├── scripts/               Helper scripts (Yahoo Finance downloader)
├── results/
│   ├── timegan/           TimeGAN NVIDIA output plots
│   ├── quantgan/          QuantGAN NVIDIA output plots
│   ├── fingan/            FinGAN NVIDIA output plots
│   ├── llm/               Zero-shot / few-shot / fine-tuning outputs
│   │   ├── inputs/        NVIDIA CSVs fed to the LLMs
│   │   └── outputs/       Generated returns and analysis images
│   └── synthetic_vs_real.csv   Early metric comparison table
├── models/
│   ├── deepseek_lora/     DeepSeek-R1 LoRA adapter fine-tuned on NVIDIA log-returns
│   └── gan_weights/       Saved TimeGAN component weights (.pth)
└── images/                Plots and diagrams produced during the thesis
```

## Key findings (thesis)

| Method    | Dataset          | Best metric     | Note                                 |
|-----------|------------------|-----------------|--------------------------------------|
| TimeGAN   | NVIDIA 2020–2021 | Wasserstein ↓   | 5 epochs — underfit; needs ≥500      |
| QuantGAN  | NVIDIA 2020–2021 | ACF MAE ↓       | Best stylized-fact reproduction      |
| FinGAN    | NVIDIA 2020–2021 | Kurtosis diff ↓ | Good tail fidelity                   |
| DeepSeek  | NVIDIA 2020–2021 | —               | LLM zero-/few-shot: poor calibration |

The thesis concluded that TCN-based QuantGAN best preserves stylized facts for
single-asset daily log-returns, and that LLMs in zero-/few-shot mode are not yet
competitive with purpose-built GAN architectures for financial time series.
