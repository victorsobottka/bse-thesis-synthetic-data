# Copilot Instructions for Synthetic Financial Time Series Generation

## Project Overview
This repository generates synthetic financial time series using Large Language Models (LLMs) and Generative Adversarial Networks (GANs). It covers:
- Data collection (Yahoo Finance, CSVs in `data/`)
- Exploratory analysis (`0_2_Data_Exploration.ipynb`)
- Synthetic data generation (LLMs: zero-shot, few-shot, fine-tuning; GANs: TimeGAN, QuantGAN, FinGAN)
- Statistical comparison of real vs. synthetic data (`4_Model_Comparative.ipynb`)

## Key Directories & Files
- `data/`: Source and processed datasets (CSV format)
- `0_1_Download_Yahoo_Finance_Data.py`: Download stock data using yfinance
- `Paper_Calculate_Log_Return.py`: Calculate log returns for financial series
- `2_1_LLM_Zero_Shot.ipynb`, `2_2_LLM_Few_Shot.ipynb`, `2_3_1_LLM_Fine_Tunning_Model_Trainning.ipynb`, `2_3_2_LLM_Fine_Tunning_Model_Inference.ipynb`: LLM-based synthetic data workflows
- `3_1_Time_GANs.ipynb`, `3_2_Quant_GAN.ipynb`, `3_3_Fin_GAN.ipynb`: GAN-based synthetic data workflows
- `4_Model_Comparative.ipynb`: Compare statistical properties of datasets

## Developer Workflows
- **Data Download:** Use `0_1_Download_Yahoo_Finance_Data.py` or yfinance in notebooks/scripts
- **Data Exploration:** Use Jupyter notebooks for EDA and visualization
- **Synthetic Data Generation:** Run LLM or GAN notebooks for generation tasks
- **Model Training:** Fine-tune LLMs or train GANs in respective notebooks
- **Comparison:** Use `4_Model_Comparative.ipynb` for statistical analysis

## Conventions & Patterns
- All data files are stored in `data/` and referenced by scripts/notebooks
- Notebooks are numbered by workflow stage (0: data, 2: LLM, 3: GAN, 4: comparison)
- Use pandas for data manipulation, numpy for calculations, matplotlib/seaborn for plots
- Model outputs and results are saved in dedicated folders (`LLMs_results/`, `TimeGAN_results/`, etc.)
- Scripts and notebooks are designed for reproducibility; outputs are saved to disk

## External Dependencies
- Python 3.8+
- yfinance, pandas, numpy, matplotlib, seaborn
- Jupyter for interactive workflows

## Example Patterns
- To calculate log returns:
  ```python
  df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1)).round(4)
  ```
- To download data:
  ```python
  import yfinance as yf
  df = yf.download("NVDA", start="2019-12-31", end="2022-01-01")
  ```

## Integration Points
- All synthetic data generation (LLM/GAN) workflows depend on preprocessed CSVs in `data/`
- Model results are saved for downstream analysis and comparison

---

For questions about workflow, file naming, or integration, see the README or relevant notebook for examples.
