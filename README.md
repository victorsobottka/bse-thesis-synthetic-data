# Synthetic Financial Time Series Generation


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This repository provides code and notebooks for generating synthetic financial time series (e.g., NVIDIA log returns) using both Large Language Models (LLMs) and Generative Adversarial Networks (GANs).  
It includes data collection, exploratory analysis, LLM-based generation (zero-shot, few-shot, fine-tuning), GAN-based generation, and statistical comparison of real vs. synthetic data.


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen)

This repository provides code and notebooks for generating synthetic financial time series (e.g., NVIDIA log returns) using both Large Language Models (LLMs) and Generative Adversarial Networks (GANs).  
It includes data collection, exploratory analysis, LLM-based generation (zero-shot, few-shot, fine-tuning), GAN-based generation, and statistical comparison of real vs. synthetic data.

---

## 📁 Project Structure

- **data/**
  - `nvidia_2020_2021_returns.csv` – Downloaded/processed dataset
- `0_1_Download_Yahoo_Finance_Data.py` – Script to fetch Yahoo Finance data
- `0_2_Data_Exploration.ipynb` – Exploratory Data Analysis (EDA)
- `2_1_LLM_Zero_Shot.ipynb` – Zero-shot generation with LLMs
- `2_2_LLM_Few_Shot.ipynb` – Few-shot generation with LLMs
- `2_3_1_LLM_Fine_Tunning_Model_Trainning.ipynb` – Fine-tuning LLMs (Training)
- `2_3_2_LLM_Fine_Tunning_Model_Inference.ipynb` – Fine-tuning LLMs (Inference)
- `3_1_Time_GANs.ipynb` – TimeGAN synthetic data generation
- `3_2_Quant_GAN.ipynb` – QuantGAN synthetic data generation
- `3_3_Fin_GAN.ipynb` – FinGAN synthetic data generation
- `4_Model_Comparative.ipynb` – Statistical comparison notebook


### Explore Data
Open the Jupyter notebook for EDA:  
`0_2_Data_Exploration.ipynb`

---

### Synthetic Data Generation

- **LLMs:**

    - **LLM Zero-Shot:**  
  `2_1_LLM_Zero_Shot.ipynb`

    - **LLM Few-Shot:**  
  `2_2_LLM_Few_Shot.ipynb`

    - **LLM Fine-Tuning (Training):**  
  `2_3_1_LLM_Fine_Tunning_Model_Trainning.ipynb`

    - **LLM Fine-Tuning (Inference):**  
  `2_3_2_LLM_Fine_Tunning_Model_Inference.ipynb`

- **GANs:**
    - **TimeGAN:** `3_1_Time_GANs.ipynb`
    - **QuantGAN:** `3_2_Quant_GAN.ipynb`
    - **FinGAN:** `3_3_Fin_GAN.ipynb`

---

### Model Comparison
Compare the statistical properties of real and synthetic series:  
`4_Model_Comparative.ipynb`
