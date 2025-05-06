
import numpy as np
import yfinance as yf
import pandas as pd

# 1. Prepare Data

df_nvda = yf.download("NVDA", start="2020-01-01", end="2022-01-01")
df_nvda = df_nvda[["Open", "High", "Low", "Close", "Volume"]].round(2)

# Show first few rows
print("✅ Sample NVIDIA data:")
print(df_nvda.head(10))

# Step 4: Save to CSV
output_path = "./data/nvidia_2020_2021.csv"
df_nvda.to_csv(output_path)
print(f"💾 File saved to: {output_path}")