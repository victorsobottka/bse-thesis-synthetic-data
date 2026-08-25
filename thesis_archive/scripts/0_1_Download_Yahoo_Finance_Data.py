
import numpy as np
import yfinance as yf
import pandas as pd

# 1. Prepare Data

df_nvda = yf.download("NVDA", start="2019-12-31", end="2022-01-01")
df_nvda = df_nvda[["Open", "High", "Low", "Close", "Volume"]].round(2)

# 2. Calculate returns
df_nvda["Return"] = df_nvda["Close"].pct_change().round(4)  # Simple return
df_nvda["LogReturn"] = np.log(df_nvda["Close"] / df_nvda["Close"].shift(1)).round(4)  # Log return

# 3. Show sample
print("✅ Sample NVIDIA data with returns:")
print(df_nvda[["Close", "Return", "LogReturn"]].head(10))

# 4. Save to CSV
output_path = "./data/nvidia_2020_2021_returns.csv"
df_nvda.to_csv(output_path)
print(f"💾 File saved to: {output_path}")