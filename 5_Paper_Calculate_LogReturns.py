import pandas as pd
import numpy as np
import os

files = {
    "BOVESPA": "Bovespa Historical Data.csv",
    "FTSE_JSE": "FTSE_JSE All Share Historical Data.csv",
    "MSCI": "MSCI Stock Price History.csv",
    "NIFTY50": "Nifty 50 Historical Data.csv",
    "SHANGHAI": "Shanghai Composite Historical Data.csv"
}

data_dir = "./data"

for key, filename in files.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"⚠️ No encontrado: {filename}")
        continue

    df = pd.read_csv(path)

    # Usa "Price" como columna de cierre
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("$", "")
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Ordenar por fecha
    df = df.sort_values("Date")

    # Calcular log return
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))

    print(f"\n✅ {key} - primeras filas:")
    print(df[["Date", "Price", "LogReturn"]].head(5))

    # Guardar en el folder processed_files  
    processed_dir = "./data/processed_files"
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"{key}_processed.csv")
    df.to_csv(processed_path, index=False)
    print(f"Archivo procesado guardado en: {processed_path}")