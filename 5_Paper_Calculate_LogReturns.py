"""
Preprocess 20-year BRICS price CSVs into log-return CSVs.

Source: data/20years/  (investing.com exports, MM/DD/YYYY date format)
Output: data/processed_files/<MARKET>_processed.csv

Markets (genuine BRICS basket):
  BOVESPA   Brazil
  FTSE_JSE  South Africa
  MOEX      Russia  ← replaces MSCI (a global index, not a BRICS market)
  NIFTY50   India
  SHANGHAI  China

Integrity gate rationale
────────────────────────
The gate uses MAX_ROUTINE_GAP = 12 to allow documented legitimate closures:
  MOEX     : Russian New Year (Jan 1-12, up to 12 days) — six occurrences
  SHANGHAI : Chinese New Year / Golden Week (≤ 11 days)  — four occurrences

One additional known structural break is explicitly allowed:
  MOEX 2026-02-25 → 2026-03-24 : 27-day trading suspension post-invasion.
  This is NOT a data error. It is a genuine market event documented in
  financial history. Never interpolate across it.
"""

import pandas as pd
import numpy as np
import os

# ── Source mapping ────────────────────────────────────────────────────────────
files = {
    "BOVESPA":  "Bovespa Historical Data.csv",
    "FTSE_JSE": "FTSE_JSE All Share Historical Data.csv",
    "MOEX":     "MOEX Russia Index Historical Data.csv",
    "NIFTY50":  "Nifty 50 Historical Data.csv",
    "SHANGHAI": "Shanghai Composite Historical Data.csv",
}

data_dir     = "./data/20years"
processed_dir = "./data/processed_files"
os.makedirs(processed_dir, exist_ok=True)

# ── Integrity gate constants ──────────────────────────────────────────────────
MIN_YEARS      = 15
MIN_ROWS       = 3_500
MAX_ROUTINE_GAP = 12   # covers Russian New Year (12d) and Chinese New Year (11d)

# Gaps above MAX_ROUTINE_GAP that are known market events, not data errors.
# Key = market; value = list of (gap_start, gap_end, explanation).
KNOWN_CLOSURES: dict[str, list[tuple[str, str, str]]] = {
    "MOEX": [
        ("2022-02-25", "2022-03-24",
         "post-invasion trading suspension — 27 calendar days (invasion: 2022-02-24)"),
    ],
}

# ── Process each market ───────────────────────────────────────────────────────
for key, filename in files.items():
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"WARNING: {filename} not found in {data_dir} — skipping {key}")
        continue

    df = pd.read_csv(path)

    # Price column: strip commas and currency symbols
    df["Price"] = (
        df["Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    # Date parse: source uses MM/DD/YYYY; lexicographic sort is wrong across
    # year boundaries (e.g. "01/01/2021" < "12/31/2020" alphabetically).
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False)
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Date", "Price"])

    # ── Integrity gate ────────────────────────────────────────────────────────
    span_years = (df["Date"].max() - df["Date"].min()).days / 365.25

    if len(df) < MIN_ROWS or span_years < MIN_YEARS:
        raise ValueError(
            f"{key}: {len(df)} rows spanning {span_years:.1f} years. "
            f"Expected >={MIN_ROWS} rows over >={MIN_YEARS} years. "
            "Check the date-range selector on the investing.com export."
        )

    gaps = df["Date"].diff().dt.days
    big_gap_mask = gaps > MAX_ROUTINE_GAP
    big_gap_dates = df.loc[big_gap_mask, "Date"].tolist()

    allowed_ends = {
        pd.Timestamp(end)
        for (_, end, _) in KNOWN_CLOSURES.get(key, [])
    }
    unexplained = [d for d in big_gap_dates if d not in allowed_ends]

    # Log every large gap for audit transparency
    if big_gap_dates:
        print(f"\n{key}: gaps > {MAX_ROUTINE_GAP} calendar days:")
        for d in big_gap_dates:
            days = int(gaps[df["Date"] == d].iloc[0])
            label = "(KNOWN)" if d in allowed_ends else "(** CHECK **)"
            print(f"  gap of {days:3d} days ending {d.date()} {label}")
        # Log known closure details
        for start, end, note in KNOWN_CLOSURES.get(key, []):
            print(f"  [{start} → {end}]: {note}")

    if unexplained:
        raise ValueError(
            f"{key}: unexplained gap(s) ending {[d.date() for d in unexplained]}. "
            "If this is a legitimate closure, add it to KNOWN_CLOSURES."
        )

    # ── Log returns ───────────────────────────────────────────────────────────
    df["LogReturn"] = np.log(df["Price"] / df["Price"].shift(1))

    # Save
    out_path = os.path.join(processed_dir, f"{key}_processed.csv")
    df.to_csv(out_path, index=False)

    print(f"\n✅ {key}: {len(df)} rows | "
          f"{df['Date'].min().date()} → {df['Date'].max().date()} | "
          f"vol={df['LogReturn'].std()*100:.2f}% | "
          f"kurt={df['LogReturn'].dropna().kurtosis():.2f}")
    print(f"   Saved: {out_path}")

print("\nAll markets processed.")
