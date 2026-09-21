"""Part 4 (incidental): does the series saved in downstream_utility_inputs.npz reproduce the recorded metrics?
comprehensive_evaluation calls model.generate() once for the metrics and a second time for the npz. Test: |skew(real) - skew(npz series)|
against the recorded skewness_diff, per market-seed and model (float32 storage -> agreement to ~1e-7 relative)."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import skew
ROOT = Path(__file__).resolve().parents[2]; P = ROOT / 'thesis_results/production'; OUT = P / 'overnight_2026-09-21'
rows = []
for m in ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']:
    for s in (42, 43, 44):
        o = np.load(P / m / f'seed{s}' / 'downstream_utility_inputs.npz'); real = o['__real__'].flatten()
        met = pd.read_csv(P / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model')
        for mod in ['GARCH', 'GJR-GARCH', 'TimeGAN', 'QuantGAN', 'CNN-WGAN-GP']:
            v = abs(skew(real) - skew(o[mod].flatten())); rec = met.loc[mod, 'skewness_diff']
            rows.append(dict(market=m, seed=s, model=mod, skewness_diff_from_npz=v, skewness_diff_recorded=rec, rel_gap=abs(v - rec) / max(abs(rec), 1e-12)))
D = pd.DataFrame(rows); D['agrees'] = D.rel_gap < 1e-3; D.to_csv(OUT / 'part4_npz_vs_recorded_metrics.csv', index=False)
print(D.groupby('model').agg(n=('agrees', 'size'), agree=('agrees', 'sum'), median_rel_gap=('rel_gap', 'median')).to_string())
print(D[(D.market == 'BOVESPA') & (D.seed == 42)].round(4).to_string(index=False))
