"""Every generator against each fair baseline, S0 and S4: unit composites from ranks_by_draw.csv.gz (frame A: iid and sb_r; frame B: the |r|-tuned block bootstrap sb_abs),
margin = composite(generator) - composite(baseline) (negative = generator ahead), mean over the 15 units, bootstrap over units (19,999, default_rng(20260922)). Output: generator_vs_baseline_margins.csv."""
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; O = ROOT / 'thesis_results/production/baseline_generators'
R = pd.read_csv(O / 'ranks_by_draw.csv.gz'); GENS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; UN = R[['market', 'seed']].drop_duplicates().itertuples(index=False, name=None)
UN = sorted(set(UN)); B = 19999; rng = np.random.default_rng(20260922); rows = []
for fr, bases in (('A', ['iid', 'sb_r']), ('B', ['sb_abs'])):
    for sc in ('S0', 'S4'):
        g = R[(R.frame == fr) & (R.scenario == sc)].groupby(['market', 'seed', 'entry']).composite.mean().unstack('entry').loc[UN]
        idx = rng.integers(0, len(UN), (B, len(UN)))
        for b in bases:
            for gen in GENS:
                d = (g[gen] - g[b]).to_numpy(); bs = d[idx].mean(1); lo, hi = np.percentile(bs, [2.5, 97.5])
                rows.append(dict(frame=fr, scenario=sc, generator=gen, baseline=b, margin_generator_minus_baseline=float(d.mean()), ci_lo=float(lo), ci_hi=float(hi), generator_ahead_significant=bool(hi < 0), baseline_ahead_significant=bool(lo > 0), units_generator_ahead=int((d < 0).sum()), n_units=len(UN)))
D = pd.DataFrame(rows); D.to_csv(O / 'generator_vs_baseline_margins.csv', index=False); pd.set_option('display.width', 200); print(D.round(3).to_string(index=False))
