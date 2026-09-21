"""Small supporting numbers for the report: (1) within-unit draw-to-draw sd of acf_absolute_mae per entry; (2) bootstrap of sb_abs vs GARCH / QuantGAN / control-free pairs (frame B, 15 units, B=19,999)."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'task11_2026-09-21'))
from harness import *
O = RES / 'baseline_generators'; D = pd.read_csv(O / 'metrics_all_entries_by_draw.csv')
sd = D.groupby(['market', 'seed', 'model']).acf_absolute_mae.std(ddof=1).groupby('model').mean(); print('within-unit sd over 20 draws of acf_absolute_mae:'); print(sd.round(4).to_string())
R = pd.read_csv(O / 'ranks_by_draw.csv.gz'); UNITS = [(m, s) for m in MARKETS for s in (42, 43, 44)]; rng = np.random.default_rng(7); B = 19999; out = {}
for sc in ('S0', 'S4'):
    g = R[(R.frame == 'B') & (R.scenario == sc)].groupby(['market', 'seed', 'entry']).composite.mean().unstack('entry').loc[UNITS]
    for other in ('GARCH', 'QuantGAN'):
        d = (g['sb_abs'] - g[other]).to_numpy(); idx = rng.integers(0, 15, (B, 15)); b = d[idx].mean(1); lo, hi = np.percentile(b, [2.5, 97.5]); out[f'{sc}: sb_abs - {other}'] = dict(margin=float(d.mean()), lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0))
print(json.dumps(out, indent=1)); json.dump({'acf_abs_within_unit_sd': sd.to_dict(), 'sb_abs_vs': out}, open(O / 'extra_checks.json', 'w'), indent=1)
