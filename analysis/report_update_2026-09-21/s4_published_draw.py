"""S4 on the PUBLISHED draw, from the rewritten {market}_metrics.csv files (the values are computable exactly because S4 needs no AUC). Overall composite per model, first-second margin,
first ahead of second in k of 15 units, margins by seed, and a bootstrap of the 15 market-seed units (19,999 resamples, default_rng(20260921), as in the scenario-matrix work)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
RES = ROOT / 'thesis_results/production'; MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']; SEEDS = (42, 43, 44)
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']
units = [(m, s) for m in MARKETS for s in SEEDS]
comp = np.array([[pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model').loc[x, 'composite_rank'] for x in MODELS] for m, s in units])
fid = np.array([[pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model').loc[x, 'fidelity_rank'] for x in MODELS] for m, s in units])
tmp = np.array([[pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model').loc[x, 'temporal_rank'] for x in MODELS] for m, s in units])
ov = comp.mean(0); order = np.argsort(ov); a, b = int(order[0]), int(order[1])
rng = np.random.default_rng(20260921); B = 19999; idx = rng.integers(0, 15, (B, 15)); means = comp[idx].mean(1); mg = means[:, b] - means[:, a]; lo, hi = np.percentile(mg, [2.5, 97.5])
per_seed = comp.reshape(5, 3, 5).mean(0); by_seed = (per_seed[:, b] - per_seed[:, a]).tolist()
out = {'scheme': 'S4: five temporal metrics (both discriminative AUC columns descriptive); published draw; read from the rewritten metrics CSVs',
       'models': MODELS, 'overall_fidelity': fid.mean(0).tolist(), 'overall_temporal': tmp.mean(0).tolist(), 'overall_composite': ov.tolist(),
       'first': MODELS[a], 'second': MODELS[b], 'margin': float(ov[b] - ov[a]), 'bootstrap': {'resamples': B, 'unit': 'market-seed', 'n_units': 15, 'seed': 20260921, 'ci_lo': float(lo), 'ci_hi': float(hi), 'excludes_zero': bool(lo > 0 or hi < 0)},
       'first_ahead_of_second_units': int((comp[:, a] < comp[:, b]).sum()), 'ties': int((comp[:, a] == comp[:, b]).sum()), 'n_units': 15, 'margin_by_seed': dict(zip(map(str, SEEDS), by_seed))}
(RES / 'scenario_matrix' / 's4_published_draw.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ('first', 'second', 'margin', 'bootstrap', 'first_ahead_of_second_units', 'margin_by_seed')}, indent=1))
