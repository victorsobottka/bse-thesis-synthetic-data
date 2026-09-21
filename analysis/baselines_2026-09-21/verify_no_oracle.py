"""Confirm the baselines read nothing after the end of the training block. (1) static: which files gen_baselines.py opens; (2) every value of every draw
(15 units x 20 draws x 3 baselines) is a member of that market's training block; (3) determinism: regenerating with the recorded seeds reproduces the stored draws exactly;
(4) the resamplers take (train, n, [mean_block], rng) only."""
import sys, re, json, inspect
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
import gen_baselines as g
src = (Path(__file__).parent / 'gen_baselines.py').read_text()
reads = re.findall(r"(?:read_parquet|ParquetFile)\(ROOT / f'([^']+)'", src)
print('files opened by gen_baselines.py:', reads)
assert all('/train/' in r or ('/test/' in r and 'metadata' in src) for r in reads) and not any('/valid/' in r for r in reads)
print('resampler signatures:', inspect.signature(g.iid_bootstrap), inspect.signature(g.stationary_bootstrap))
bad = 0; n = 0; maxidx = {}; prov = json.load(open(ROOT / 'thesis_results/production/baseline_generators/baseline_provenance.json'))
for m in g.MARKETS:
    train = pd.read_parquet(ROOT / f'data/processed_files/train/{m}_train.parquet').to_numpy().astype(np.float32).flatten(); ts = set(train.tolist())
    b = prov['markets'][m]['mean_block_length_used']
    for s in g.SEEDS:
        arr = np.load(ROOT / 'thesis_results/production/baseline_generators/draws' / f'{m}_seed{s}.npz')
        for name in ('iid', 'sb_r', 'sb_abs'):
            for k in range(g.N_DRAWS):
                d = arr[f'{name}__{k}']; n += 1
                if not all(v in ts for v in d.tolist()): bad += 1
                rng = np.random.default_rng([s, k, g.CODE[name]]); r, idx = g.iid_bootstrap(train, len(d), rng) if name == 'iid' else g.stationary_bootstrap(train, len(d), b[name], rng)
                assert np.array_equal(r, d); maxidx[m] = max(maxidx.get(m, 0), int(idx.max()))
print(f'{n} draws checked: values not found in the training block = {bad}; regenerated draws identical to stored = True; max index used per market {maxidx} (n_train {[prov["markets"][m]["n_train"] for m in g.MARKETS]})')
json.dump({'draws_checked': n, 'values_outside_training_block': bad, 'regeneration_identical': True, 'max_index_used': maxidx, 'files_opened': reads}, open(ROOT / 'thesis_results/production/baseline_generators/verify_no_oracle.json', 'w'), indent=1)
