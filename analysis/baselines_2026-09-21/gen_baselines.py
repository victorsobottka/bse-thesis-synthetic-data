"""Part 1: no-oracle baseline generators. Both read ONLY the market's training block (data/processed_files/train/<market>_train.parquet).
  iid       historical simulation: returns resampled with replacement from the training block.
  sb_r      stationary block bootstrap (Politis-Romano 1994), mean block length = Politis-White (arch.bootstrap.optimal_block_length) on the
            training returns, floored at 1 (a mean block length below one observation is not defined).  The specified baseline.
  sb_abs    same resampler, mean block length = Politis-White on |training returns|, the dependence scale of volatility (about 110-130).  Added
            sensitivity, NOT the specified baseline.
The generating functions receive the training array, the target length and an rng, nothing else: the validation and test blocks are never loaded here.
20 draws per market-seed; draw k of seed s uses default_rng([s, k, code]). Output: draws/<market>_seed<s>.npz and baseline_provenance.json."""
import sys, json, hashlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'task11_2026-09-21'))
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd, arch
import pyarrow.parquet as pq
from arch.bootstrap import optimal_block_length
OUT = ROOT / 'thesis_results/production/baseline_generators'; (OUT / 'draws').mkdir(parents=True, exist_ok=True)
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']; SEEDS = (42, 43, 44); N_DRAWS = 20
CODE = {'iid': 11, 'sb_r': 12, 'sb_abs': 13}

def iid_bootstrap(train, n, rng):
    idx = rng.integers(0, len(train), n); return train[idx], idx
def stationary_bootstrap(train, n, mean_block, rng):
    """Politis & Romano (1994): start at a uniform index; at every step continue to the next observation (circularly, within the training block)
    with probability 1 - 1/b, otherwise jump to a new uniform start. Block lengths are geometric with mean b."""
    N = len(train); p = 1.0 / mean_block; jump = rng.random(n) < p; jump[0] = True; starts = rng.integers(0, N, n); idx = np.empty(n, dtype=np.int64)
    for t in range(n): idx[t] = starts[t] if jump[t] else (idx[t - 1] + 1) % N
    return train[idx], idx

if __name__ == '__main__':
    prov = {'arch': arch.__version__, 'numpy': np.__version__, 'n_draws': N_DRAWS, 'seeds': list(SEEDS), 'rng': 'default_rng([seed, draw, code]); codes ' + json.dumps(CODE),
            'source': 'training block only: data/processed_files/train/<market>_train.parquet. The validation block is not opened. From the test file only the row count is taken, from the parquet footer metadata (target length); no observation of it is read', 'markets': {}}
    for m in MARKETS:
        train = pd.read_parquet(ROOT / f'data/processed_files/train/{m}_train.parquet').to_numpy().astype(np.float32).flatten(); n_test = int(pq.ParquetFile(ROOT / f'data/processed_files/test/{m}_test.parquet').metadata.num_rows)   # row count from the parquet footer only; no observation is read
        x = train.astype(np.float64)
        b_r, b_abs, b_sq = (float(optimal_block_length(v)['stationary'].iloc[0]) for v in (x, np.abs(x), x ** 2))
        b = {'sb_r': max(b_r, 1.0), 'sb_abs': max(b_abs, 1.0)}
        rec = {'n_train': int(len(train)), 'train_sha256': hashlib.sha256(train.tobytes()).hexdigest(), 'target_length': int(n_test), 'politis_white_stationary': {'returns': b_r, 'abs_returns': b_abs, 'squared_returns': b_sq},
               'mean_block_length_used': b, 'block_length_note': 'sb_r: Politis-White on the training returns (floored at 1); sb_abs: Politis-White on |returns|, the volatility-persistence scale', 'draws': {}}
        for s in SEEDS:
            arr = {}; chk = {'max_index_used': 0, 'all_values_in_train': True, 'empirical_mean_block_length': {}}
            for k in range(N_DRAWS):
                for name in ('iid', 'sb_r', 'sb_abs'):
                    rng = np.random.default_rng([s, k, CODE[name]])
                    d, idx = iid_bootstrap(train, n_test, rng) if name == 'iid' else stationary_bootstrap(train, n_test, b[name], rng)
                    assert idx.min() >= 0 and idx.max() < len(train)                     # only indices inside the training block
                    assert np.array_equal(d, train[idx])
                    chk['max_index_used'] = max(chk['max_index_used'], int(idx.max())); arr[f'{name}__{k}'] = d
                    if name != 'iid': chk['empirical_mean_block_length'].setdefault(name, []).append(float(n_test / (1 + np.sum(np.diff(idx) != 1))))
            chk['empirical_mean_block_length'] = {k_: float(np.mean(v)) for k_, v in chk['empirical_mean_block_length'].items()}
            np.savez_compressed(OUT / 'draws' / f'{m}_seed{s}.npz', **arr); rec['draws'][str(s)] = chk
        prov['markets'][m] = rec
        print(f"{m}: n_train {len(train)}, target length {n_test}; Politis-White b: returns {b_r:.2f}, |r| {b_abs:.1f}, r^2 {b_sq:.1f}; used sb_r {b['sb_r']:.2f}, sb_abs {b['sb_abs']:.1f}; empirical mean block (seed 42) {rec['draws']['42']['empirical_mean_block_length']}; max index used {max(v['max_index_used'] for v in rec['draws'].values())} < {len(train)}")
    json.dump(prov, open(OUT / 'baseline_provenance.json', 'w'), indent=1); print('BASELINES_DONE')
