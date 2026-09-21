"""Regenerate the 20 Phase B draws per (market, seed, model) exactly as embedding_distance.py did and cache the raw series.

Why: Part 2c (fixed-bandwidth null) and Part 5 (encoder robustness) re-encode the same series with different
kernels / encoders. Regenerating them once, with the same code path (reconstruct_model + _regenerate + _draw_seed),
keeps every comparison on identical draws. Nothing is written to Phase A outputs. Verified against the stored
embedding_distance JSONs in part2c (statistic reproduced draw by draw).
"""
import sys, io, json, time, contextlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
import embedding_distance as ed
OUT = ROOT / 'thesis_results/production/overnight_2026-09-21/draws'; OUT.mkdir(parents=True, exist_ok=True)
N_DRAWS = 20
ns = ed.load_notebook(); root = ns['RESULTS']; factories = ns['_build_model_factories']()[1]
t0 = time.time()
for market in ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']:
    parts, mu, sd = ed.load_market(market, ns); train, test = parts['train'], parts['test']; L = len(test)
    for seed in (42, 43, 44):
        f = OUT / f'{market}_seed{seed}.npz'
        if f.exists(): continue
        wdir = root / market / f'seed{seed}' / 'weights'
        with contextlib.redirect_stdout(io.StringIO()):
            models = {m: ns['reconstruct_model'](m, factories[m][0], wdir, train.astype(np.float32).reshape(-1, 1)) for m in ed.MODELS}
        arr = {}
        for k in range(N_DRAWS):
            ds = ns['_draw_seed'](seed, k)
            for m in ed.MODELS:
                with contextlib.redirect_stdout(io.StringIO()):
                    arr[f'{m}__{k}'] = ns['_regenerate'](models[m], L, ds).astype(np.float64)
            arr[f'{ed.CONTROL}__{k}'] = np.random.default_rng(42 if k == 0 else ds).permutation(test)
        np.savez_compressed(f, **arr)
        print(f'{market} seed{seed}: {len(arr)} series cached ({time.time()-t0:.0f}s)', flush=True)
print('DRAWS_CACHE_DONE', round(time.time() - t0), 's', flush=True)
