"""Two notebooks side by side: PRE (the leaky scaler; the notebook as committed before this task) and POST (Pipeline, scaler fitted within each training fold)."""
import sys, os, io, json, contextlib, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']; RES = ROOT / 'thesis_results/production'; OUT = RES / 'scaler_fix_measurement'
PRE_NB = Path('/tmp/claude-1000/-home-sobottka-BSE-Master-Thesis-bse-thesis-synthetic-data/d811ca7f-e463-4bd9-934d-2554e3bfbe65/scratchpad/nb_pre_S4.ipynb'); POST_NB = ROOT / '3_4_integrated_pipeline.ipynb'
def load_ns(path):
    nb = json.load(open(path)); ns = {'__name__': 'nb'}
    skip = lambda s: s.lstrip().startswith('# ── GPU / device check') or 'This is the MAIN execution cell' in s
    with contextlib.redirect_stdout(io.StringIO()):
        for c in nb['cells']:
            s = ''.join(c['source'])
            if c['cell_type'] == 'code' and not skip(s): exec(compile(s, '<nb>', 'exec'), ns)
    return ns
_FM = {}
def _init():
    _FM['pre'] = load_ns(PRE_NB)['FinancialMetrics']; _FM['post'] = load_ns(POST_NB)['FinancialMetrics']
def job(t):
    a, b, which, cv = t
    try: return float(_FM[which](a, b).compute_discriminative_score(cv=cv)['auc'])
    except ValueError: return float('nan')
def pmap(tasks, procs=18):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(procs, initializer=_init) as p: return p.map(job, tasks, chunksize=16)
def load_test(m): return pd.read_parquet(ROOT / f'data/processed_files/test/{m}_test.parquet').to_numpy().astype(np.float32).flatten()
def load_full(m): return np.concatenate([pd.read_parquet(ROOT / f'data/processed_files/{s}/{m}_{s}.parquet').to_numpy().astype(np.float32).flatten() for s in ('train', 'valid', 'test')])
