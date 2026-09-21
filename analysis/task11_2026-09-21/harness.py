"""Shared harness for the Task 11 scenario matrix. Uses the notebook's own FinancialMetrics (with the purged option),
in a fork pool; 'shuffled' is a DIAGNOSTIC re-implementation (the notebook refuses it) checked against the notebook for the other two schemes."""
import sys, os, io, json, contextlib, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ.setdefault(v, '1')
import numpy as np, pandas as pd
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
RES = ROOT / 'thesis_results/production'; OUT = RES / 'scenario_matrix'
_FM = None
def load_fm():
    nb = json.load(open(ROOT / '3_4_integrated_pipeline.ipynb')); ns = {'__name__': 'task11_nb'}
    skip = lambda s: s.lstrip().startswith('# ── GPU / device check') or 'This is the MAIN execution cell' in s
    with contextlib.redirect_stdout(io.StringIO()):
        for c in nb['cells']:
            s = ''.join(c['source'])
            if c['cell_type'] == 'code' and not skip(s): exec(compile(s, '<nb>', 'exec'), ns)
    return ns
def _init():
    global _FM
    _FM = load_fm()['FinancialMetrics']
def _feats(x, w=20):
    return np.array([[x[i - w:i].mean(), x[i - w:i].std(), np.abs(x[i - w:i]).mean(), (x[i - w:i] ** 2).mean()] for i in range(w, len(x))], dtype=np.float32)
def auc_shuffled(a, b, w=20):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    n = min(len(a), len(b)); r, s = _feats(a[:n], w), _feats(b[:n], w); k = min(len(r), len(s))
    X = StandardScaler().fit_transform(np.vstack([r[:k], s[:k]])); y = np.hstack([np.ones(k), np.zeros(k)])
    return float(np.mean(cross_val_score(LogisticRegression(max_iter=300, random_state=42), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')))
def job(t):
    a, b, scheme = t
    try:
        if scheme == 'shuffled': return auc_shuffled(a, b)
        return float(_FM(a, b).compute_discriminative_score(cv=scheme)['auc'])
    except ValueError:                       # purged fold too small to fit (raised by the notebook option), or a single class
        return float('nan')
def pmap(tasks, procs=18):
    import multiprocessing as mp
    with mp.get_context('fork').Pool(procs, initializer=_init) as p: return p.map(job, tasks, chunksize=16)
def load_test(m): return pd.read_parquet(ROOT / f'data/processed_files/test/{m}_test.parquet').to_numpy().astype(np.float32).flatten()
def load_full(m): return np.concatenate([pd.read_parquet(ROOT / f'data/processed_files/{s}/{m}_{s}.parquet').to_numpy().astype(np.float32).flatten() for s in ('train', 'valid', 'test')])
