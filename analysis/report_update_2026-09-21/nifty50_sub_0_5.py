"""NIFTY50, the draws of the full-series sliding null whose contiguous AUC is below 0.5: their mean under contiguous, purged and shuffled folds, all with the FIXED scaler, at both
evaluation lengths. Contiguous and purged come from the regenerated auc_null.json; shuffled (a diagnostic that leaks through the window overlap; the notebook refuses it) is computed here on the
same pairs with the same estimator and the scaler inside the fold. Output: auc_null_reference/nifty50_sub_0_5.json."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 's4_adoption_2026-09-21'))
from harness2 import *
import multiprocessing as mp
def _feats(x, w=20): return np.array([[x[i-w:i].mean(), x[i-w:i].std(), np.abs(x[i-w:i]).mean(), (x[i-w:i]**2).mean()] for i in range(w, len(x))], dtype=np.float32)
def _shuffled(t):
    a, b = t
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    n = min(len(a), len(b)); r, s = _feats(a[:n]), _feats(b[:n]); k = min(len(r), len(s)); X = np.vstack([r[:k], s[:k]]); y = np.hstack([np.ones(k), np.zeros(k)])
    clf = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=300, random_state=42))])
    return float(np.mean(cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')))
j = json.load(open(RES / 'NIFTY50' / 'auc_null.json'))['sliding']; full = load_full('NIFTY50'); out = {'market': 'NIFTY50', 'scaler': 'fit_within_training_fold'}
with mp.get_context('fork').Pool(18) as pool:
    for tag in ('track_a', 'walk_forward'):
        L = j[tag]['length']; pairs = [(full[s:s + L], full[s + L:s + 2 * L]) for s in range(0, len(full) - 2 * L + 1)]
        c, p = np.array(j[tag]['contiguous']['draws']), np.array(j[tag]['purged']['draws']); sh = np.array(pool.map(_shuffled, pairs, chunksize=16)); low = c < .5
        out[tag] = {'length': L, 'positions': int(len(c)), 'n_sub_0_5': int(low.sum()), 'share_sub_0_5': float(low.mean()), 'mean_contiguous': float(c[low].mean()), 'mean_purged': float(p[low].mean()), 'mean_shuffled': float(sh[low].mean()),
                    'share_above_0_5_purged': float((p[low] > .5).mean()), 'share_above_0_5_shuffled': float((sh[low] > .5).mean()), 'share_null_below_0_5_contiguous': float(low.mean()), 'share_null_below_0_5_purged': float((p < .5).mean())}
        print(tag, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[tag].items()}, flush=True)
json.dump(out, open(RES / 'auc_null_reference' / 'nifty50_sub_0_5.json', 'w'), indent=1)
