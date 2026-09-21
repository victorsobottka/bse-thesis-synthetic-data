"""Part 1: the stationary GARCH-t reference null under the FIXED scaler (Task 10 recipe: parameters from the seed-42 GARCH fit, independent path pairs from seeds 100000+2k / +2k+1,
300 pairs per length, both evaluation lengths, contiguous and purged CV). Written to thesis_results/production/auc_null_reference/stationary_reference_null.json."""
import sys, json, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 's4_adoption_2026-09-21'))
from harness2 import *
import multiprocessing as mp
OUTD = RES / 'auc_null_reference'; t0 = time.time(); ns = load_ns(POST_NB); GARCHModel = ns['GARCHModel']; GenErr = ns['GenerationSanityError']; B = 300; out = {}
_P = {}
def _init(): _P['FM'] = load_ns(POST_NB)['FinancialMetrics']
def _feats(x, w=20): return np.array([[x[i-w:i].mean(), x[i-w:i].std(), np.abs(x[i-w:i]).mean(), (x[i-w:i]**2).mean()] for i in range(w, len(x))], dtype=np.float32)
def _shuffled(a, b):
    # DIAGNOSTIC ONLY (shuffled folds leak through the window overlap; the notebook refuses this scheme): same estimator as the notebook, scaler inside the fold.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    n = min(len(a), len(b)); r, s = _feats(a[:n]), _feats(b[:n]); k = min(len(r), len(s)); X = np.vstack([r[:k], s[:k]]); y = np.hstack([np.ones(k), np.zeros(k)])
    clf = Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=300, random_state=42))])
    return float(np.mean(cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring='roc_auc')))
def _job(t):
    a, b, cv = t
    if cv == 'shuffled': return _shuffled(a, b)
    return float(_P['FM'](a, b).compute_discriminative_score(**({} if cv == 'contiguous' else {'cv': cv}))['auc'])
qs = (2.5, 25, 50, 75, 97.5)
with mp.get_context('fork').Pool(18, initializer=_init) as pool:
    for m in MARKETS:
        rec = json.load(open(RES / m / 'seed42' / 'weights' / 'GARCH_params.json')); train = pd.read_parquet(ROOT / f'data/processed_files/train/{m}_train.parquet').to_numpy().astype(np.float32)
        model = GARCHModel.from_record(rec, train); test = load_test(m); N = len(load_full(m)); src = rec['generated_from']
        out[m] = {'model': 'GARCH(1,1)-t, seed-42 fit on the training block', 'generated_from': src, 'persistence': rec['constrained_persistence'] if src == 'constrained' else rec['unconstrained_persistence'], 'nulls': {}}
        for tag, L in (('track_a', len(test)), ('walk_forward', N // 6)):
            pairs, rej, k = [], 0, 0
            while len(pairs) < B and k < 4 * B:
                sa, sb = 100000 + 2 * k, 100000 + 2 * k + 1; k += 1
                try: model.seed = sa; a = model.generate(L).flatten(); model.seed = sb; b = model.generate(L).flatten()
                except GenErr: rej += 1; continue
                pairs.append((a, b))
            blk = {'length': int(L), 'n_pairs': len(pairs), 'rejected_by_generation_guard': rej, 'seed_base': 100000}
            for cv in ('contiguous', 'purged', 'shuffled'):
                a_ = np.array(pool.map(_job, [(x, y, cv) for x, y in pairs], chunksize=8))
                blk[cv] = {'mean': float(a_.mean()), 'std': float(a_.std()), 'quantiles': {str(q): float(np.percentile(a_, q)) for q in qs}, 'frac_below_0.5': float((a_ < .5).mean()), 'draws': [float(v) for v in a_]}
            out[m]['nulls'][tag] = blk
            print(m, tag, L, {cv: (round(blk[cv]['mean'], 3), round(blk[cv]['std'], 3), round(blk[cv]['quantiles']['97.5'], 3)) for cv in ('contiguous', 'purged', 'shuffled')}, f'rejected {rej}', f'({time.time()-t0:.0f}s)', flush=True)
out['_note'] = ('Reference for the AUC null under IDEAL conditions: independent paths of one stationary GARCH(1,1)-t process (persistence recorded per market; SHANGHAI is the constrained boundary fit), '
                'scored with the fixed scaler (fitted within each training fold). Regenerated 2026-09-21; the earlier Task 10 / Task 13 versions used the leaky scaler.')
json.dump(out, open(OUTD / 'stationary_reference_null.json', 'w'), indent=1); print('STATIONARY_DONE', round(time.time() - t0), 's')
