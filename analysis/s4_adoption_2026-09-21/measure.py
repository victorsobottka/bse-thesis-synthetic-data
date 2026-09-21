"""Part 4 (compute). POST-fix (scaler fitted within each training fold) full-series sliding nulls at both evaluation lengths, contiguous and purged, on the same
pairs as Task 11's PRE-fix nulls (reused from scenario_matrix/nulls, spot-checked against the PRE notebook); the legacy 999-draw null (compute_auc_null, contiguous) POST-fix;
Frame F observed AUCs PRE and POST for the 20 fresh Phase B draws x 3 seeds x (5 models + control). Nothing is written to auc_null.json or any published file."""
import sys, time, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness2 import *
t0 = time.time(); (OUT / 'nulls').mkdir(exist_ok=True)
T11 = RES / 'scenario_matrix' / 'nulls'
# ---- spot check: PRE notebook reproduces Task 11's stored leaky nulls
chk = []; tasks = []; keys = []
for m in MARKETS:
    full, test = load_full(m), load_test(m); L = len(test); rng = np.random.default_rng(1); st = rng.integers(0, len(full) - 2 * L + 1, 40)
    for s in st:
        for cv in ('contiguous', 'purged'): tasks.append((full[s:s + L], full[s + L:s + 2 * L], 'pre', cv)); keys.append((m, int(s), cv))
res = pmap(tasks)
for (m, s, cv), v in zip(keys, res): chk.append(abs(v - np.load(T11 / f'{m}_track_a_{cv}.npy')[s]))
print('PRE notebook vs Task 11 stored leaky nulls (400 positions): max |diff| = %.2e' % max(chk), flush=True)
# ---- POST nulls
for m in MARKETS:
    full, test = load_full(m), load_test(m); N = len(full)
    for tag, L in (('track_a', len(test)), ('walk_forward', N // 6)):
        pairs = [(full[s:s + L], full[s + L:s + 2 * L]) for s in range(0, N - 2 * L + 1)]
        for cv in ('contiguous', 'purged'):
            a = np.array(pmap([(x, y, 'post', cv) for x, y in pairs])); np.save(OUT / 'nulls' / f'{m}_{tag}_{cv}_fixed.npy', a)
        print(f'POST null {m} {tag} L={L} done ({time.time()-t0:.0f}s)', flush=True)
# ---- Frame F observed, PRE and POST
tasks = []; meta = []; CT = 'CONTROL: shuffled real'
for m in MARKETS:
    real = load_test(m); ctrl = np.random.default_rng(42).permutation(real).copy()
    for s in (42, 43, 44):
        ser = np.load(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / 'synthetic_phase_b_COMMIT_DRIFT.npz')
        items = [(k.split('__draw')[0], int(k.split('__draw')[1]), ser[k].flatten().astype(np.float32)) for k in ser.files if '__draw' in k] + [(CT, k, ctrl) for k in range(20)]
        for mod, k, x in items:
            for which in ('pre', 'post'):
                for cv in ('contiguous', 'purged'): tasks.append((real, x, which, cv))
            meta.append((m, s, mod, k))
res = np.array(pmap(tasks)).reshape(-1, 4)
D = pd.DataFrame(meta, columns=['market', 'seed', 'model', 'draw']); D[['pre_contiguous', 'pre_purged', 'post_contiguous', 'post_purged']] = res; D.to_csv(OUT / 'observed_frameF_auc.csv', index=False)
old = pd.read_csv(RES / 'scenario_matrix' / 'observed_auc_by_draw.csv'); g = D.merge(old, on=['market', 'seed', 'model', 'draw'])
print('Frame F: PRE (this run) vs Task 11 stored: max |diff| contiguous %.2e, purged %.2e (%d series)' % ((g.pre_contiguous - g.auc_contiguous).abs().max(), (g.pre_purged - g.auc_purged).abs().max(), len(g)), flush=True)
# ---- legacy 999-draw half-split null, POST (compute_auc_null), contiguous; PRE values are the cached auc_null.json
post = load_ns(POST_NB)['FinancialMetrics']; leg = {}
for m in MARKETS:
    real = load_test(m); r = post(real, real).compute_auc_null(n_rep=999, seed=post.AUC_NULL_SEED, return_draws=True); d = np.array(r['draws']); j = json.load(open(RES / m / 'auc_null.json'))
    leg[m] = dict(pre=dict(mean=j['mean'], sd=j['std'], p97_5=j['quantiles']['97.5'], legacy20_mean=j['legacy_null']['mean'], legacy20_sd=j['legacy_null']['std']), post=dict(mean=float(d.mean()), sd=float(d.std()), p97_5=float(np.percentile(d, 97.5)), legacy20_mean=float(d[:20].mean()), legacy20_sd=float(d[:20].std())),
                  max_abs_change_first_20=float(np.abs(d[:20] - np.array(j['draws'][:20])).max()), max_abs_change_all=float(np.abs(d - np.array(j['draws'])).max()))
    print('legacy null', m, leg[m]['pre'], leg[m]['post'], flush=True)
json.dump(leg, open(OUT / 'legacy_999_null_pre_post.json', 'w'), indent=1); print('MEASURE_DONE', round(time.time() - t0), 's')
