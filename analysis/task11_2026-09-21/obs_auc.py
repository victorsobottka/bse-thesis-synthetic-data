"""Observed discriminative AUC on the Phase B fresh draws (20 per market-seed x 6 series), under contiguous / purged (notebook option) / shuffled (diagnostic).
Series come from each unit's own synthetic_phase_b_COMMIT_DRIFT.npz, so the contiguous value must equal Phase B's discriminative_auc_raw (checked)."""
import sys, time
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
t0 = time.time(); rows = []; tasks = []; meta = []
for m in MARKETS:
    real = load_test(m)
    for s in (42, 43, 44):
        d = RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT'; ser = np.load(d / 'synthetic_phase_b_COMMIT_DRIFT.npz')
        for key in [k_ for k_ in ser.files if '__draw' in k_]:
            mod, k = key.split('__draw'); x = ser[key].flatten()
            for sc in ('contiguous', 'purged', 'shuffled'): tasks.append((real, x.astype(np.float32) if x.dtype != np.float32 else x, sc))
            meta.append((m, s, mod, int(k)))
res = np.array(pmap(tasks)).reshape(-1, 3)
D = pd.DataFrame(meta, columns=['market', 'seed', 'model', 'draw']); D[['auc_contiguous', 'auc_purged', 'auc_shuffled']] = res
# The control: evaluate_multiple_models adds default_rng(42).permutation(real) itself, so it is the SAME permutation in every Phase B draw (as in the published run).
CTRL = 'CONTROL: shuffled real'; crow = []; ctasks = []; cmeta = []
for m in MARKETS:
    real = load_test(m)
    for sd_ in [42] + list(range(1, 21)):        # 42 = the pipeline's control; 1..20 = extra permutations (sensitivity, not used in the scenarios)
        x = np.random.default_rng(sd_).permutation(real).copy()
        for sc in ('contiguous', 'purged', 'shuffled'): ctasks.append((real, x, sc))
        cmeta.append((m, sd_))
cres = np.array(pmap(ctasks)).reshape(-1, 3); CP = pd.DataFrame(cmeta, columns=['market', 'perm_seed']); CP[['auc_contiguous', 'auc_purged', 'auc_shuffled']] = cres; CP.to_csv(OUT / 'control_permutation_auc.csv', index=False)
for m in MARKETS:
    c42 = CP[(CP.market == m) & (CP.perm_seed == 42)].iloc[0]
    for s in (42, 43, 44):
        for k in range(20): crow.append((m, s, CTRL, k, c42.auc_contiguous, c42.auc_purged, c42.auc_shuffled))
D = pd.concat([D, pd.DataFrame(crow, columns=D.columns)], ignore_index=True)
# reproduction of Phase B's own raw AUC
chk = []
for m in MARKETS:
    for s in (42, 43, 44):
        pb = pd.read_csv(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / f'{m}_metrics_phase_b_COMMIT_DRIFT.csv')
        g = D[(D.market == m) & (D.seed == s)].merge(pb[['draw', 'model', 'discriminative_auc_raw']], on=['draw', 'model'])
        chk.append((g.auc_contiguous - g.discriminative_auc_raw).abs().max())
D.to_csv(OUT / 'observed_auc_by_draw.csv', index=False)
print(f'{len(D)} series scored (incl. control); max |contiguous - Phase B discriminative_auc_raw| over all units = {max(chk):.3g} ({time.time()-t0:.0f}s)')
