"""Parts 2-3: the scenario matrix and the bootstrap. Measurement only: reads published CSVs, Phase B draws, the nulls; writes comparison artifacts.

Frames.  P (published draw): S0 and S1 only, exact, from {market}_metrics.csv (the recorded GAN series behind those metrics were never stored, so
purged AUCs of the recorded draws cannot be computed).  F (fresh draws): every scenario, per Phase B draw k (20 per unit) using that draw's own metrics and
AUCs; unit composite = mean over the 20 draws of the per-draw composite. Only the scenario differs between F rows.
Ranks: among the five generators, na_option='bottom', method 'average' (the pipeline's). Control: re-ranked as a SIXTH competitor (6-pool) to report its composite.
"""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
FID = ['mean_diff', 'std_diff', 'wasserstein', 'quantile_mse', 'tail_index_diff', 'extreme_events_diff', 'energy_distance']
T5 = ['acf_returns_mae', 'acf_absolute_mae', 'acf_squared_mae', 'hurst_diff', 'resid_kurtosis_diff']
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; CTRL = 'CONTROL: shuffled real'
SEEDS = (42, 43, 44)
SCEN = {  # name: (temporal columns, auc variant, null)
    'S0': (T5 + ['dist', 'absz'], 'contiguous', 'legacy'),
    'S1': (T5 + ['dist', 'absz'], 'contiguous', 'length-matched'),
    'S2': (T5 + ['dist', 'absz'], 'purged', 'length-matched'),
    'S3': (T5 + ['dist'], 'purged', None),
    'S4': (T5, None, None),
    'S5': (T5 + ['dist', 'absz'], 'purged', 'stationary GARCH-t'),
}
def composite(tab, temporal):
    """tab: rows = competitors, columns = raw metrics + 'dist','absz'. Returns fidelity, temporal, composite (Series, same index)."""
    f = pd.concat([tab[c].rank(na_option='bottom') for c in FID], axis=1).mean(axis=1)
    t = pd.concat([tab[c].rank(na_option='bottom') for c in temporal], axis=1).mean(axis=1)
    return f, t, (f + t) / 2
def stats(a): a = np.asarray(a, float); return a.mean(), a.std()          # ddof 0, as the pipeline's null_std

# ------------------------------------------------------------------ nulls (Track A length)
NUL = {}
for m in MARKETS:
    j = json.load(open(RES / m / 'auc_null.json')); c = np.load(OUT / 'nulls' / f'{m}_track_a_contiguous.npy'); p = np.load(OUT / 'nulls' / f'{m}_track_a_purged.npy')
    sc = np.load(OUT / 'nulls' / f'stationary_{m}_track_a_contiguous.npy'); sp = np.load(OUT / 'nulls' / f'stationary_{m}_track_a_purged.npy')
    NUL[m] = {'legacy': (j['legacy_null']['mean'], j['legacy_null']['std']), 'lm_contiguous': stats(c), 'lm_purged': stats(p), 'st_contiguous': stats(sc), 'st_purged': stats(sp)}
# ------------------------------------------------------------------ S0 validation against the published ranks
pub, vrows = {}, []
for m in MARKETS:
    for s in SEEDS:
        df = pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv'); pub[(m, s)] = df
        g = df[~df.model.str.startswith('CONTROL')].set_index('model'); g['dist'] = g['discriminative_auc_dist']; g['absz'] = g['discriminative_auc_absz']
        f, t, c = composite(g, SCEN['S0'][0])
        vrows.append(dict(market=m, seed=s, d_fid=(f - g.fidelity_rank).abs().max(), d_tmp=(t - g.temporal_rank).abs().max(), d_comp=(c - g.composite_rank).abs().max(),
                          d_ranks=max((g[x].rank(na_option='bottom') - g[x + '_rank']).abs().max() for x in FID + T5 + ['discriminative_auc_dist', 'discriminative_auc_absz'])))
V = pd.DataFrame(vrows); V.to_csv(OUT / 'validation_S0_vs_published.csv', index=False)
ov = pd.read_csv(RES / 'overall_performance.csv'); ov = ov[~ov.model.str.startswith('CONTROL')].set_index('model')
print('S0 validation: max |recomputed - published| fidelity %.3g temporal %.3g composite %.3g per-metric ranks %.3g over 15 units x 5 models' % (V.d_fid.max(), V.d_tmp.max(), V.d_comp.max(), V.d_ranks.max()))

# ------------------------------------------------------------------ frame P: S0 and S1 from published values (5-pool; control in 6-pool)
P_rows = []
for (m, s), df in pub.items():
    for scen in ('S0', 'S1'):
        d = df.set_index('model').copy(); auc = d['discriminative_auc_raw']
        d['dist'] = (auc - 0.5).abs()
        if scen == 'S0': d['absz'] = d['discriminative_auc_absz']
        else: mu, sd = NUL[m]['lm_contiguous']; d['absz'] = (auc - mu).abs() / sd
        g = d[d.index != CTRL]; f, t, c = composite(g, SCEN[scen][0])
        f6, t6, c6 = composite(d, SCEN[scen][0])
        for mod in g.index: P_rows.append(dict(frame='P', scenario=scen, market=m, seed=s, model=mod, fidelity=f[mod], temporal=t[mod], composite=c[mod], composite_6pool=c6[mod], fidelity_6pool=f6[mod], temporal_6pool=t6[mod]))
        P_rows.append(dict(frame='P', scenario=scen, market=m, seed=s, model=CTRL, fidelity=np.nan, temporal=np.nan, composite=np.nan, composite_6pool=c6[CTRL], fidelity_6pool=f6[CTRL], temporal_6pool=t6[CTRL]))
P = pd.DataFrame(P_rows)
# ------------------------------------------------------------------ frame F
OBS = pd.read_csv(OUT / 'observed_auc_by_draw.csv'); F_rows = []; chk = []
for (m, s), df in pub.items():
    pb = pd.read_csv(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / f'{m}_metrics_phase_b_COMMIT_DRIFT.csv')
    ob = OBS[(OBS.market == m) & (OBS.seed == s)]
    for k, dk in pb.groupby('draw'):
        d = dk.set_index('model').copy(); o = ob[ob.draw == k].set_index('model')
        d['auc_c'] = o.auc_contiguous; d['auc_p'] = o.auc_purged
        chk.append(((d.auc_c - d.discriminative_auc_raw).abs().max()))
        for scen, (temporal, variant, nullname) in SCEN.items():
            t = d.copy()
            if variant is not None:
                auc = t['auc_c'] if variant == 'contiguous' else t['auc_p']; t['dist'] = (auc - 0.5).abs()
                if nullname == 'legacy': mu, sd = NUL[m]['legacy']
                elif nullname == 'length-matched': mu, sd = NUL[m]['lm_contiguous' if variant == 'contiguous' else 'lm_purged']
                elif nullname == 'stationary GARCH-t': mu, sd = NUL[m]['st_purged']
                else: mu = sd = np.nan
                t['absz'] = (auc - mu).abs() / sd if nullname else np.nan
            g = t[t.index != CTRL]; f, tm, c = composite(g, temporal); f6, t6, c6 = composite(t, temporal)
            for mod in g.index: F_rows.append(dict(frame='F', scenario=scen, market=m, seed=s, draw=int(k), model=mod, fidelity=f[mod], temporal=tm[mod], composite=c[mod], composite_6pool=c6[mod], fidelity_6pool=f6[mod], temporal_6pool=t6[mod]))
            F_rows.append(dict(frame='F', scenario=scen, market=m, seed=s, draw=int(k), model=CTRL, fidelity=np.nan, temporal=np.nan, composite=np.nan, composite_6pool=c6[CTRL], fidelity_6pool=f6[CTRL], temporal_6pool=t6[CTRL]))
        # sanity: S0 recomputed from the draw equals Phase B's own composite_rank
        g0 = pd.DataFrame([r for r in F_rows[-6 * len(SCEN):] if r['scenario'] == 'S0' and r['model'] != CTRL]).set_index('model').composite
        chk[-1] = max(chk[-1], (g0 - d.loc[g0.index, 'composite_rank']).abs().max()) if 'composite_rank' in d else chk[-1]
F = pd.DataFrame(F_rows); print('frame F: max |contiguous AUC - Phase B raw AUC| / |S0 composite - Phase B composite_rank| = %.3g over %d unit-draws' % (max(chk), len(chk)))
Fu = F.groupby(['scenario', 'market', 'seed', 'model'], as_index=False)[['fidelity', 'temporal', 'composite', 'composite_6pool', 'fidelity_6pool', 'temporal_6pool']].mean()
Fu['frame'] = 'F'; Pu = P.copy()
U = pd.concat([Pu, Fu], ignore_index=True); U.to_csv(OUT / 'unit_composites.csv', index=False)
F.to_csv(OUT / 'draw_composites_frame_F.csv.gz', index=False)
print('SCENARIOS_COMPUTED')
