"""Parts 2-3: rank the baselines with the generators and the control in one pool, under S0 (published columns) and S4 (both discriminative columns out),
per-metric ranks, the ACF mechanism, bootstrap of margins to the best generator, draw-to-draw spread. Comparison entries only: nothing here feeds the published composite."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'task11_2026-09-21'))
from harness import *
OUT2 = RES / 'baseline_generators'
FID = ['mean_diff', 'std_diff', 'wasserstein', 'quantile_mse', 'tail_index_diff', 'extreme_events_diff', 'energy_distance']
T5 = ['acf_returns_mae', 'acf_absolute_mae', 'acf_squared_mae', 'hurst_diff', 'resid_kurtosis_diff']
GENS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; CTRL = 'CONTROL: shuffled real'
SC = {'S0': T5 + ['dist', 'absz'], 'S4': T5}
D = pd.read_csv(OUT2 / 'metrics_all_entries_by_draw.csv'); D['dist'] = D.discriminative_auc_dist; D['absz'] = D.discriminative_auc_absz
UNITS = [(m, s) for m in MARKETS for s in (42, 43, 44)]
FRAMES = {'A': GENS + [CTRL, 'iid', 'sb_r'], 'B': GENS + [CTRL, 'iid', 'sb_abs'], 'C': GENS + [CTRL, 'iid', 'sb_r', 'sb_abs']}
def comp(t, temporal):
    f = pd.concat([t[c].rank(na_option='bottom') for c in FID], axis=1).mean(axis=1); tm = pd.concat([t[c].rank(na_option='bottom') for c in temporal], axis=1).mean(axis=1); return f, tm, (f + tm) / 2
rows = []
for fr, ents in FRAMES.items():
    for (m, s, k), g in D[D.model.isin(ents)].groupby(['market', 'seed', 'draw']):
        t = g.set_index('model').loc[ents]
        for sc, tmp in SC.items():
            f, tm, c = comp(t, tmp)
            for e in ents: rows.append((fr, sc, m, s, k, e, f[e], tm[e], c[e]))
# frame P: the published generators and control (one recorded draw) against each baseline draw
PUB = {}
for m, s in UNITS: PUB[(m, s)] = pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model')
for (m, s, k), g in D[D.model.isin(['iid', 'sb_r'])].groupby(['market', 'seed', 'draw']):
    pubt = PUB[(m, s)][[c for c in FID + T5 + ['discriminative_auc_dist', 'discriminative_auc_absz'] if c in PUB[(m, s)].columns]].copy(); pubt = pubt.rename(columns={'discriminative_auc_dist': 'dist', 'discriminative_auc_absz': 'absz'})
    base = g.set_index('model')[FID + T5 + ['dist', 'absz']]; t = pd.concat([pubt.loc[GENS + [CTRL]], base]); ents = GENS + [CTRL, 'iid', 'sb_r']
    for sc, tmp in SC.items():
        f, tm, c = comp(t, tmp)
        for e in ents: rows.append(('P', sc, m, s, k, e, f[e], tm[e], c[e]))
R = pd.DataFrame(rows, columns=['frame', 'scenario', 'market', 'seed', 'draw', 'entry', 'fidelity', 'temporal', 'composite']); R.to_csv(OUT2 / 'ranks_by_draw.csv.gz', index=False)
# ---- overall table per frame / scenario
ov = []
for (fr, sc), g in R.groupby(['frame', 'scenario']):
    u = g.groupby(['market', 'seed', 'entry'])[['fidelity', 'temporal', 'composite']].mean().reset_index(); o = u.groupby('entry')[['fidelity', 'temporal', 'composite']].mean(); o['position'] = o.composite.rank(method='min').astype(int)
    for e, r in o.iterrows(): ov.append(dict(frame=fr, scenario=sc, entry=e, fidelity_rank=r.fidelity, temporal_rank=r.temporal, composite=r.composite, position=int(r.position), n_entries=len(o)))
OV = pd.DataFrame(ov); OV.to_csv(OUT2 / 'overall_ranks.csv', index=False)
pd.set_option('display.width', 220)
for fr in ('A', 'B', 'C', 'P'):
    print(f'\n== frame {fr}'); print(OV[OV.frame == fr].pivot_table(index='entry', columns='scenario', values=['fidelity_rank', 'temporal_rank', 'composite', 'position']).round(3).to_string())
# ---- per-metric ranks (frame A, 8 entries, fresh draws): mean rank per entry and metric; and the no-dynamics entries against the generators
PM = []
for (m, s, k), g in D[D.model.isin(FRAMES['A'])].groupby(['market', 'seed', 'draw']):
    t = g.set_index('model').loc[FRAMES['A']]
    for c in FID + T5 + ['dist', 'absz']:
        r = t[c].rank(na_option='bottom')
        for e in FRAMES['A']: PM.append((c, e, r[e]))
PM = pd.DataFrame(PM, columns=['metric', 'entry', 'rank']); PMR = PM.groupby(['metric', 'entry'])['rank'].mean().unstack('entry')[FRAMES['A']]; PMR['generators_mean'] = PMR[GENS].mean(axis=1); PMR.to_csv(OUT2 / 'per_metric_mean_rank_frame_A.csv')
print('\nper-metric mean rank of 8 (1 = best), frame A'); print(PMR.loc[FID + T5 + ['dist', 'absz']].round(2).to_string())
RAW = D[D.model.isin(FRAMES['C'])].groupby('model')[T5 + ['dist', 'absz']].mean().loc[FRAMES['C']]; RAW.to_csv(OUT2 / 'raw_temporal_metrics_mean_by_entry.csv'); print('\nraw temporal metric means (15 units x 20 draws)'); print(RAW.round(4).to_string())
# published-run raw values: where do 0.0730 / 0.0981 / 0.1232 come from?
pubraw = pd.concat([PUB[u].assign(market=u[0], seed=u[1]) for u in UNITS]); pr = pubraw.reset_index().groupby('model').acf_absolute_mae.agg(['mean', 'median', 'min', 'max']); print('\npublished acf_absolute_mae by model (15 units)'); print(pr.round(4).to_string()); pr.to_csv(OUT2 / 'published_acf_absolute_mae_by_model.csv')
ps = pubraw.reset_index(); ps = ps[ps.model.isin(['QuantGAN', 'TimeGAN', CTRL])].pivot_table(index=['market', 'seed'], columns='model', values='acf_absolute_mae'); print('units where (control, QuantGAN, TimeGAN) close to (0.0730, 0.0981, 0.1232):'); ps['dist_to_note'] = ((ps[CTRL] - .0730).abs() + (ps['QuantGAN'] - .0981).abs() + (ps['TimeGAN'] - .1232).abs()); print(ps.sort_values('dist_to_note').head(3).round(4).to_string())
# ---- ACF mechanism: profiles of |r| for real, control, generators, baselines (lags 1..50), from the same series the metrics used
def acf(x, L):
    x = x - x.mean(); c0 = np.dot(x, x) / len(x); return np.array([np.dot(x[:-l], x[l:]) / len(x) / c0 for l in range(1, L + 1)])
prof = {e: [] for e in ['real'] + FRAMES['C']}; bias = {e: [] for e in FRAMES['C']}; mae = {e: [] for e in FRAMES['C']}
for m, s in UNITS:
    real = load_test(m).astype(np.float64); ra = acf(np.abs(real), 50); ser = np.load(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / 'synthetic_phase_b_COMMIT_DRIFT.npz'); bs = np.load(OUT2 / 'draws' / f'{m}_seed{s}.npz')
    cs = np.random.default_rng(42).permutation(load_test(m)).astype(np.float64)
    for k in range(20):
        src = {**{g: ser[f'{g}__draw{k}'].astype(np.float64) for g in GENS}, CTRL: cs, 'iid': bs[f'iid__{k}'].astype(np.float64), 'sb_r': bs[f'sb_r__{k}'].astype(np.float64), 'sb_abs': bs[f'sb_abs__{k}'].astype(np.float64)}
        if k == 0: prof['real'].append(ra)
        for e, x in src.items():
            a = acf(np.abs(x), 50); bias[e].append((a - ra).mean()); mae[e].append(np.abs(a - ra).mean())
            if k == 0: prof[e].append(a)
PR = pd.DataFrame({e: np.mean(v, axis=0) for e, v in prof.items()}, index=range(1, 51)); PR.to_csv(OUT2 / 'acf_abs_profile_mean.csv')
MECH = pd.DataFrame({'mean_acf_abs_lags_1_50': PR.mean(), 'signed_bias_vs_real_mean': pd.Series({e: np.mean(v) for e, v in bias.items()}), 'mae_vs_real_mean': pd.Series({e: np.mean(v) for e, v in mae.items()}),
                     'share_units_draws_mae_gt_control': pd.Series({e: float(np.mean(np.array(mae[e]) > np.array(mae[CTRL]))) for e in FRAMES['C']}), 'acf_abs_lag1': PR.loc[1], 'acf_abs_lag10': PR.loc[10], 'acf_abs_lag50': PR.loc[50]}); MECH.to_csv(OUT2 / 'acf_abs_mechanism.csv')
print('\nACF(|r|) mechanism, lags 1-50 (draw 0 profiles for lag columns; bias/MAE over 15 units x 20 draws)'); print(MECH.round(4).to_string())
# ---- Part 3: bootstrap of the margin best generator - entry (positive = entry ahead), units resampled, B = 19,999; and units+draws
B = 19999; rng = np.random.default_rng(20260922); boot = []
for fr in ('A', 'B', 'C', 'P'):
    ents = FRAMES['A'] if fr == 'P' else FRAMES[fr]
    for sc in SC:
        g = R[(R.frame == fr) & (R.scenario == sc)]
        Cd = np.stack([np.stack([g[(g.market == m) & (g.seed == s) & (g.draw == k)].set_index('entry').composite.reindex(ents).to_numpy() for k in range(20)]) for m, s in UNITS])   # (15, 20, E)
        Cu = Cd.mean(1); ovr = Cu.mean(0); gi = [ents.index(x) for x in GENS]; best = gi[int(np.argmin(ovr[gi]))]
        idx = rng.integers(0, 15, (B, 15)); means = Cu[idx].mean(1)
        # two-level: units, then draws within each resampled unit (same draw index for all entries, as ranks are per draw)
        two = np.empty((B, len(ents)))
        for a in range(0, B, 1000):
            bsz = min(1000, B - a); ui = rng.integers(0, 15, (bsz, 15)); di = rng.integers(0, 20, (bsz, 15, 20)); two[a:a + bsz] = Cd[ui[:, :, None], di].mean(axis=(1, 2))
        for e in ents:
            if e in GENS: continue
            j = ents.index(e); mg = ovr[best] - ovr[j]; b1 = means[:, best] - means[:, j]; b2 = two[:, best] - two[:, j]; lo, hi = np.percentile(b1, [2.5, 97.5]); lo2, hi2 = np.percentile(b2, [2.5, 97.5])
            boot.append(dict(frame=fr, scenario=sc, entry=e, best_generator=ents[best], best_generator_composite=ovr[best], entry_composite=ovr[j], margin_best_minus_entry=mg, ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0 or hi < 0), p_entry_ahead=float((b1 > 0).mean()),
                             two_level_ci_lo=lo2, two_level_ci_hi=hi2, two_level_excludes_zero=bool(lo2 > 0 or hi2 < 0)))
BT = pd.DataFrame(boot); BT.to_csv(OUT2 / 'bootstrap_margins.csv', index=False); print('\nbootstrap margin (best generator - entry; positive = entry ahead)'); print(BT.round(4).to_string(index=False))
# ---- draw-to-draw spread (frame A and B)
sp = []
for fr in ('A', 'B', 'P'):
    for sc in SC:
        g = R[(R.frame == fr) & (R.scenario == sc)]; per = g.groupby(['draw', 'entry']).composite.mean().unstack('entry'); pos = per.rank(axis=1, method='min'); within = g.groupby(['market', 'seed', 'entry']).composite.std(ddof=1).groupby('entry').mean()
        for e in [x for x in per.columns if x in ('iid', 'sb_r', 'sb_abs', CTRL)]:
            sp.append(dict(frame=fr, scenario=sc, entry=e, overall_composite_mean=per[e].mean(), overall_composite_sd_over_draws=per[e].std(ddof=1), min=per[e].min(), max=per[e].max(), within_unit_sd_mean=within[e], position_min=int(pos[e].min()), position_max=int(pos[e].max()), position_mode=int(pos[e].mode().iloc[0]),
                           draws_ahead_of_all_generators=int((per[e] < per[GENS].min(axis=1)).sum()), draws=20))
SP = pd.DataFrame(sp); SP.to_csv(OUT2 / 'draw_to_draw_spread.csv', index=False); print('\ndraw-to-draw spread'); print(SP.round(3).to_string(index=False))
print('ANALYSE_DONE')
