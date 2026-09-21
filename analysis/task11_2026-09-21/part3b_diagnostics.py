"""Supplementary measurements behind the scenario matrix: (1) the control's rank on every temporal metric in the 6-pool (which columns reward / punish it),
(2) single-draw noise floors for the rank-change counts, (3) paired scenario effects on the same draw."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
FID = ['mean_diff', 'std_diff', 'wasserstein', 'quantile_mse', 'tail_index_diff', 'extreme_events_diff', 'energy_distance']
T5 = ['acf_returns_mae', 'acf_absolute_mae', 'acf_squared_mae', 'hurst_diff', 'resid_kurtosis_diff']
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; CTRL = 'CONTROL: shuffled real'; SEEDS = (42, 43, 44)
OBS = pd.read_csv(OUT / 'observed_auc_by_draw.csv')
nul = {m: {k: (lambda a: (a.mean(), a.std()))(np.load(OUT / 'nulls' / f'{m}_track_a_{k}.npy')) for k in ('contiguous', 'purged')} for m in MARKETS}
leg = {m: (json.load(open(RES / m / 'auc_null.json'))['legacy_null']['mean'], json.load(open(RES / m / 'auc_null.json'))['legacy_null']['std']) for m in MARKETS}
# ---- (1) control per-metric rank, 6-pool, published draw (frame P) and fresh draws (frame F, S0 columns)
rk = {c: [] for c in FID + T5 + ['dist', 'absz']}; rkF = {c: [] for c in FID + T5 + ['dist_c', 'dist_p', 'absz_lm_c', 'absz_lm_p']}
for m in MARKETS:
    for s in SEEDS:
        df = pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv').set_index('model'); df['dist'] = df.discriminative_auc_dist; df['absz'] = df.discriminative_auc_absz
        for c in rk: rk[c].append(df[c].rank(na_option='bottom')[CTRL])
        pb = pd.read_csv(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / f'{m}_metrics_phase_b_COMMIT_DRIFT.csv'); ob = OBS[(OBS.market == m) & (OBS.seed == s)]
        for k, dk in pb.groupby('draw'):
            d = dk.set_index('model').copy(); o = ob[ob.draw == k].set_index('model')
            d['dist_c'] = (o.auc_contiguous - .5).abs(); d['dist_p'] = (o.auc_purged - .5).abs()
            d['absz_lm_c'] = (o.auc_contiguous - nul[m]['contiguous'][0]).abs() / nul[m]['contiguous'][1]; d['absz_lm_p'] = (o.auc_purged - nul[m]['purged'][0]).abs() / nul[m]['purged'][1]
            for c in rkF: rkF[c].append(d[c].rank(na_option='bottom')[CTRL])
CR = pd.DataFrame({'published_draw_mean_rank_of_control_of_6': {c: np.mean(v) for c, v in rk.items()}})
CRF = pd.DataFrame({'fresh_draws_mean_rank_of_control_of_6': {c: np.mean(v) for c, v in rkF.items()}})
CR.to_csv(OUT / 'control_metric_ranks_published.csv'); CRF.to_csv(OUT / 'control_metric_ranks_fresh.csv')
print(CR.round(2).T.to_string()); print(CRF.round(2).T.to_string())
# ---- (2)(3) noise floors from per-draw composites
F = pd.read_csv(OUT / 'draw_composites_frame_F.csv.gz'); F = F[F.model != CTRL]
pub = {}
for m in MARKETS:
    for s in SEEDS:
        df = pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv'); df = df[~df.model.str.startswith('CONTROL')].set_index('model'); pub[(m, s)] = df.composite_rank.reindex(MODELS).rank(method='min').to_numpy()
piv = F.pivot_table(index=['scenario', 'market', 'seed', 'draw'], columns='model', values='composite')[MODELS]
pos = piv.apply(lambda r: pd.Series(r.rank(method='min').to_numpy(), index=MODELS), axis=1)
rows = []
for scen in ['S0', 'S1', 'S2', 'S3', 'S4', 'S5']:
    ch_pub, ch_s0 = [], []
    for k in range(20):
        a = np.concatenate([pos.loc[(scen, m, s, k)].to_numpy() for m in MARKETS for s in SEEDS]); p = np.concatenate([pub[(m, s)] for m in MARKETS for s in SEEDS]); b = np.concatenate([pos.loc[('S0', m, s, k)].to_numpy() for m in MARKETS for s in SEEDS])
        ch_pub.append(int((a != p).sum())); ch_s0.append(int((a != b).sum()))
    rows.append(dict(scenario=scen, single_draw_position_changes_vs_published_mean=np.mean(ch_pub), min=min(ch_pub), max=max(ch_pub), same_draw_position_changes_vs_S0_mean=np.mean(ch_s0), same_draw_min=min(ch_s0), same_draw_max=max(ch_s0)))
# draw-to-draw noise floor: two different draws, same scenario S0
dd = []
for k in range(19):
    a = np.concatenate([pos.loc[('S0', m, s, k)].to_numpy() for m in MARKETS for s in SEEDS]); b = np.concatenate([pos.loc[('S0', m, s, k + 1)].to_numpy() for m in MARKETS for s in SEEDS]); dd.append(int((a != b).sum()))
NF = pd.DataFrame(rows); NF.to_csv(OUT / 'noise_floor_position_changes.csv', index=False); print(NF.round(1).to_string(index=False)); print('S0: position changes between two different fresh draws (adjacent k, k+1), of 75: mean %.1f range %d-%d' % (np.mean(dd), min(dd), max(dd)))
json.dump(dict(draw_to_draw_S0_mean=float(np.mean(dd)), draw_to_draw_S0_min=min(dd), draw_to_draw_S0_max=max(dd)), open(OUT / 'noise_floor_draw_to_draw.json', 'w'))
# ---- top-two under single draws: how often each model is first, per scenario (unit-draw level), across the 300 unit-draws
ov = piv.groupby(level=['scenario', 'draw']).mean()          # overall composite over the 15 units, per draw
first = ov.idxmin(axis=1).groupby(level='scenario').value_counts().unstack(fill_value=0); print('\nOverall winner across the 20 fresh draws (overall composite over 15 units, one draw each):'); print(first.to_string())
first.to_csv(OUT / 'winner_counts_by_draw.csv')
