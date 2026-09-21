"""Part 1 tables: embargo cost, NIFTY50 first (sub-0.5 draws under contiguous / shuffled / purged), all five markets (null and observed AUC by CV scheme at both lengths)."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
ns = load_fm(); FM = ns['FinancialMetrics']
SC = ('contiguous', 'purged', 'shuffled'); MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; CTRL = 'CONTROL: shuffled real'
out = {}; md = []
# ---------------------------------------------------------------- 1. embargo cost
cost = []
for m in MARKETS:
    full, test = load_full(m), load_test(m); N = len(full)
    for tag, L in (('track_a', len(test)), ('walk_forward', N // 6)):
        k = L - FM.AUC_WINDOW; folds = FM._purged_folds(k, FM.AUC_N_SPLITS, FM.AUC_WINDOW)
        cost.append(dict(market=m, length=tag, L=L, windows_per_class=k, embargo=FM.AUC_WINDOW, train_rows_contiguous=[2 * k - len(f[1]) for f in folds], train_rows_purged=[len(f[0]) for f in folds],
                         dropped_per_fold=[f[2] for f in folds], test_rows_per_fold=[len(f[1]) for f in folds], min_train_rows=min(len(f[0]) for f in folds), pct_dropped_min=min(100 * f[2] / (2 * k - len(f[1])) for f in folds), pct_dropped_max=max(100 * f[2] / (2 * k - len(f[1])) for f in folds)))
COST = pd.DataFrame(cost); COST.to_csv(OUT / 'part1_embargo_cost.csv', index=False)
print(COST[['market', 'length', 'L', 'windows_per_class', 'dropped_per_fold', 'min_train_rows', 'pct_dropped_min', 'pct_dropped_max']].round(1).to_string(index=False))
# ---------------------------------------------------------------- 2. nulls, all markets, three schemes
nrows = []
def summ(a): a = np.asarray(a); return dict(mean=a.mean(), sd=a.std(), p97_5=np.percentile(a, 97.5), mean_plus_1_96sd=a.mean() + 1.96 * a.std(), frac_lt_0_5=(a < 0.5).mean(), p2_5=np.percentile(a, 2.5), n=len(a))
meta = json.load(open(OUT / 'nulls' / 'sliding_meta.json'))
for m in MARKETS:
    for tag in ('track_a', 'walk_forward'):
        for sc in SC:
            a = np.load(OUT / 'nulls' / f'{m}_{tag}_{sc}.npy'); r = summ(a); r.update(market=m, length=tag, scheme=sc, null='full-series sliding', L=meta[f'{m}/{tag}']['L'], independent_blocks=meta[f'{m}/{tag}']['independent_blocks'], nan=int(np.isnan(a).sum())); nrows.append(r)
            s = np.load(OUT / 'nulls' / f'stationary_{m}_{tag}_{sc}.npy'); r = summ(s); r.update(market=m, length=tag, scheme=sc, null='stationary GARCH-t', L=meta[f'{m}/{tag}']['L'], independent_blocks=np.nan, nan=int(np.isnan(s).sum())); nrows.append(r)
NUL = pd.DataFrame(nrows); NUL.to_csv(OUT / 'part1_nulls_by_scheme.csv', index=False)
pd.set_option('display.width', 250)
print(NUL[NUL.null == 'full-series sliding'][['market', 'length', 'scheme', 'n', 'independent_blocks', 'mean', 'sd', 'p97_5', 'frac_lt_0_5']].round(3).to_string(index=False))
# ---------------------------------------------------------------- 3. NIFTY50 first
nif = []
x = load_full('NIFTY50')
for tag in ('track_a', 'walk_forward'):
    c, p, s = (np.load(OUT / 'nulls' / f'NIFTY50_{tag}_{sc}.npy') for sc in SC); low = c < 0.5
    nif.append(dict(construction=f'full-series sliding null, {tag}', n_positions=len(c), n_sub_0_5_contiguous=int(low.sum()), share_sub_0_5=float(low.mean()),
                    mean_contiguous=float(c[low].mean()), mean_shuffled=float(s[low].mean()), mean_purged=float(p[low].mean()),
                    share_above_0_5_shuffled=float((s[low] > 0.5).mean()), share_above_0_5_purged=float((p[low] > 0.5).mean()),
                    null_mean_contiguous=float(c.mean()), null_mean_shuffled=float(s.mean()), null_mean_purged=float(p.mean()), share_below_0_5_all_contiguous=float((c < .5).mean()), share_below_0_5_all_purged=float((p < .5).mean()), share_below_0_5_all_shuffled=float((s < .5).mean())))
# the quoted numbers: test-block adjacent pairs [0:n] vs [n:2n] whose contiguous AUC < 0.5 (Task 10 variant V1); purged infeasible for small n
D = pd.read_csv('/tmp/claude-1000/-home-sobottka-BSE-Master-Thesis-bse-thesis-synthetic-data/d811ca7f-e463-4bd9-934d-2554e3bfbe65/scratchpad/variants_detail.csv')
g = D[(D.market == 'NIFTY50') & D.variant.str.startswith('V1')]; ns_low = g[g.auc < 0.5].key.to_numpy(); t = load_test('NIFTY50')
res = {sc: np.array(pmap([(t[:n], t[n:2 * n], sc) for n in ns_low])) for sc in SC}
feas = ~np.isnan(res['purged'])
sub = lambda a, msk: float(np.mean(a[msk])) if msk.any() else float('nan')
nif.append(dict(construction='test-block pairs [0:n] vs [n:2n], contiguous AUC<0.5 (the 0.300 / 0.665 / 94% numbers), all draws', n_positions=len(g), n_sub_0_5_contiguous=len(ns_low), mean_contiguous=float(res['contiguous'].mean()), mean_shuffled=float(res['shuffled'].mean()),
                mean_purged=sub(res['purged'], feas), share_above_0_5_shuffled=float((res['shuffled'] > .5).mean()), share_above_0_5_purged=float((res['purged'][feas] > .5).mean()) if feas.any() else float('nan'), purged_feasible=int(feas.sum())))
nif.append(dict(construction='same, restricted to draws where purged CV is feasible', n_sub_0_5_contiguous=int(feas.sum()), mean_contiguous=sub(res['contiguous'], feas), mean_shuffled=sub(res['shuffled'], feas), mean_purged=sub(res['purged'], feas),
                share_above_0_5_shuffled=float((res['shuffled'][feas] > .5).mean()) if feas.any() else float('nan'), share_above_0_5_purged=float((res['purged'][feas] > .5).mean()) if feas.any() else float('nan'), min_n_feasible=int(ns_low[feas].min()) if feas.any() else None))
NIF = pd.DataFrame(nif); NIF.to_csv(OUT / 'part1_nifty50_sub_0_5.csv', index=False); print(NIF.round(3).T.to_string())
# ---------------------------------------------------------------- 4. observed AUC by scheme (Track A, fresh draws) and vs the nulls
OBS = pd.read_csv(OUT / 'observed_auc_by_draw.csv'); rows = []
for m in MARKETS:
    for mod in MODELS + [CTRL]:
        o = OBS[(OBS.market == m) & (OBS.model == mod)]; r = dict(market=m, model=mod, n=len(o))
        for sc in SC:
            n = NUL[(NUL.market == m) & (NUL.length == 'track_a') & (NUL.scheme == sc) & (NUL.null == 'full-series sliding')].iloc[0]; a = o[f'auc_{sc}'].mean()
            r.update({f'auc_{sc}': a, f'above_p97_5_{sc}': bool(a > n.p97_5), f'above_mean_1_96sd_{sc}': bool(a > n.mean_plus_1_96sd), f'draws_above_p97_5_{sc}': float((o[f'auc_{sc}'] > n.p97_5).mean())})
        rows.append(r)
OB = pd.DataFrame(rows); OB.to_csv(OUT / 'part1_observed_auc_by_scheme.csv', index=False)
gen = OB[OB.model != CTRL]
print('\nObserved (mean over 3 seeds x 20 draws) - 25 market-model cells above the same-scheme null:')
for sc in SC: print(f'  {sc:10s} above p97.5: {int(gen[f"above_p97_5_{sc}"].sum())}/25   above mean+1.96sd: {int(gen[f"above_mean_1_96sd_{sc}"].sum())}/25   mean AUC over cells {gen[f"auc_{sc}"].mean():.3f}   control mean {OB[OB.model==CTRL][f"auc_{sc}"].mean():.3f}')
# walk-forward observed (published, contiguous only)
wf = []
for m in MARKETS:
    for mod in MODELS:
        v = np.concatenate([pd.read_csv(RES / 'walk_forward' / m / f'walk_forward_{mod}_seed{s}.csv').discriminative_auc.to_numpy() for s in (42, 43, 44)])
        n = NUL[(NUL.market == m) & (NUL.length == 'walk_forward') & (NUL.scheme == 'contiguous') & (NUL.null == 'full-series sliding')].iloc[0]
        wf.append(dict(market=m, model=mod, wf_auc_contiguous_mean=float(np.nanmean(v)), n_folds=int(len(v)), above_p97_5=bool(np.nanmean(v) > n.p97_5), above_mean_1_96sd=bool(np.nanmean(v) > n.mean_plus_1_96sd), folds_above_p97_5=float(np.nanmean(v > n.p97_5))))
WF = pd.DataFrame(wf); WF.to_csv(OUT / 'part1_observed_walk_forward_contiguous.csv', index=False)
print('walk-forward (contiguous, published): cells above null p97.5:', int(WF.above_p97_5.sum()), '/25; above mean+1.96sd:', int(WF.above_mean_1_96sd.sum()), '/25')
print('PART1_REPORT_DONE')
