"""Part 4 tables: leaky (PRE) vs fixed (POST) scaler."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness2 import *
T11 = RES / 'scenario_matrix' / 'nulls'; CT = 'CONTROL: shuffled real'; GENS5 = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']
def null(m, tag, cv, which): return np.load((T11 / f'{m}_{tag}_{cv}.npy') if which == 'pre' else (OUT / 'nulls' / f'{m}_{tag}_{cv}_fixed.npy'))
S = lambda a: dict(mean=a.mean(), sd=a.std(), p97_5=np.percentile(a, 97.5), thr196=a.mean() + 1.96 * a.std(), frac_lt_0_5=(a < .5).mean())
rows = []
for m in MARKETS:
    for tag in ('track_a', 'walk_forward'):
        for cv in ('contiguous', 'purged'):
            for w in ('pre', 'post'):
                a = null(m, tag, cv, w); r = S(a); r.update(market=m, length=tag, cv=cv, scaler=w, n=len(a)); rows.append(r)
N = pd.DataFrame(rows); N.to_csv(OUT / 'nulls_pre_post_summary.csv', index=False)
# largest change in the null values (same pairs)
chg = {(m, tag, cv): float(np.abs(null(m, tag, cv, 'post') - null(m, tag, cv, 'pre')).max()) for m in MARKETS for tag in ('track_a', 'walk_forward') for cv in ('contiguous', 'purged')}
mean_chg = {k: float((null(*k, 'post') - null(*k, 'pre')).mean()) for k in chg}
# observed, Frame F
O = pd.read_csv(OUT / 'observed_frameF_auc.csv'); P = pd.read_csv(OUT / 'published_subset_auc.csv')
def clear(df, cols_by, which, cv, models):
    out = []
    for m in MARKETS:
        n = N[(N.market == m) & (N.length == 'track_a') & (N.cv == cv) & (N.scaler == which)].iloc[0]
        for mod in models:
            v = df[(df.market == m) & (df.model == mod)][f'{which}_{cv}']; out.append(dict(market=m, model=mod, mean_auc=v.mean(), above_p97_5=bool(v.mean() > n.p97_5), above_196=bool(v.mean() > n.thr196), share_above_p97_5=float((v > n.p97_5).mean())))
    return pd.DataFrame(out)
res = {}; cl_all = []
for cv in ('contiguous', 'purged'):
    for w in ('pre', 'post'):
        c = clear(O, None, w, cv, GENS5); ctl = O[O.model == CT].groupby('market')[f'{w}_{cv}'].mean(); c['frame'] = 'F'; c['scaler'] = w; c['cv'] = cv; cl_all.append(c)
        sub = clear(P.assign(**{f'{w}_{cv}': P[f'{w}_{cv}']}), None, w, cv, ['TimeGAN', 'GARCH', 'GJR-GARCH']); sub['frame'] = 'published subset'; sub['scaler'] = w; sub['cv'] = cv; cl_all.append(sub)
        pc = P[P.model == CT].groupby('market')[f'{w}_{cv}'].mean()
        res[(cv, w)] = dict(F_cells=len(c), F_above_p97_5=int(c.above_p97_5.sum()), F_above_196=int(c.above_196.sum()), F_draw_share_above_p97_5=float(c.share_above_p97_5.mean()), F_mean_auc=float(c.mean_auc.mean()),
                            pub_cells=len(sub), pub_above_p97_5=int(sub.above_p97_5.sum()), pub_above_196=int(sub.above_196.sum()), F_control_range=(float(ctl.min()), float(ctl.max())), F_control_by_market=ctl.round(3).to_dict(), pub_control_range=(float(pc.min()), float(pc.max())), pub_control_by_market=pc.round(3).to_dict())
CL = pd.concat(cl_all); CL.to_csv(OUT / 'cells_vs_null_pre_post.csv', index=False)
chgs = {}
for cv in ('contiguous', 'purged'):
    chgs[f'observed_frameF_{cv}'] = float((O[f'post_{cv}'] - O[f'pre_{cv}']).abs().max()); chgs[f'observed_frameF_{cv}_mean_signed'] = float((O[f'post_{cv}'] - O[f'pre_{cv}']).mean()); chgs[f'observed_published_subset_{cv}'] = float((P[f'post_{cv}'] - P[f'pre_{cv}']).abs().max())
    chgs[f'null_max_abs_change_{cv}'] = max(v for k, v in chg.items() if k[2] == cv)
leg = json.load(open(OUT / 'legacy_999_null_pre_post.json'))
# NIFTY50 sub-0.5 draws, contiguous vs purged, PRE and POST (Task 11's audit statement)
nif = {}
for w in ('pre', 'post'):
    for tag in ('track_a', 'walk_forward'):
        c, p = null('NIFTY50', tag, 'contiguous', w), null('NIFTY50', tag, 'purged', w); low = c < .5; nif[f'{w}_{tag}'] = dict(n_sub=int(low.sum()), mean_contiguous=float(c[low].mean()), mean_purged=float(p[low].mean()), share_purged_above_0_5=float((p[low] > .5).mean()), share_null_below_0_5_contiguous=float(low.mean()), share_null_below_0_5_purged=float((p < .5).mean()))
json.dump(dict(cells={'|'.join(k): v for k, v in res.items()}, max_abs_change=chgs, null_max_abs_change_by_market={'|'.join(k): v for k, v in chg.items()}, null_mean_signed_change={'|'.join(k): v for k, v in mean_chg.items()}, legacy_999=leg, nifty50=nif), open(OUT / 'pre_post_summary.json', 'w'), indent=1, default=float)
pd.set_option('display.width', 220)
print(N[N.length == 'track_a'].pivot_table(index=['market', 'cv'], columns='scaler', values=['mean', 'sd', 'p97_5', 'frac_lt_0_5']).round(3).to_string()); print(N[N.length == 'walk_forward'].pivot_table(index=['market', 'cv'], columns='scaler', values=['mean', 'sd', 'p97_5']).round(3).to_string())
print(json.dumps(res, indent=1, default=float)); print(json.dumps(chgs, indent=1)); print(json.dumps(nif, indent=1)); print({m: (round(v['pre']['mean'], 3), round(v['post']['mean'], 3), round(v['pre']['p97_5'], 3), round(v['post']['p97_5'], 3), round(v['max_abs_change_all'], 3)) for m, v in leg.items()})
