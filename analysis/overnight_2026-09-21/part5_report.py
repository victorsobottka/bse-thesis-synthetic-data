"""Part 5 (tables): the Task 5 v3 summary table for the main encoder and the two robustness configurations, from one function.
Main-encoder numbers are read from the 15 embedding_distance_COMMIT_DRIFT.json files; d320 / tsgbench from part5_<cfg>.json."""
import sys, json, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
RES = ROOT / 'thesis_results/production'; OUT = RES / 'overnight_2026-09-21'; REPORT = RES / 'overnight_report_2026-09-21.md'
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
ORDER = ['TimeGAN', 'GARCH', 'QuantGAN', 'GJR-GARCH', 'CNN-WGAN-GP', 'CONTROL: shuffled real']
SHORT = {'CONTROL: shuffled real': 'control'}; short = lambda m: SHORT.get(m, m)
CFGS = [c for c in ('main', 'd320', 'tsgbench') if c == 'main' or (OUT / f'part5_{c}.json').exists()]
LABEL = {'main': 'main run (d = 100, up to 300 epochs, patience 10)', 'd320': 'd = 320 (otherwise as main)', 'tsgbench': 'TSGBench literal (d = 100, batch 8, fit() default 600 iterations, no early stopping)'}

def load(cfg):
    U, N, meta = {}, {}, {}
    if cfg == 'main':
        fixed = pd.read_csv(OUT / 'part2c_fixed_sigma_observed.csv')
        for m in MARKETS:
            for s in (42, 43, 44):
                j = json.load(open(RES / m / f'seed{s}' / 'embedding_distance_COMMIT_DRIFT.json'))
                U[(m, s)] = {mod: dict(rbf=[d['rbf_x1'] for d in r['draws']], poly3=[d['poly3'] for d in r['draws']], p_rbf=[d['p_rbf_x1'] for d in r['draws']], p_poly3=[d['p_poly3'] for d in r['draws']],
                                       sigma=[d['sigma'] for d in r['draws']], fixed=fixed[(fixed.market == m) & (fixed.seed == s) & (fixed.model == mod)].sort_values('k').mmd2_fixed.tolist()) for mod, r in j['results'].items()}
            st = j['null']['stats']; N[m] = dict(rbf_mean=st['rbf_x1']['mean'], rbf_sd=st['rbf_x1']['sd'], rbf_p97_5=st['rbf_x1']['quantiles']['97.5'], poly3_p97_5=st['poly3']['quantiles']['97.5'], sigma_median=st['sigma']['quantiles']['50'])
            t = j['ts2vec']['training']; meta[m] = dict(d=100, iterations=f"{t['epochs_run']} epochs × {t['iterations_per_epoch']}", seconds=t['seconds'], final_loss=t['final_loss'], stop=t['stop_reason'])
        return U, N, meta
    J = json.load(open(OUT / f'part5_{cfg}.json'))
    for m, r in J['markets'].items():
        for s, v in r['units'].items(): U[(m, int(s))] = v
        N[m] = dict(rbf_mean=r['null']['rbf']['mean'], rbf_sd=r['null']['rbf']['sd'], rbf_p97_5=r['null']['rbf']['97.5'], poly3_p97_5=r['null']['poly3']['97.5'], sigma_median=r['null']['sigma_median'])
        t = r['training']
        meta[m] = dict(d=r['embedding_dim'], iterations=(f"{t['epochs_run']} epochs × {t['iterations_per_epoch']}" if 'epochs_run' in t else f"{t['iterations_run']} iterations (fit default {t['n_iters_default_applied']})"), seconds=t['seconds'],
                       final_loss=t.get('final_loss', t.get('final_epoch_mean_loss')), stop=t['stop_reason'])
    return U, N, meta

def table(cfg):
    U, N, meta = load(cfg); rows = []
    for mod in ORDER:
        a = dict(rbf=[], sd=[], p=[], above=[], poly=[], polyp=[], polyabove=[], sigma=[])
        for (m, s), v in U.items():
            r = np.array(v[mod]['rbf']); pr = np.array(v[mod]['p_rbf']); q = np.array(v[mod]['poly3']); pq = np.array(v[mod]['p_poly3'])
            a['rbf'].append(r.mean()); a['sd'].append(r.std()); a['p'].append((pr <= .05).mean()); a['above'].append((r > N[m]['rbf_p97_5']).mean()); a['poly'].append(q.mean()); a['polyp'].append((pq <= .05).mean()); a['sigma'].append(np.mean(v[mod]['sigma']))
        rows.append(dict(series=short(mod), mmd2=np.mean(a['rbf']), draw_sd=np.mean(a['sd']), share_p05=np.mean(a['p']), above_null_p975=np.mean(a['above']), poly3=np.mean(a['poly']), poly3_share_p05=np.mean(a['polyp']), sigma_obs=np.mean(a['sigma'])))
    # control rank, per unit, three statistics
    rk = {}
    for stat in ('rbf', 'poly3', 'fixed'):
        lowest, ranks = 0, []
        for (m, s), v in U.items():
            means = {mod: np.mean(v[mod][stat]) for mod in ORDER}; order = sorted(ORDER, key=means.get); ranks.append(order.index('CONTROL: shuffled real') + 1); lowest += order[0] == 'CONTROL: shuffled real'
        rk[stat] = dict(units_control_lowest=int(lowest), units=len(U), mean_rank=float(np.mean(ranks)))
    per_series_rank = {mod: np.mean([sorted(ORDER, key=lambda x: np.mean(v[x]['rbf'])).index(mod) + 1 for v in U.values()]) for mod in ORDER}
    return pd.DataFrame(rows), N, meta, rk, per_series_rank, U

md = ['', '## Part 5: encoder robustness (same 15 market-seeds, same 20 cached draws, same statistic; only the encoder changes)', '',
      'Two further encoder configurations were trained per market (5 markets; seed 42; GPU training is not bit-reproducible, so a retrain will not reproduce these encoders to the last digit — loss trajectories and encoder files are in `thesis_results/production/<market>/ts2vec_<config>/`). '
      'The effective-sample-size diagnostic is not repeated. Everything else is as in Task 5 v3: unbiased MMD², RBF at the per-pair median-heuristic bandwidth (primary), degree-3 polynomial (secondary), exhaustive adjacent-sliding null computed **with the encoder under test**, p = (1 + #{null ≥ obs})/(B + 1). '
      'Verified draw identity: the cached draws reproduce the main run\'s stored statistics to at most 8.6e-06 (Part 2c).', '',
      '**Encoders.**', '', '| config | market | embedding dim | training | seconds | final loss | stop |', '|---|---|---|---|---|---|---|']
res = {}
for cfg in CFGS:
    T, N, meta, rk, psr, U = table(cfg); res[cfg] = (T, N, meta, rk, psr)
    for m in MARKETS:
        x = meta[m]; md.append(f"| {cfg} | {m} | {x['d']} | {x['iterations']} | {x['seconds']:.0f} | {x['final_loss']:.4f} | {x['stop'][:44]} |")
md += ['', 'Note (tsgbench): the logged loss is the first-epoch mean; the run stops 121 iterations into the second epoch and that partial epoch is not logged by upstream `fit()`.', '']
for cfg in CFGS:
    T, N, meta, rk, psr = res[cfg][:5]
    md += [f'**{cfg}: {LABEL[cfg]}.** Mean over 15 market-seeds; MMD² is the per-unit mean over 20 draws, "draw sd" the per-unit sd over draws, averaged.', '',
           '| series | MMD² (RBF) | draw sd | draws p ≤ .05 | draws above null p97.5 | MMD² (deg 3) | draws p ≤ .05 (deg 3) | mean rank (1 = closest) |', '|---|---|---|---|---|---|---|---|']
    for _, r in T.iterrows():
        mod = [k for k in ORDER if short(k) == r.series][0]
        md.append(f"| {r.series} | {r.mmd2:.4f} | {r.draw_sd:.4f} | {r.share_p05:.1%} | {r.above_null_p975:.1%} | {r.poly3:.4f} | {r.poly3_share_p05:.1%} | {psr[mod]:.2f} |")
    md += ['', f"Control has the lowest MMD² of the six series in **{rk['rbf']['units_control_lowest']} of {rk['rbf']['units']}** units (RBF, mean rank {rk['rbf']['mean_rank']:.2f}); degree 3: {rk['poly3']['units_control_lowest']} of {rk['poly3']['units']} (mean rank {rk['poly3']['mean_rank']:.2f}); scale-matched RBF (σ* fixed, as Part 2c): {rk['fixed']['units_control_lowest']} of {rk['fixed']['units']} (mean rank {rk['fixed']['mean_rank']:.2f}).", '']
md += ['**Null geometry per encoder** (RBF, per-pair median heuristic): the mismatch that motivated Part 2 is a property of the encoder too.', '', '| config | market | null mean | null sd | null p97.5 | median σ, null | median σ, real-vs-generator |', '|---|---|---|---|---|---|---|']
for cfg in CFGS:
    U = table(cfg)[5]
    for m in MARKETS:
        n = res[cfg][1][m]; so = np.median([x for (mm, s), v in U.items() if mm == m for mod in ORDER for x in v[mod]['sigma']])
        md.append(f"| {cfg} | {m} | {n['rbf_mean']:.4f} | {n['rbf_sd']:.4f} | {n['rbf_p97_5']:.4f} | {n['sigma_median']:.2f} | {so:.2f} |")
md.append('')
if not os.environ.get('DRY'): REPORT.open('a').write('\n'.join(md) + '\n')
json.dump({cfg: dict(table=res[cfg][0].round(6).to_dict('records'), control_rank=res[cfg][3], mean_rank_by_series={short(k): v for k, v in res[cfg][4].items()}) for cfg in CFGS}, open(OUT / ('part5_summary_dry.json' if os.environ.get('DRY') else 'part5_summary.json'), 'w'), indent=1)
for cfg in CFGS:
    print('\n==', cfg); print(res[cfg][0].round(4).to_string(index=False)); print(res[cfg][3])
print('PART5_REPORT_WRITTEN', CFGS)
