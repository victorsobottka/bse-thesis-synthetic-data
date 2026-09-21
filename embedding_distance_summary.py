"""Aggregate the per-(market, seed) embedding_distance*.json files.

Reads thesis_results/production/<market>/seed<N>/embedding_distance_COMMIT_DRIFT.json
(or embedding_distance.json when no drift was accepted), prints the tables the study
needs and writes thesis_results/production/embedding_distance_summary<tag>.json.
Reads only; touches no Phase A output.

    python embedding_distance_summary.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / 'thesis_results' / 'production'
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']
CONTROL = 'CONTROL: shuffled real'
MULTS = ['0.25', '0.5', '1', '2', '4']
RBF = [f'rbf_x{c}' for c in MULTS]
STATS = RBF + ['poly3']

files = sorted(glob.glob(str(RES / '*' / 'seed*' / 'embedding_distance_COMMIT_DRIFT.json')) or
               glob.glob(str(RES / '*' / 'seed*' / 'embedding_distance.json')))
tag = '_COMMIT_DRIFT' if any('COMMIT_DRIFT' in f for f in files) else ''
J = [json.load(open(f)) for f in files]
print(f'{len(J)} units: ' + ', '.join(sorted({f"{j['market']}/{j['seed']}" for j in J})))
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)
out = {'units': len(J), 'note': 'summary of embedding_distance*.json; nothing here enters FIDELITY_COLS, TEMPORAL_COLS or the composite'}

# ---- per market: encoder, null, bandwidth
rows = []
for m in sorted({j['market'] for j in J}):
    js = [j for j in J if j['market'] == m]; j0 = js[0]
    sig = [r['sigma']['mean'] for j in js for r in j['results'].values()]
    rows.append(dict(market=m, epochs_run=j0['ts2vec']['training']['epochs_run'], best_epoch=j0['ts2vec']['training']['best_epoch'],
                     stop=j0['ts2vec']['training']['stop_reason'][:34], encoder_seconds=j0['ts2vec']['training']['seconds'],
                     L=j0['null']['length'], null_positions=j0['null']['n_positions_B'], indep_blocks=round(j0['null']['approx_independent_blocks'], 1),
                     null_sigma_median=round(j0['null']['stats']['sigma']['quantiles']['50'], 3),
                     sigma_real_vs_gen_min=round(min(sig), 3), sigma_real_vs_gen_max=round(max(sig), 3),
                     null_rbf1_mean=round(j0['null']['stats']['rbf_x1']['mean'], 4), null_rbf1_p97_5=round(j0['null']['stats']['rbf_x1']['quantiles']['97.5'], 4),
                     participation_ratio=round(j0['c_fid']['real_test_window_embedding_covariance']['participation_ratio'], 2),
                     n_eff=round(j0['c_fid']['real_test_window_embedding_covariance']['n_eff_from_autocorrelation'], 1)))
PM = pd.DataFrame(rows); print('\nPER MARKET: encoder, null and bandwidth'); print(PM.to_string(index=False))
out['per_market'] = rows

# ---- per model, mean over the 15 units of the per-draw mean and sd
recs = []
for j in J:
    for m, r in j['results'].items():
        recs.append(dict(market=j['market'], seed=j['seed'], model=m, sigma=r['sigma']['mean'],
                         **{f'{k}_mean': r[k]['mean'] for k in STATS}, **{f'{k}_sd': r[k]['sd'] for k in STATS},
                         **{f'p_{k}': r[f'p_{k}']['mean'] for k in STATS}, **{f'pfrac_{k}': r[k]['fraction_p_le_0.05'] for k in STATS},
                         **{f'above975_{k}': r[k]['exceeds_null_p97.5_fraction'] for k in STATS}))
R = pd.DataFrame(recs)
g = R.groupby('model')
tab = pd.DataFrame({'rbf_x1 mean': g.rbf_x1_mean.mean(), 'draw sd (mean)': g.rbf_x1_sd.mean(), 'CV over draws': g.rbf_x1_sd.mean() / g.rbf_x1_mean.mean(),
                    'p (mean)': g.p_rbf_x1.mean(), 'draws p<=.05': g.pfrac_rbf_x1.mean(), 'above null p97.5': g.above975_rbf_x1.mean(),
                    'poly3 mean': g.poly3_mean.mean(), 'poly3 draw sd': g.poly3_sd.mean()}).loc[MODELS + [CONTROL]]
print('\nPER MODEL over 15 market-seeds (RBF, median-heuristic bandwidth; per-draw statistic, mean and sd across the 20 draws)'); print(tab.round(4).to_string())
out['per_model_rbf_x1'] = tab.round(6).reset_index().to_dict('records')

# ---- bandwidth sweep: ordering stability
ident = [j['ordering_identical_across_bandwidth_sweep'] for j in J]
print(f"\nBANDWIDTH SWEEP (sigma x 0.25..4): identical model ordering across all 5 multipliers in {sum(ident)} of {len(J)} units")
sweep = []
for j in J:
    o = j['ordering_by_mean_statistic']
    sweep.append(dict(market=j['market'], seed=j['seed'], identical=j['ordering_identical_across_bandwidth_sweep'],
                      top_by_c={c: o[f'rbf_x{c}'][0] for c in MULTS}, bottom_by_c={c: o[f'rbf_x{c}'][-1] for c in MULTS},
                      min_tau_vs_x1=min(j['kendall_tau_vs_rbf_x1_ordering_models_only'][f'rbf_x{c}'] for c in MULTS),
                      poly3_tau=j['kendall_tau_vs_rbf_x1_ordering_models_only']['poly3']))
S = pd.DataFrame(sweep)
print(S[['market', 'seed', 'identical', 'min_tau_vs_x1', 'poly3_tau']].round(2).to_string(index=False))
top = pd.DataFrame({c: pd.Series([s['top_by_c'][c] for s in sweep]).value_counts() for c in MULTS}).fillna(0).astype(int)
print('\ncount of units in which each model has the SMALLEST MMD (closest to real), by bandwidth multiplier'); print(top.to_string())
bot = pd.DataFrame({c: pd.Series([s['bottom_by_c'][c] for s in sweep]).value_counts() for c in MULTS}).fillna(0).astype(int)
print('count of units in which each model has the LARGEST MMD, by bandwidth multiplier'); print(bot.to_string())
out['bandwidth_sweep'] = dict(identical_units=int(sum(ident)), units=len(J), per_unit=sweep, smallest_by_multiplier=top.to_dict(), largest_by_multiplier=bot.to_dict())

# ---- the control
print('\nCONTROL (a permutation of the real test block: identical marginal, no dynamics)')
C = []
for j in J:
    r = j['results'][CONTROL]; n = j['null']['stats']
    row = dict(market=j['market'], seed=j['seed'])
    for k in ('rbf_x1', 'poly3'):
        row[f'{k}_mean'] = r[k]['mean']; row[f'{k}_p'] = r[f'p_{k}']['mean']; row[f'{k}_pfrac'] = r[k]['fraction_p_le_0.05']
        row[f'{k}_null_p97.5'] = n[k]['quantiles']['97.5']; row[f'{k}_null_median'] = n[k]['quantiles']['50']
    row['best_model_rbf'] = min(j['results'][m]['rbf_x1']['mean'] for m in MODELS)
    row['n_models_farther_than_control_rbf'] = int(sum(j['results'][m]['rbf_x1']['mean'] > r['rbf_x1']['mean'] for m in MODELS))
    for c in MULTS:
        row[f'ctrl_p_rbf_x{c}'] = r[f'p_rbf_x{c}']['mean']; row[f'ctrl_pfrac_rbf_x{c}'] = r[f'rbf_x{c}']['fraction_p_le_0.05']
    C.append(row)
C = pd.DataFrame(C)
print(C[['market', 'seed', 'rbf_x1_mean', 'rbf_x1_null_median', 'rbf_x1_null_p97.5', 'rbf_x1_p', 'rbf_x1_pfrac', 'poly3_mean', 'poly3_p', 'poly3_pfrac',
         'best_model_rbf', 'n_models_farther_than_control_rbf']].round(4).to_string(index=False))
det = {k: int((C[f'ctrl_pfrac_{k}'] >= 0.5).sum()) for k in RBF}
print(f"\nunits (of {len(C)}) where the control's MMD is significant (p<=.05) in at least half of its 20 draws: " + ', '.join(f'{k}: {v}' for k, v in det.items()) + f", poly3: {int((C['poly3_pfrac'] >= 0.5).sum())}")
print(f"units where the control is CLOSER to real than the best generator (rbf, sigma x1): {int((C.rbf_x1_mean < C.best_model_rbf).sum())}; farther than all five: {int((C.n_models_farther_than_control_rbf == 0).sum())}")
out['control'] = dict(units=len(C), significant_in_half_of_draws_by_kernel=det, poly3=int((C['poly3_pfrac'] >= 0.5).sum()),
                      closer_than_best_generator=int((C.rbf_x1_mean < C.best_model_rbf).sum()), per_unit=C.round(6).to_dict('records'))

# ---- agreement with the published composite ranking
agree = []
for j in J:
    csv = pd.read_csv(RES / j['market'] / f"seed{j['seed']}" / f"{j['market']}_metrics.csv"); csv = csv[~csv.model.str.startswith('CONTROL')].set_index('model')
    m = {k: [j['results'][x][k]['mean'] for x in MODELS] for k in STATS}
    comp = [csv.loc[x, 'composite_rank'] for x in MODELS]
    agree.append({k: float(spearmanr(m[k], comp)[0]) for k in ('rbf_x1', 'poly3')})
A = pd.DataFrame(agree); print(f"\nSpearman between the models' mean MMD and their published composite_rank, over {len(A)} units: rbf_x1 mean {A.rbf_x1.mean():+.2f} (range {A.rbf_x1.min():+.2f}..{A.rbf_x1.max():+.2f}); poly3 mean {A.poly3.mean():+.2f}")
out['spearman_vs_published_composite'] = dict(rbf_x1_mean=float(A.rbf_x1.mean()), rbf_x1_min=float(A.rbf_x1.min()), rbf_x1_max=float(A.rbf_x1.max()), poly3_mean=float(A.poly3.mean()))
libs = J[0]['libraries']; print('\nlibraries:', libs)
out['libraries'] = libs
(RES / f'embedding_distance_summary{tag}.json').write_text(json.dumps(out, indent=1, default=str))
print('wrote', RES / f'embedding_distance_summary{tag}.json')
