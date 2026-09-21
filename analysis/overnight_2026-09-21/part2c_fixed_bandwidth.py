"""Part 2c: scale-matched null. The kernel bandwidth is fixed at sigma* (median of the market's observed real-vs-generator sigma) for BOTH
the null (every adjacent-sliding position, exhaustive) and the observed statistic, so both live on one kernel scale.
Observed statistics are recomputed from the cached Phase B draws; the per-draw median-heuristic statistic is recomputed too and checked
against the stored JSON (reproduction check). Encoders are loaded, not retrained.
"""
import sys, json, time, glob, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
import embedding_distance as ed
from ts2vec import TS2Vec
RES = ROOT / 'thesis_results/production'; OUT = RES / 'overnight_2026-09-21'; REPORT = RES / 'overnight_report_2026-09-21.md'
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
ORDER = ['TimeGAN', 'GARCH', 'QuantGAN', 'GJR-GARCH', 'CNN-WGAN-GP', ed.CONTROL]
SETS = [('2L (valid+test)', 2), ('3L', 3), ('4L', 4), ('6L', 6), ('all (exhaustive)', None)]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; ns = ed.load_notebook()
mc_p = ed.mc_p
t0 = time.time(); null_rows, obs_rows, repro = [], [], []
for market in MARKETS:
    parts, mu, sd = ed.load_market(market, ns); train, valid, test = parts['train'], parts['valid'], parts['test']
    full = np.concatenate([train, valid, test]); L, N = len(test), len(full); n_win = L - ed.WINDOW + 1
    z = lambda x: (np.asarray(x, dtype=np.float64) - mu) / sd
    enc = TS2Vec(input_dims=1, device=dev, **ed.TS2VEC); enc.load(str(RES / market / 'ts2vec' / 'encoder.pt'))
    E_full = ed.encode(enc, ed.windows(z(full))); E_real = E_full[len(train) + len(valid):]; assert len(E_real) == n_win
    stored = {s: json.load(open(RES / market / f'seed{s}' / 'embedding_distance_COMMIT_DRIFT.json')) for s in (42, 43, 44)}
    sig_all = np.array([d['sigma'] for s in stored.values() for r in s['results'].values() for d in r['draws']]); sigma_star = float(np.median(sig_all))
    # ---- null at sigma*: every position, exhaustive
    K = np.exp(-ed.sqdist(E_full, E_full) / (2 * sigma_star ** 2)); n_pos = N - 2 * L + 1
    null = np.array([ed.mmd2_u(K[s:s + n_win, s:s + n_win], K[s + L:s + L + n_win, s + L:s + L + n_win], K[s:s + n_win, s + L:s + L + n_win]) for s in range(n_pos)]); del K
    nulls = {lab: (null if m is None else null[-(m * L - 2 * L + 1):]) for lab, m in SETS}
    for lab, nl in nulls.items():
        null_rows.append(dict(market=market, null=lab, positions=len(nl), independent_pairs=round((N if lab.startswith('all') else dict(SETS)[lab] * L) / (2 * L), 2), sigma_star=sigma_star,
                              mean=float(nl.mean()), sd=float(nl.std()), p97_5=float(np.percentile(nl, 97.5)), median=float(np.median(nl))))
    # ---- observed at sigma*, plus reproduction of the stored per-draw statistic
    real_D2 = ed.sqdist(E_real, E_real); real_G = E_real @ E_real.T; d = E_real.shape[1]
    for seed in (42, 43, 44):
        arr = np.load(OUT / 'draws' / f'{market}_seed{seed}.npz')
        for mod in ORDER:
            for k in range(20):
                Es = ed.encode(enc, ed.windows(z(arr[f'{mod}__{k}']))); D2yy, D2xy = ed.sqdist(Es, Es), ed.sqdist(E_real, Es)
                st = ed.statistics(real_D2, D2yy, D2xy, real_G, Es @ Es.T, E_real @ Es.T, d)
                ref = stored[seed]['results'][mod]['draws'][k]
                repro.append(dict(market=market, seed=seed, model=mod, k=k, d_sigma=abs(st['sigma'] - ref['sigma']), d_rbf=abs(st['rbf_x1'] - ref['rbf_x1']), d_poly3=abs(st['poly3'] - ref['poly3'])))
                s2 = 2 * sigma_star ** 2; v = ed.mmd2_u(np.exp(-real_D2 / s2), np.exp(-D2yy / s2), np.exp(-D2xy / s2))
                row = dict(market=market, seed=seed, model=mod, k=k, mmd2_fixed=float(v), sigma_draw=st['sigma'], sigma_star=sigma_star)
                for lab, nl in nulls.items(): row[f'p_{lab}'] = mc_p(v, nl); row[f'above_p97_5_{lab}'] = float(v > np.percentile(nl, 97.5))
                obs_rows.append(row)
    print(f'{market}: sigma* {sigma_star:.3f}, null mean {null.mean():.4f} sd {null.std():.4f} p97.5 {np.percentile(null, 97.5):.4f} | {n_pos} positions ({time.time()-t0:.0f}s)', flush=True)
NULL, OBS, REP = pd.DataFrame(null_rows), pd.DataFrame(obs_rows), pd.DataFrame(repro)
NULL.to_csv(OUT / 'part2c_fixed_sigma_null.csv', index=False); OBS.to_csv(OUT / 'part2c_fixed_sigma_observed.csv', index=False); REP.to_csv(OUT / 'part2c_reproduction_check.csv', index=False)
labels = [s[0] for s in SETS]
rate = lambda lab, m: OBS[OBS.model == m].groupby(['market', 'seed'])[f'p_{lab}'].apply(lambda p: (p <= .05).mean()).mean()
above = lambda lab, m: OBS[OBS.model == m].groupby(['market', 'seed'])[f'above_p97_5_{lab}'].mean().mean()
def ctrl_units(lab):
    f = OBS[OBS.model == ed.CONTROL].groupby(['market', 'seed'])[f'p_{lab}'].apply(lambda p: (p <= .05).mean()); return int((f >= .5).sum()), int((f > 0).sum())
short = lambda m: 'control' if m == ed.CONTROL else m
md = ['', '### 2c: scale-matched null (bandwidth fixed at the observed σ)', '',
      'Both the null and the observed statistic use one Gaussian kernel with σ* = the market\'s median observed real-vs-generator σ (below), instead of a per-pair median heuristic. The null is the exhaustive adjacent-sliding null over all positions; the recency restrictions are as in 2a. '
      f'Observed statistics were recomputed from the cached Phase B draws; against the stored median-heuristic values the maximum absolute difference is {REP.d_rbf.max():.2e} (RBF), {REP.d_sigma.max():.2e} (σ), {REP.d_poly3.max():.2e} (degree 3), over {len(REP)} draws.', '',
      '| market | σ* | null | positions | independent pairs | mean MMD² | sd | p97.5 |', '|---|---|---|---|---|---|---|---|']
for r in null_rows: md.append(f"| {r['market']} | {r['sigma_star']:.2f} | {r['null']} | {r['positions']} | {r['independent_pairs']} | {r['mean']:.4f} | {r['sd']:.4f} | {r['p97_5']:.4f} |")
md += ['', 'Share of the 300 draws per series (15 market-seeds × 20) with p ≤ .05 under the scale-matched null:', '', '| null | ' + ' | '.join(short(m) for m in ORDER) + ' | control units ≥ half / any draw (of 15) |', '|---|' + '---|' * len(ORDER) + '---|']
for lab in labels:
    a, b = ctrl_units(lab)
    md.append(f'| {lab} | ' + ' | '.join(f'{rate(lab, m):.1%}' for m in ORDER) + f' | {a} / {b} |')
md += ['', 'Share above the null\'s own p97.5:', '', '| null | ' + ' | '.join(short(m) for m in ORDER) + ' |', '|---|' + '---|' * len(ORDER)]
for lab in labels: md.append(f'| {lab} | ' + ' | '.join(f'{above(lab, m):.1%}' for m in ORDER) + ' |')
mm = OBS.groupby('model').mmd2_fixed.mean()
md += ['', 'Mean observed MMD² at σ*, per series (mean over 15 market-seeds and 20 draws): ' + ', '.join(f'{short(m)} {mm[m]:.4f}' for m in ORDER) + '.', '']
REPORT.open('a').write('\n'.join(md) + '\n')
json.dump(dict(null=null_rows, reproduction=dict(max_abs_d_rbf=float(REP.d_rbf.max()), max_abs_d_sigma=float(REP.d_sigma.max()), max_abs_d_poly3=float(REP.d_poly3.max()), n=len(REP)),
                rates_p_le_05={lab: {short(m): float(rate(lab, m)) for m in ORDER} for lab in labels}, above_p97_5={lab: {short(m): float(above(lab, m)) for m in ORDER} for lab in labels},
                control_units={lab: dict(zip(['at_least_half_of_draws', 'any_draw'], ctrl_units(lab))) for lab in labels}, mean_observed_mmd2_at_sigma_star={short(m): float(mm[m]) for m in ORDER}),
          open(OUT / 'part2c_fixed_sigma.json', 'w'), indent=1)
print('PART2C_DONE', round(time.time() - t0), 's | reproduction max |d rbf| %.2e |d sigma| %.2e |d poly3| %.2e' % (REP.d_rbf.max(), REP.d_sigma.max(), REP.d_poly3.max()))
for lab in labels: print(f'{lab:18s}', ' '.join(f'{short(m)[:8]:>8s} {rate(lab, m):5.1%}' for m in ORDER), '| control units', ctrl_units(lab))
