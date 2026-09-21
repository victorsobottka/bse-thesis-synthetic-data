"""Part 2a: null-population mismatch. Recency-restricted versions of the exhaustive adjacent-sliding null.

A null built ONLY from test-period blocks at the same evaluation length needs two adjacent blocks of length L
inside a block of length L: zero positions. So the count is reported and the sensitivity is a null restricted to
positions whose two blocks both lie in the last M observations (M = 2L: the single valid|test pair, 3L, 4L, 6L, all).
Reads the stored null draws and the stored per-draw statistics; recomputes p-values only. No encoder, no regeneration.
"""
import sys, json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
RES = ROOT / 'thesis_results/production'; OUT = RES / 'overnight_2026-09-21'; REPORT = RES / 'overnight_report_2026-09-21.md'
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
MODELS = ['TimeGAN', 'GARCH', 'QuantGAN', 'GJR-GARCH', 'CNN-WGAN-GP', 'CONTROL: shuffled real']
SHORT = {'CONTROL: shuffled real': 'control'}
KEYS = ['rbf_x1', 'poly3']
mc_p = lambda obs, null: (1 + np.sum(null >= obs)) / (len(null) + 1)

def load_null(market):
    f = glob.glob(str(RES / market / 'ts2vec' / 'null_adjacent_L*.json'))[0]; d = json.load(open(f))
    return d['length'], d['series_length'], {k: np.array(v) for k, v in d['draws'].items()}

def restrict(null, L, N, M):
    """Positions whose blocks [s, s+L) and [s+L, s+2L) both lie in the last M observations: s >= N - M."""
    n = M - 2 * L + 1
    return None if n <= 0 else {k: v[-n:] for k, v in null.items()}

SETS = [('L (test block only)', 1), ('2L (valid+test)', 2), ('3L', 3), ('4L', 4), ('6L', 6), ('all (existing null)', None)]
units = {(m, s): json.load(open(RES / m / f'seed{s}' / 'embedding_distance_COMMIT_DRIFT.json')) for m in MARKETS for s in (42, 43, 44)}
null_rows, p_rows, ctrl_rows = [], [], []
for market in MARKETS:
    L, N, null = load_null(market)
    for label, mult in SETS:
        M = N if mult is None else mult * L
        nl = null if mult is None else restrict(null, L, N, M)
        n_pos = 0 if nl is None else len(nl['sigma'])
        row = dict(market=market, null=label, L=L, N=N, M_obs=int(M), positions=n_pos, approx_independent_pairs=round(M / (2 * L), 2))
        if nl is not None:
            for k in KEYS:
                row[f'{k}_mean'] = float(nl[k].mean()); row[f'{k}_sd'] = float(nl[k].std()); row[f'{k}_p97_5'] = float(np.percentile(nl[k], 97.5)); row[f'{k}_median'] = float(np.median(nl[k]))
            row['sigma_median'] = float(np.median(nl['sigma'])); row['sigma_min'] = float(nl['sigma'].min()); row['sigma_max'] = float(nl['sigma'].max())
        null_rows.append(row)
        for seed in (42, 43, 44):
            u = units[(market, seed)]
            for mod in MODELS:
                ds = u['results'][mod]['draws']
                for k in KEYS:
                    obs = np.array([d[k] for d in ds])
                    if nl is None: p = np.full(len(obs), np.nan)
                    else: p = np.array([mc_p(o, nl[k]) for o in obs])
                    p_rows.append(dict(market=market, seed=seed, model=mod, null=label, stat=k, n_draws=len(obs), positions=n_pos,
                                       share_p_le_05=float(np.mean(p <= 0.05)) if nl is not None else np.nan, mean_p=float(np.mean(p)) if nl is not None else np.nan,
                                       share_above_p97_5=float(np.mean(obs > row[f'{k}_p97_5'])) if nl is not None else np.nan,
                                       p_floor=1 / (n_pos + 1) if n_pos else np.nan))
NULL = pd.DataFrame(null_rows); P = pd.DataFrame(p_rows)
NULL.to_csv(OUT / 'part2a_null_by_recency.csv', index=False); P.to_csv(OUT / 'part2a_pvalues_by_recency.csv', index=False)

# -------------------------------------------------------------- summaries
labels = [s[0] for s in SETS]
def rates(stat):
    r = {}
    for lab in labels:
        sub = P[(P.null == lab) & (P.stat == stat)]
        r[lab] = {m: (sub[sub.model == m].share_p_le_05.mean(), sub[sub.model == m].share_above_p97_5.mean(), sub[sub.model == m].mean_p.mean()) for m in MODELS}
    return r
R1, R2 = rates('rbf_x1'), rates('poly3')
def ctrl_units(stat, lab):
    sub = P[(P.null == lab) & (P.stat == stat) & (P.model == 'CONTROL: shuffled real')]
    return int((sub.share_p_le_05 >= 0.5).sum()), int((sub.share_p_le_05 > 0).sum()), len(sub)
print(NULL[['market', 'null', 'positions', 'approx_independent_pairs', 'rbf_x1_mean', 'rbf_x1_sd', 'rbf_x1_p97_5', 'sigma_median']].round(4).to_string(index=False))
print()
for lab in labels:
    print(f'{lab:22s}', ' '.join(f'{SHORT.get(m, m)[:8]:>8s} {R1[lab][m][0]:5.1%}' for m in MODELS), '| control units >=half, any, of:', ctrl_units('rbf_x1', lab))

# -------------------------------------------------------------- report
md = ['', '## Part 2: null-population mismatch (2a: test-period and recency-restricted nulls)', '',
      'The mismatch: the exhaustive null has median σ 7.3–7.8 (median-heuristic bandwidth of the pooled pair of 20-year-sliding blocks), the real-versus-generator comparisons on the test block 2.7–4.8 (median 3.0–3.3). '
      'MMD² at different kernel bandwidths is not on one scale, so a p-value from that null answers a question about a different population.', '',
      '**Feasibility of the requested null.** A null from test-period blocks only, at the same evaluation length, needs two adjacent blocks of L observations (L = 486–501 here) inside the test period. The test block holds L observations, so the number of positions is **0**. '
      'Extending to validation+test (2L observations) gives **1** position (the valid|test pair itself). Neither can form a null. What follows is the sensitivity the request asked for in that case: the same exhaustive adjacent-sliding null restricted to positions whose two blocks both lie in the last M observations '
      '(M = 2L, 3L, 4L, 6L, all ≈ 10L). "Independent pairs" = M/2L, the number of non-overlapping block pairs that fit; the positions themselves overlap almost completely, so positions ≫ independent pairs and the p-value floor 1/(positions+1) is not a resolution the data support.', '',
      '| market | null (last M obs) | positions | independent pairs | mean MMD² | sd | p97.5 | median σ (min–max) |', '|---|---|---|---|---|---|---|---|']
for r in null_rows:
    if r['positions'] == 0: md.append(f"| {r['market']} | {r['null']} | 0 | {r['approx_independent_pairs']} | n/a | n/a | n/a | n/a |")
    else: md.append(f"| {r['market']} | {r['null']} | {r['positions']} | {r['approx_independent_pairs']} | {r['rbf_x1_mean']:.4f} | {r['rbf_x1_sd']:.4f} | {r['rbf_x1_p97_5']:.4f} | {r['sigma_median']:.2f} ({r['sigma_min']:.2f}–{r['sigma_max']:.2f}) |")
md += ['', 'RBF MMD² at the median-heuristic bandwidth (×1). The degree-3 columns are in `part2a_null_by_recency.csv`.', '',
       '**Where σ changes.** Median σ is ≈3.0–3.7 in the last 3L observations (the roughly six most recent years) and ≈6.5–7.6 as soon as one block reaches further back (see the 3L → 4L rows): the ≈7.5 is a property of blocks that contain older periods, and the test-period comparisons sit in the regime where σ ≈ 3. The mismatch is therefore real, and a same-scale null is one drawn from the recent end.', '',
       '**Recomputed p-values.** Share of the 300 draws (15 market-seeds × 20) per series with p ≤ .05, RBF at the median-heuristic bandwidth, under each null. "Existing" is the Task 5 v3 result.', '',
       '| null | ' + ' | '.join(SHORT.get(m, m) for m in MODELS) + ' | control units ≥ half of draws / any draw (of 15) |', '|---|' + '---|' * len(MODELS) + '---|']
for lab in labels:
    tag = lab + (' — not computable (0 positions)' if lab.startswith('L (') else ' — floor p = 0.5, cannot reach .05' if lab.startswith('2L') else '')
    if lab.startswith('L ('): md.append(f'| {tag} |' + ' n/a |' * len(MODELS) + ' n/a |'); continue
    a, b, n = ctrl_units('rbf_x1', lab)
    md.append(f'| {tag} | ' + ' | '.join(f'{R1[lab][m][0]:.1%}' for m in MODELS) + f' | {a} / {b} |')
md += ['', 'Share of draws above the null\'s own p97.5 (same nulls, RBF):', '', '| null | ' + ' | '.join(SHORT.get(m, m) for m in MODELS) + ' |', '|---|' + '---|' * len(MODELS)]
for lab in labels:
    if lab.startswith('L ('): continue
    md.append(f'| {lab} | ' + ' | '.join(f'{R1[lab][m][1]:.1%}' for m in MODELS) + ' |')
md += ['', 'Degree-3 polynomial, share of draws with p ≤ .05:', '', '| null | ' + ' | '.join(SHORT.get(m, m) for m in MODELS) + ' |', '|---|' + '---|' * len(MODELS)]
for lab in labels:
    if lab.startswith('L ('): continue
    md.append(f'| {lab} | ' + ' | '.join(f'{R2[lab][m][0]:.1%}' for m in MODELS) + ' |')
md.append('')
REPORT.open('a').write('\n'.join(md) + '\n')
json.dump(dict(feasibility=dict(test_block_positions=0, valid_plus_test_positions=1), null=null_rows, rates_rbf_x1={l: {m: dict(share_p_le_05=v[0], share_above_p97_5=v[1], mean_p=v[2]) for m, v in R1[l].items()} for l in labels},
          rates_poly3={l: {m: dict(share_p_le_05=v[0], share_above_p97_5=v[1], mean_p=v[2]) for m, v in R2[l].items()} for l in labels}),
          open(OUT / 'part2a_recency_null.json', 'w'), indent=1, default=str)
print('PART2A_DONE')
