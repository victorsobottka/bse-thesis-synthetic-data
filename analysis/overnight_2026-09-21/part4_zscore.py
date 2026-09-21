"""Part 4: is the |z|>3 cluster (BOVESPA/42: 11 of 57 GAN metrics) an artifact of the recorded draws' RNG history?

Recipe (identical to the BOVESPA/42 gate analysis): for each GAN (TimeGAN, QuantGAN, CNN-WGAN-GP) and each of the 19 metrics the
comparison CSV carries (composite_rank excluded), z = (recorded - mean of 20 fresh Phase B draws) / sd of those draws (ddof=0).
Reference distribution: each fresh draw in turn is treated as 'the recorded one' and scored against the other 19, giving the count
of |z|>3 cells among 57 that exchangeable draws produce (metrics are strongly correlated, so it is far from Poisson).
Reads only. Repo units: BOVESPA/42 (reference), FTSE/43, NIFTY50/44 (Phase B run in phase_b_venv into the pipeline's own directory).
Supplementary: all 15 market-seeds from the earlier scratch Phase B run (pandas 3.0.5; shown identical to the pinned env on BOVESPA/42).
"""
import sys, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
RES = ROOT / 'thesis_results/production'; OUT = RES / 'overnight_2026-09-21'; REPORT = RES / 'overnight_report_2026-09-21.md'
SCR = Path('/tmp/claude-1000/-home-sobottka-BSE-Master-Thesis-bse-thesis-synthetic-data/d811ca7f-e463-4bd9-934d-2554e3bfbe65/scratchpad/prod_all')
GANS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP']
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']

def load(market, seed, source):
    if source == 'repo':
        d = RES / market / f'seed{seed}' / 'phase_b_COMMIT_DRIFT'
        return pd.read_csv(d / f'{market}_metrics_comparison_COMMIT_DRIFT.csv'), pd.read_csv(d / f'{market}_metrics_phase_b_COMMIT_DRIFT.csv')
    d = SCR / market / f'seed{seed}' / 'phase_b'
    return pd.read_csv(d / f'{market}_metrics_comparison.csv'), pd.read_csv(d / f'{market}_metrics_phase_b.csv')

def analyse(market, seed, source):
    cmp_, pb = load(market, seed, source)
    g = cmp_[cmp_.model.isin(GANS) & (cmp_.metric != 'composite_rank')].copy()
    g['z'] = (g.recorded - g.phase_b_mean) / g.phase_b_sd
    metrics = list(g.metric.unique()); row = dict(market=market, seed=seed, source=source, cells=len(g))
    row['abs_z_gt3'] = int((g.z.abs() > 3).sum()); row['above_all_draws'] = int((g.recorded_percentile >= 100).sum()); row['below_all_draws'] = int((g.recorded_percentile <= 0).sum())
    row['outside_range'] = int((~g.recorded_within_draw_range.astype(bool)).sum())
    for m in GANS:
        gm = g[g.model == m]; row[f'z3_{m}'] = int((gm.z.abs() > 3).sum()); row[f'above_all_{m}'] = int((gm.recorded_percentile >= 100).sum())
    # leave-one-out reference
    loo, loo_by_model = [], {m: [] for m in GANS}
    X = {m: pb[pb.model == m][metrics].to_numpy(float) for m in GANS}; n = len(X[GANS[0]])
    for j in range(n):
        tot = 0
        for m in GANS:
            rest = np.delete(X[m], j, axis=0); mu, sd = rest.mean(0), rest.std(0)
            with np.errstate(divide='ignore', invalid='ignore'): z = (X[m][j] - mu) / sd
            c = int((np.abs(z) > 3).sum()); tot += c; loo_by_model[m].append(c)
        loo.append(tot)
    loo = np.array(loo); row['loo_mean'] = float(loo.mean()); row['loo_max'] = int(loo.max()); row['loo_share_ge_recorded'] = float((loo >= row['abs_z_gt3']).mean())
    for m in GANS: row[f'loo_mean_{m}'] = float(np.mean(loo_by_model[m])); row[f'loo_any_share_{m}'] = float(np.mean(np.array(loo_by_model[m]) > 0))
    big = g[g.z.abs() > 3][['model', 'metric', 'recorded', 'phase_b_mean', 'phase_b_sd', 'z', 'recorded_percentile']]
    return row, big

repo_units = [('BOVESPA', 42, 'repo'), ('FTSE', 43, 'repo'), ('NIFTY50', 44, 'repo')]
rows, bigs = [], {}
for m, s, src in repo_units:
    if not (RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT').exists(): print('missing', m, s); continue
    r, b = analyse(m, s, src); rows.append(r); bigs[(m, s)] = b
R = pd.DataFrame(rows); R.to_csv(OUT / 'part4_zscore_repo_units.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(R.round(3).to_string(index=False))
for k, b in bigs.items(): print('\n', k); print(b.round(4).to_string(index=False))
# supplementary: all 15 from the scratch run
sup = []
for m in MARKETS:
    for s in (42, 43, 44):
        if (SCR / m / f'seed{s}' / 'phase_b').exists(): sup.append(analyse(m, s, 'scratch')[0])
S = pd.DataFrame(sup); S.to_csv(OUT / 'part4_zscore_all15_scratch.csv', index=False)
print('\nALL 15 (scratch Phase B, pandas 3.0.5)'); print(S[['market', 'seed', 'abs_z_gt3', 'z3_TimeGAN', 'z3_QuantGAN', 'z3_CNN-WGAN-GP', 'above_all_draws', 'below_all_draws', 'outside_range', 'loo_mean', 'loo_max', 'loo_share_ge_recorded']].round(3).to_string(index=False))
print('\nTOTAL |z|>3:', int(S.abs_z_gt3.sum()), 'of', int(S.cells.sum()), '| LOO expected', round(S.loo_mean.sum(), 1), '| by model', {m: int(S[f'z3_{m}'].sum()) for m in GANS}, '| LOO by model', {m: round(S[f'loo_mean_{m}'].sum(), 1) for m in GANS})
print('above-all', int(S.above_all_draws.sum()), 'below-all', int(S.below_all_draws.sum()))

# ---------------------------------------------------------------- tilt test: are the recorded draws systematically off the fresh-draw distribution?
from scipy.stats import binomtest, chi2
pct = []
for m in MARKETS:
    for s in (42, 43, 44):
        c, _ = load(m, s, 'scratch'); g = c[c.model.isin(GANS) & (c.metric != 'composite_rank')]
        for mod, gm in g.groupby('model'): pct.append(dict(market=m, seed=s, model=mod, median_pct=float(gm.recorded_percentile.median()), mean_pct=float(gm.recorded_percentile.mean()), n_above_all=int((gm.recorded_percentile >= 100).sum()), n_below_all=int((gm.recorded_percentile <= 0).sum())))
PC = pd.DataFrame(pct); PC.to_csv(OUT / 'part4_recorded_percentiles_all15_scratch.csv', index=False)
print('\nRECORDED DRAW PERCENTILE within its 20 fresh draws (per unit x model: median over 19 metrics; 50 = typical, >50 = recorded worse-side)')
print(PC.groupby('model')[['median_pct', 'mean_pct']].agg(['mean', 'median']).round(1).to_string())
for lab, sub in [('all 45 unit-model pairs', PC), ('excluding BOVESPA/42', PC[~((PC.market == 'BOVESPA') & (PC.seed == 42))])]:
    a = int((sub.median_pct > 50).sum()); b = int((sub.median_pct < 50).sum())
    print(f'{lab}: n={len(sub)}, median percentile > 50 in {a}, < 50 in {b}, sign test p={binomtest(a, a + b, 0.5).pvalue:.3f}; mean of the medians {sub.median_pct.mean():.1f}')
print('cells above all 20 / below all 20: all 15 units', int(S.above_all_draws.sum()), '/', int(S.below_all_draws.sum()), '| excluding BOVESPA/42', int(S[~((S.market == 'BOVESPA') & (S.seed == 42))].above_all_draws.sum()), '/', int(S[~((S.market == 'BOVESPA') & (S.seed == 42))].below_all_draws.sum()), '| expected each side under exchangeability', round(855 / 21, 1), '(all) /', round(798 / 21, 1), '(excl.)')
p_units = (1 + (S.loo_share_ge_recorded * 20).round()) / 21
fisher = -2 * np.log(p_units).sum(); print('unit-level p (1+#LOO>=recorded)/21:', p_units.round(3).tolist(), '\nFisher combination chi2(%d) = %.1f, p = %.3f (approximate: discrete p-values, correlated metrics inside a unit)' % (2 * len(p_units), fisher, chi2.sf(fisher, 2 * len(p_units))))
print('units with unit-level p <= 0.10:', int((p_units <= 0.10).sum()), 'of', len(p_units), '| P(at least one unit with p <= 1/21 among 15, exchangeable) =', round(1 - (1 - 1 / 21) ** 15, 2))
