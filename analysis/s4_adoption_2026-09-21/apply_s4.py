"""S4 adoption (Victor's decision, 2026-09-21): discriminative_auc_dist and discriminative_auc_absz leave the ranked set.
Recomputes temporal_rank, composite_rank and avg_rank in the published CSVs from the EXISTING per-metric values (no regeneration; none of the twelve
remaining ranked metrics depends on the AUC). fidelity_rank must be unchanged. The two AUC rank columns are dropped (the notebook no longer produces them
for descriptive columns). Files rewritten: {market}_metrics.csv, overall_summary.csv, model_performance_summary.csv per market-seed, and the two root
aggregates, which the pipeline derives from them (report_data.py reads them and 'never recomputes a rank').

    python apply_s4.py validate   # in memory only: reproduces the reference table or exits 1
    python apply_s4.py write      # runs validate first, then writes (backups must exist)
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
RES = ROOT / 'thesis_results/production'; MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']; SEEDS = (42, 43, 44)
FID = ['mean_diff', 'std_diff', 'wasserstein', 'quantile_mse', 'tail_index_diff', 'extreme_events_diff', 'energy_distance']
T5 = ['acf_returns_mae', 'acf_absolute_mae', 'acf_squared_mae', 'hurst_diff', 'resid_kurtosis_diff']
DROP = ['discriminative_auc_dist_rank', 'discriminative_auc_absz_rank']; CTRL = 'CONTROL'
REF = {'GJR-GARCH': (2.667, 2.320, 2.493), 'CNN-WGAN-GP': (2.429, 2.813, 2.621), 'GARCH': (3.152, 2.693, 2.923), 'QuantGAN': (3.138, 3.053, 3.096), 'TimeGAN': (3.614, 4.120, 3.867)}
KEY = ['composite_rank', 'fidelity_rank', 'temporal_rank', 'avg_rank', 'wasserstein', 'energy_distance', 'quantile_mse', 'acf_absolute_mae', 'tail_index_diff', 'hurst_diff', 'resid_kurtosis_diff', 'discriminative_auc_dist']

def apply_s4(df):
    """df: a metrics-style table (raw metrics + rank columns), control row(s) with NaN ranks. Returns the S4 table and the largest change in fidelity_rank."""
    d = df.copy(); gen = ~d.model.str.contains(CTRL); g = d[gen]
    f = pd.concat([g[c].rank(na_option='bottom') for c in FID], axis=1).mean(axis=1); t = pd.concat([g[c].rank(na_option='bottom') for c in T5], axis=1).mean(axis=1)
    dfid = float((f - d.loc[gen, 'fidelity_rank']).abs().max())
    # per-metric ranks already stored must equal the recomputed ones (raw values reproduce the published ranks)
    dr = max(float((g[c].rank(na_option='bottom') - d.loc[gen, f'{c}_rank']).abs().max()) for c in FID + T5)
    d = d.drop(columns=[c for c in DROP if c in d.columns])
    d.loc[gen, 'temporal_rank'] = t; d.loc[gen, 'composite_rank'] = (d.loc[gen, 'fidelity_rank'] + t) / 2
    rk = [c for c in d.columns if c.endswith('_rank') and c not in ('fidelity_rank', 'temporal_rank', 'composite_rank')]
    d.loc[gen, 'avg_rank'] = d.loc[gen, rk].mean(axis=1)
    return d, dfid, dr

def load_units():
    return {(m, s): pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv') for m in MARKETS for s in SEEDS}

def validate():
    U = load_units(); new, worst_f, worst_r = {}, 0.0, 0.0
    for u, df in U.items():
        n, dfid, dr = apply_s4(df); new[u] = n; worst_f = max(worst_f, dfid); worst_r = max(worst_r, dr)
    gens = [m for m in REF]; tab = pd.DataFrame({m: {'fidelity': np.mean([new[u].set_index('model').loc[m, 'fidelity_rank'] for u in new]), 'temporal': np.mean([new[u].set_index('model').loc[m, 'temporal_rank'] for u in new]),
                                                     'composite': np.mean([new[u].set_index('model').loc[m, 'composite_rank'] for u in new])} for m in gens}).T
    ok = all(round(tab.loc[m, 'fidelity'], 3) == REF[m][0] and round(tab.loc[m, 'temporal'], 3) == REF[m][1] and round(tab.loc[m, 'composite'], 3) == REF[m][2] for m in gens)
    print('fidelity_rank recomputed vs stored, max |difference| over 75 evaluations: %.3g; stored per-metric ranks vs recomputed: %.3g' % (worst_f, worst_r))
    print(tab.round(4).to_string()); print('reference:', REF); print('EXACT MATCH to 3 decimals on fidelity, temporal and composite for all five models:', ok)
    C = np.array([[new[u].set_index('model').loc[m, 'composite_rank'] for m in gens] for u in new]); order = tab.composite.sort_values().index.tolist(); a, b = gens.index(order[0]), gens.index(order[1])
    print('first', order[0], 'second', order[1], 'margin %.4f' % (tab.loc[order[1], 'composite'] - tab.loc[order[0], 'composite']), '| first ahead of second in', int((C[:, a] < C[:, b]).sum()), 'of 15 units (ties', int((C[:, a] == C[:, b]).sum()), ')')
    rng = np.random.default_rng(20260921); idx = rng.integers(0, 15, (19999, 15)); mg = C[idx].mean(1)[:, b] - C[idx].mean(1)[:, a]; lo, hi = np.percentile(mg, [2.5, 97.5])
    print('bootstrap 19,999 units, margin second-first 95%% interval [%+.3f, %+.3f]; excludes zero: %s' % (lo, hi, bool(lo > 0)))
    return ok and worst_f < 1e-12 and worst_r == 0, new

def aggregate(per_seed_files):
    """The pipeline's own aggregation (run_complete_pipeline), applied to the per-market-seed model_performance_summary.csv files."""
    combined = pd.concat([pd.read_csv(p, index_col=0).reset_index() for p in per_seed_files], ignore_index=True)
    available = [m for m in KEY if m in combined.columns]
    overall = combined.groupby('model')[available].mean().round(4).sort_values('composite_rank')
    return overall, combined

if __name__ == '__main__':
    mode = sys.argv[1]
    ok, new = validate()
    if mode == 'validate':
        sys.exit(0 if ok else 1)
    if not ok:
        print('VALIDATION FAILED: nothing written'); sys.exit(1)
    # aggregator regression check on the CURRENT files before anything is written
    pf = [RES / m / f'seed{s}' / 'model_performance_summary.csv' for m in MARKETS for s in SEEDS]
    ov0, cb0 = aggregate(pf)
    import io
    assert ov0.to_csv() == (RES / 'overall_performance.csv').read_text() and cb0.to_csv(index=False) == (RES / 'per_seed_market_performance.csv').read_text(), 'aggregator does not reproduce the current root files'
    print('aggregator reproduces the current overall_performance.csv and per_seed_market_performance.csv byte for byte')
    flag = '--flag-pre-scaler-fix' in sys.argv
    for (m, s), n in new.items():
        d = RES / m / f'seed{s}'
        if flag: n['discriminative_auc_pre_scaler_fix'] = True
        n.to_csv(d / f'{m}_metrics.csv', index=False)
        o = pd.read_csv(d / 'overall_summary.csv'); on, dfid, dr = apply_s4(o); assert dfid < 1e-12 and dr == 0, (m, s, dfid, dr)
        if flag: on['discriminative_auc_pre_scaler_fix'] = True
        on.to_csv(d / 'overall_summary.csv', index=False)
        mp = pd.read_csv(d / 'model_performance_summary.csv'); nn = n.set_index('model')
        for c in ('temporal_rank', 'composite_rank', 'avg_rank'): mp[c] = [round(float(nn.loc[x, c]), 4) if x in nn.index else np.nan for x in mp.model]
        assert np.allclose([round(float(nn.loc[x, 'fidelity_rank']), 4) for x in mp.model if not str(x).startswith('CONTROL')], mp.loc[~mp.model.str.startswith('CONTROL'), 'fidelity_rank'])
        mp.to_csv(d / 'model_performance_summary.csv', index=False)
    ov, cb = aggregate(pf); ov.to_csv(RES / 'overall_performance.csv'); cb.to_csv(RES / 'per_seed_market_performance.csv', index=False)
    print('WRITTEN: 15 x 3 per-market-seed CSVs + overall_performance.csv + per_seed_market_performance.csv'); print(ov.round(4).to_string())
