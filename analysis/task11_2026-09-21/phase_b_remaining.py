"""Phase B, 20 fresh draws (pinned env), for the market-seeds that do not yet have <seed dir>/phase_b_COMMIT_DRIFT/ (drift accepted and stamped,
exactly as for BOVESPA/42, FTSE/43, NIFTY50/44). Phase A outputs are not touched. The default discriminative path is unchanged by the purged option, so
these must equal the earlier scratch runs cell for cell (checked in part2)."""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import embedding_distance as ed
ns = ed.load_notebook(); t0 = time.time()
for market in ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']:
    for seed in (42, 43, 44):
        if (ns['RESULTS'] / market / f'seed{seed}' / 'phase_b_COMMIT_DRIFT' / f'{market}_metrics_phase_b_COMMIT_DRIFT.csv').exists(): continue
        ns['run_evaluate_phase'](markets=[market], seeds=[seed], n_draws=20, accept_commit_drift=True); print(f'PHASE_B {market}/{seed} done ({time.time()-t0:.0f}s)', flush=True)
print('PHASE_B_ALL_DONE')
