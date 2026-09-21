"""Part 4 (data step): Phase B, 20 fresh draws, for two further market-seed pairs, written to the pipeline's own
<seed dir>/phase_b_COMMIT_DRIFT/ location (drift accepted and stamped, exactly as for BOVESPA/42). Chosen before looking: FTSE/43, NIFTY50/44."""
import sys, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import pandas as pd, embedding_distance as ed
ns = ed.load_notebook(); print('pandas', pd.__version__, '| results root', ns['RESULTS'], flush=True)
for market, seed in [('FTSE', 43), ('NIFTY50', 44)]:
    t = time.time(); ns['run_evaluate_phase'](markets=[market], seeds=[seed], n_draws=20, accept_commit_drift=True)
    print(f'PHASE_B_DONE {market}/{seed} {time.time()-t:.0f}s', flush=True)
