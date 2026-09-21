"""Part 1: recompute one market's auc_null.json through the notebook's own precompute_auc_null (fixed scaler; legacy + 999 half-split nulls; approved fixed-length adjacent
sliding null at both evaluation lengths, contiguous and purged). The old file is stale by the auc_scaler key. Usage: regen_auc_null.py <market>"""
import sys, time, io, contextlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import embedding_distance as ed
m = sys.argv[1]; t0 = time.time(); ns = ed.load_notebook()
real = ns['load_dataset'](ns['TEST_DIR'] / f'{m}_test.parquet').flatten()
ns['precompute_auc_null'](real, m, ns['RESULTS'] / m)
print(f'REGEN_DONE {m} {time.time()-t0:.0f}s', flush=True)
