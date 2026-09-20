"""Phase B correctness gate, TimeGAN anchor (BOVESPA seed 42).

TimeGAN draws its generation noise with torch.randn(...) on the CPU and then moves
it to the device, so the original series is a deterministic function of the CPU
generator's state at generate() time. That state is reproducible: the last torch
consumer before evaluation is CNN-WGAN-GP walk-forward fold 4, whose CPU-side
draws are (a) default module initialisation at construction and (b) two int64
draws per epoch from the DataLoader. Replaying those, then calling TimeGAN's
generate() twice, gives the two original draws: the first (whose metrics are in
BOVESPA_metrics.csv) and the second (stored in downstream_utility_inputs.npz).

Run from the repo root:  phase_b_venv/bin/python <this file>
"""
import json, io, contextlib, math, warnings, sys
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
nb = json.load(open('3_4_integrated_pipeline.ipynb')); ns = {'__name__': 'gate'}
skip = lambda s: s.lstrip().startswith('# ── GPU / device check') or 'This is the MAIN execution cell' in s
with contextlib.redirect_stdout(io.StringIO()):
    for c in nb['cells']:
        s = ''.join(c['source'])
        if c['cell_type'] == 'code' and not skip(s): exec(compile(s, '<nb>', 'exec'), ns)
P = Path('thesis_results/production/BOVESPA/seed42'); buf = io.StringIO()
train = ns['load_dataset']('data/processed_files/train/BOVESPA_train.parquet')
npz = np.load(P / 'downstream_utility_inputs.npz'); orig = npz['TimeGAN'].astype(np.float32).flatten(); real = npz['__real__'].astype(np.float32).flatten()
wf = pd.read_csv('thesis_results/production/walk_forward/BOVESPA/walk_forward_CNN-WGAN-GP_seed42.csv')
L4 = int(wf[wf.fold == 4].train_len.iloc[0]); nw = L4 - 128 + 1; nb_ = nw // 64; ne = max(1, math.ceil(9000 / nb_))
print(f'last CPU-RNG consumer before evaluation: CNN-WGAN-GP WF fold 4, train_len={L4}, {nw} windows, {nb_} batches/epoch, {ne} epochs')
with contextlib.redirect_stdout(buf): m = ns['reconstruct_model']('TimeGAN', ns['TimeGAN'], P / 'weights', train)
torch.manual_seed(42)
ns['CNN_WGAN_GP_Generator'](100, 128, 1, 32); ns['CNN_WGAN_GP_Critic'](128, 1, 32)
loader = DataLoader(TensorDataset(torch.zeros(nw, 128, 1)), 64, shuffle=True, drop_last=True)
for _ in range(ne):
    for _b in loader: pass
with contextlib.redirect_stdout(buf):
    d1 = m.generate(len(real)).astype(np.float32).flatten(); d2 = m.generate(len(real)).astype(np.float32).flatten()
diff = np.abs(d2 - orig)
print(f'draw #2 vs stored original: bit-identical={np.array_equal(d2, orig)}  corr={np.corrcoef(d2, orig)[0,1]:.6f}  max|diff|={diff.max():.3g}  median|diff|={np.median(diff):.3g}  series sd={orig.std():.4f}')
print(f'  share of points within 1e-5: {np.mean(diff < 1e-5):.2f} | within 1e-4: {np.mean(diff < 1e-4):.2f}')
print('  reference: best of 5,000 plain seeds (CPU or CUDA) reaches max|diff| = 0.0325')
null = ns['precompute_auc_null'](real, 'BOVESPA', 'thesis_results/production/BOVESPA')
with contextlib.redirect_stdout(buf): df = ns['evaluate_multiple_models'](real, {'TimeGAN': d1}, max_lag=min(50, len(real) // 4), auc_null=null)
rec = pd.read_csv(P / 'BOVESPA_metrics.csv'); a = rec[rec.model == 'TimeGAN'].iloc[0]; b = df[df.model == 'TimeGAN'].iloc[0]
cols = [c for c in rec.columns if c != 'model' and not c.endswith('_rank')]
rel = {c: abs(float(a[c]) - float(b[c])) / max(abs(float(a[c])), 1e-12) for c in cols if not np.isnan(float(a[c]))}
print('replayed draw #1 metrics vs the recorded TimeGAN row (relative difference):')
for c in sorted(rel, key=rel.get, reverse=True)[:6]: print(f'  {c:24s} recorded={float(a[c]):.6g} replayed={float(b[c]):.6g} rel diff={rel[c]:.2e}')
print(f'  median relative difference over {len(rel)} metrics = {np.median(list(rel.values())):.1e}')
