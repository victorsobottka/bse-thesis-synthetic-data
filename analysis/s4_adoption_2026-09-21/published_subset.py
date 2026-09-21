"""The reproducible subset of the PUBLISHED draw: GARCH and GJR-GARCH (deterministic; the series saved in downstream_utility_inputs.npz is the series the metrics used, float32 storage),
the control (default_rng(42) permutation of the test block) and TimeGAN (CPU-RNG replay of the first generate() call after the RNG history, as in the Task 6 replay). QuantGAN and
CNN-WGAN-GP cannot be reproduced (CUDA RNG state never saved). For each unit: PRE-fix AUC (contiguous) must equal the published discriminative_auc_raw; then PRE/POST x contiguous/purged."""
import sys, math, io, contextlib
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness2 import *
import torch
from torch.utils.data import DataLoader, TensorDataset
pre_ns = load_ns(PRE_NB); post_ns = load_ns(POST_NB); PRE = pre_ns['FinancialMetrics']; POST = post_ns['FinancialMetrics']; ns = post_ns
seq_len = 128; rows = []; series = {}
for m in MARKETS:
    train = ns['load_dataset'](ns['DATA'] / 'train' / f'{m}_train.parquet') if 'DATA' in ns else ns['load_dataset'](ROOT / f'data/processed_files/train/{m}_train.parquet')
    for s in (42, 43, 44):
        P = RES / m / f'seed{s}'; npz = np.load(P / 'downstream_utility_inputs.npz'); real = npz['__real__'].astype(np.float32).flatten(); pub = pd.read_csv(P / f'{m}_metrics.csv').set_index('model')
        wf = pd.read_csv(RES / 'walk_forward' / m / f'walk_forward_CNN-WGAN-GP_seed{s}.csv'); L4 = int(wf[wf.fold == 4].train_len.iloc[0]); n_windows = L4 - seq_len + 1; n_batches = n_windows // 64; n_epochs = max(1, math.ceil(9000 / n_batches))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf): mdl = ns['reconstruct_model']('TimeGAN', ns['TimeGAN'], P / 'weights', train)
        torch.manual_seed(s); ns['CNN_WGAN_GP_Generator'](100, seq_len, 1, 32); ns['CNN_WGAN_GP_Critic'](seq_len, 1, 32)
        loader = DataLoader(TensorDataset(torch.zeros(n_windows, seq_len, 1)), 64, shuffle=True, drop_last=True)
        for _ in range(n_epochs):
            for _b in loader: pass
        with contextlib.redirect_stdout(buf): d1 = mdl.generate(len(real)).astype(np.float32).flatten(); d2 = mdl.generate(len(real)).astype(np.float32).flatten()
        orig = npz['TimeGAN'].astype(np.float32).flatten()
        S = {'GARCH': npz['GARCH'].astype(np.float32).flatten(), 'GJR-GARCH': npz['GJR-GARCH'].astype(np.float32).flatten(), 'TimeGAN': d1, 'CONTROL: shuffled real': np.random.default_rng(42).permutation(real).copy()}
        for name, x in S.items():
            r = dict(market=m, seed=s, model=name, published_auc_raw=float(pub.loc[name, 'discriminative_auc_raw']), replay_vs_stored_second_draw_maxdiff=float(np.abs(d2 - orig).max()) if name == 'TimeGAN' else np.nan)
            for which, FM in (('pre', PRE), ('post', POST)):
                for cv in ('contiguous', 'purged'): r[f'{which}_{cv}'] = float(FM(real, x).compute_discriminative_score(cv=cv)['auc'])
            rows.append(r)
        print(m, s, 'TimeGAN replay: max |2nd draw - stored| %.2e' % np.abs(d2 - orig).max(), flush=True)
D = pd.DataFrame(rows); D['pre_minus_published'] = D.pre_contiguous - D.published_auc_raw; D.to_csv(OUT / 'published_subset_auc.csv', index=False)
print(D.groupby('model').agg(units=('market', 'size'), max_abs_pre_minus_published=('pre_minus_published', lambda v: v.abs().max())).to_string())
print('replay max |diff| to the stored second draw over 15 units:', D.replay_vs_stored_second_draw_maxdiff.max())
