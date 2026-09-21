"""Learned-representation two-sample distance for the generators: a Phase B job.

Question: does the permutation-invariance critique of the hand-crafted metrics
survive against learned-representation metrics? Regenerates synthetic series from
the saved weights (no generator retraining), embeds every 128-step window with a
TS2Vec encoder trained on the market's training block, and computes an unbiased
kernel two-sample statistic (MMD^2) between the real test windows and each
synthetic series. Nothing here touches {market}_metrics.csv, FIDELITY_COLS,
TEMPORAL_COLS or the composite.

Run from the repo root, in the pinned environment:
    phase_b_venv/bin/python embedding_distance.py [--markets BOVESPA ...] [--seeds 42 ...]
                                                  [--draws 20] [--max-epochs 300]

Kernels
-------
Primary: Gaussian RBF, k(a, b) = exp(-||a - b||^2 / (2 sigma^2)).
  * It is bounded by 1, so no single embedding dominates the sum. KID's degree-3
    polynomial default is unbounded and outlier-dominated: on returns with Hill
    alpha between 2.54 and 3.21 that would reproduce the failure mode this study
    documents in the hand-crafted metrics.
  * It is characteristic, so MMD = 0 if and only if the two distributions are
    identical. A degree-3 kernel cannot guarantee that beyond third moments, and
    the permutation-control question depends on it.
Secondary (reporting only, for comparability with the image and TSGBench
literature): polynomial degree 3, k(a, b) = (a.b / d + 1)^3.
Bandwidth: median heuristic, sigma = median pairwise Euclidean distance over the
pooled real + synthetic embeddings of the pair, reported per market and per draw;
then a sweep at sigma x {0.25, 0.5, 1, 2, 4}. Split-data power maximisation is not
used: it costs a data split that n ~ 369 cannot afford.

The null
--------
128-step windows at stride 1 share 127 of 128 observations, so MMD's asymptotic
null is invalid. It is calibrated empirically with the fixed-length adjacent
sliding construction: for every start position s of the full real series (train +
valid + test), MMD between the windows of [s, s+L) and [s+L, s+2L), L being the
evaluation length. Exhaustive (stride 1), not the legacy random cut. Full
enumeration is NOT independence: 2L-long blocks give about N/(2L) ~ 5 independent
blocks, so the calibration fixes resolution, not independence. p = (1 + #{null >=
observed}) / (B + 1).

C-FID (Fréchet distance on the same embeddings) is NOT computed. 369 stride-1
windows carry ~4 independent observations of a 100-320 dimensional distribution
(participation ratio ~5; see `c_fid` in the output), so the Fréchet form is not
estimable at this sample size, and publishing a number would repeat the failure
this study documents.
"""
import argparse, contextlib, io, json, math, random, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'ts2vec'))
from ts2vec import TS2Vec                                   # vendored, unmodified
from models.losses import hierarchical_contrastive_loss     # noqa: E402
from utils import take_per_row                              # noqa: E402

WINDOW = 128                                                # = seq_len of every gradient model
BANDWIDTH_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']
CONTROL = 'CONTROL: shuffled real'
# TS2Vec. Output dimension, batch size, learning rate and max_train_length are TSGBench's
# (src/ts2vec.py: output_dims=100, batch_size=8, lr=0.001, max_train_length=3000). TSGBench's
# code calls fit() with no n_epochs, which is TS2Vec's default of 200 iterations (600 for
# large data), and has no early stopping. The epoch cap and patience below are this job's
# choice, not TSGBench's: up to 300 epochs, stop after 10 epochs without a new best epoch loss.
TS2VEC = dict(output_dims=100, hidden_dims=64, depth=10, lr=1e-3, batch_size=8,
              max_train_length=3000, temporal_unit=0)
TRAIN = dict(max_epochs=300, patience=10, seed=42)


# ----------------------------------------------------------------------------- notebook definitions
def load_notebook():
    """Execute the notebook's definition cells (never the run) to reuse Phase B's own
    reconstruct_model / _regenerate / _draw_seed / phase_b_preflight, so the models
    are rebuilt by the same code Phase B uses."""
    nb = json.load(open(ROOT / '3_4_integrated_pipeline.ipynb'))
    ns = {'__name__': 'embedding_distance_nb'}
    skip = lambda s: s.lstrip().startswith('# ── GPU / device check') or 'This is the MAIN execution cell' in s
    with contextlib.redirect_stdout(io.StringIO()):
        for c in nb['cells']:
            s = ''.join(c['source'])
            if c['cell_type'] == 'code' and not skip(s):
                exec(compile(s, '<notebook>', 'exec'), ns)
    return ns


def library_versions():
    import arch, scipy, sklearn, statsmodels
    return {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'statsmodels': statsmodels.__version__, 'arch': arch.__version__,
            'cuda': torch.cuda.is_available(),
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}


# ----------------------------------------------------------------------------- data
def windows(x):
    """Stride-1 windows of length WINDOW, shape (n, WINDOW, 1) float32."""
    return np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), WINDOW)[:, :, None].astype(np.float32)


def load_market(market, ns):
    parts = {s: ns['load_dataset'](ns['DATA'] / s / f"{market}_{s}.parquet").flatten().astype(np.float64)
             for s in ('train', 'valid', 'test')}
    meta = json.loads((ns['RESULTS'] / market / 'seed42' / 'weights' / 'TimeGAN_meta.json').read_text())
    mu = float(np.array(meta['data_mean']['values']).ravel()[0])
    sd = float(np.array(meta['data_std']['values']).ravel()[0])
    return parts, mu, sd


# ----------------------------------------------------------------------------- encoder
def train_encoder(train_windows, device, max_epochs, patience, seed, log=print):
    """TS2Vec training with epoch patience. The loop is upstream's fit() (random crops,
    hierarchical contrastive loss, AdamW, SWA-averaged eval network) with one addition:
    stop after `patience` epochs without a new best epoch-mean loss."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = TS2Vec(input_dims=1, device=device, **TS2VEC)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(train_windows).float())
    loader = torch.utils.data.DataLoader(ds, batch_size=min(TS2VEC['batch_size'], len(ds)), shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model._net.parameters(), lr=TS2VEC['lr'])
    losses, best, best_epoch, wait, t0 = [], math.inf, -1, 0, time.time()
    stop = f'max_epochs={max_epochs}'
    for epoch in range(max_epochs):
        cum, n = 0.0, 0
        for (x,) in loader:
            x = x.to(device)
            ts_l = x.size(1)
            crop_l = np.random.randint(low=2 ** (TS2VEC['temporal_unit'] + 1), high=ts_l + 1)
            crop_left = np.random.randint(ts_l - crop_l + 1)
            crop_right = crop_left + crop_l
            crop_eleft = np.random.randint(crop_left + 1)
            crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
            crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
            opt.zero_grad()
            out1 = model._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))[:, -crop_l:]
            out2 = model._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))[:, :crop_l]
            loss = hierarchical_contrastive_loss(out1, out2, temporal_unit=TS2VEC['temporal_unit'])
            loss.backward(); opt.step(); model.net.update_parameters(model._net)
            cum += loss.item(); n += 1
        losses.append(cum / n)
        if losses[-1] < best:
            best, best_epoch, wait = losses[-1], epoch, 0
        else:
            wait += 1
        if epoch % 10 == 0:
            log(f"    epoch {epoch:3d} loss {losses[-1]:.4f} (best {best:.4f} @ {best_epoch}, wait {wait}) {time.time() - t0:.0f}s")
        if wait >= patience:
            stop = f'patience={patience} (no new best for {patience} epochs)'
            break
    info = dict(epochs_run=len(losses), best_epoch=best_epoch, best_loss=best, final_loss=losses[-1],
                stop_reason=stop, seconds=round(time.time() - t0, 1), iterations_per_epoch=len(loader),
                n_train_windows=len(ds), loss_by_epoch=[round(v, 5) for v in losses])
    return model, info


def get_encoder(market, train_w, device, max_epochs, patience, seed, log=print):
    d = ROOT / 'thesis_results' / 'production' / market / 'ts2vec'
    d.mkdir(parents=True, exist_ok=True)
    pt, js = d / 'encoder.pt', d / 'training.json'
    want = dict(ts2vec=TS2VEC, max_epochs=max_epochs, patience=patience, seed=seed, window=WINDOW,
                n_train_windows=int(len(train_w)))
    if pt.exists() and js.exists():
        rec = json.loads(js.read_text())
        if all(rec['config'].get(k) == v for k, v in want.items()):
            model = TS2Vec(input_dims=1, device=device, **TS2VEC); model.load(str(pt))
            log(f"  encoder ({market}): cached, {rec['training']['epochs_run']} epochs, best loss {rec['training']['best_loss']:.4f}")
            return model, rec
    log(f"  encoder ({market}): training on {len(train_w)} windows ...")
    model, info = train_encoder(train_w, device, max_epochs, patience, seed, log)
    model.save(str(pt))
    rec = dict(config=want, training=info, libraries=library_versions(), device=device,
               note=('GPU training is not bit-reproducible (cuDNN). Loss trajectory and the encoder '
                     'file are recorded; embeddings are computed from the saved encoder.'))
    js.write_text(json.dumps(rec, indent=1))
    log(f"    done: {info['epochs_run']} epochs, best {info['best_loss']:.4f} @ {info['best_epoch']}, "
        f"{info['seconds']}s, stop: {info['stop_reason']}")
    return model, rec


def encode(model, w):
    return model.encode(w, encoding_window='full_series', batch_size=256).astype(np.float64)


# ----------------------------------------------------------------------------- statistic
def mmd2_u(Kxx, Kyy, Kxy):
    """Unbiased MMD^2 (U-statistic). May be negative."""
    n, m = len(Kxx), len(Kyy)
    return ((Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
            + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) - 2.0 * Kxy.sum() / (n * m))


def pooled_median_distance(D2xx, D2yy, D2xy):
    """Median pairwise Euclidean distance over the pooled sample (i < j)."""
    iu = np.triu_indices(len(D2xx), 1); ju = np.triu_indices(len(D2yy), 1)
    d2 = np.concatenate([D2xx[iu], D2yy[ju], D2xy.ravel()])
    return float(np.sqrt(np.median(d2)))


def statistics(D2xx, D2yy, D2xy, Gxx, Gyy, Gxy, d):
    """All kernel statistics for one pair, from squared distances and inner products."""
    sigma = pooled_median_distance(D2xx, D2yy, D2xy)
    out = {'sigma': sigma}
    for c in BANDWIDTH_MULTS:
        s2 = 2.0 * (sigma * c) ** 2
        out[f'rbf_x{c:g}'] = float(mmd2_u(np.exp(-D2xx / s2), np.exp(-D2yy / s2), np.exp(-D2xy / s2)))
    out['poly3'] = float(mmd2_u((Gxx / d + 1) ** 3, (Gyy / d + 1) ** 3, (Gxy / d + 1) ** 3))
    return out


STAT_KEYS = [f'rbf_x{c:g}' for c in BANDWIDTH_MULTS] + ['poly3']


def sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def sliding_null(E_full, L, n_win):
    """MMD between windows of [s, s+L) and [s+L, s+2L) for EVERY start s (stride 1)."""
    N_win = len(E_full)
    n_pos = N_win + WINDOW - 1 - 2 * L + 1                  # N - 2L + 1 positions
    D2 = sqdist(E_full, E_full); G = E_full @ E_full.T; d = E_full.shape[1]
    res = {k: np.empty(n_pos) for k in ['sigma'] + STAT_KEYS}
    for s in range(n_pos):
        a, b = s, s + L
        sl1, sl2 = slice(a, a + n_win), slice(b, b + n_win)
        st = statistics(D2[sl1, sl1], D2[sl2, sl2], D2[sl1, sl2], G[sl1, sl1], G[sl2, sl2], G[sl1, sl2], d)
        for k in res:
            res[k][s] = st[k]
    return res


def summarise(a):
    a = np.asarray(a, float)
    return {'n': int(len(a)), 'mean': float(a.mean()), 'sd': float(a.std()),
            'quantiles': {q: float(np.percentile(a, float(q))) for q in ('2.5', '25', '50', '75', '97.5', '99')},
            'min': float(a.min()), 'max': float(a.max())}


def mc_p(observed, null):
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def cov_spectrum(E):
    C = np.cov(E, rowvar=False); ev = np.clip(np.linalg.eigvalsh(C)[::-1], 0, None)
    tol = ev[0] * max(C.shape) * np.finfo(C.dtype).eps
    Z = E - E.mean(0); pc = Z @ np.linalg.svd(Z, full_matrices=False)[2][0]
    ac = np.array([np.corrcoef(pc[:-k], pc[k:])[0, 1] for k in range(1, 60)])
    k1 = int(np.argmax(ac < 1 / np.e)) + 1 if (ac < 1 / np.e).any() else 59
    return dict(n=int(len(E)), d=int(E.shape[1]), numerical_rank=int((ev > tol).sum()),
                eigenvalues_above_1em3_of_max=int((ev > 1e-3 * ev[0]).sum()),
                participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()),
                condition_number=float(ev[0] / ev[ev > tol][-1]),
                n_eff_from_autocorrelation=float(len(E) / (1 + 2 * np.clip(ac[:k1], 0, None).sum())))


def ordering_table(mean_by_model):
    """{stat: [models sorted by ascending mean statistic]}, lower = closer to real."""
    return {k: sorted(v, key=v.get) for k, v in mean_by_model.items()}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', nargs='*'); ap.add_argument('--seeds', nargs='*', type=int)
    ap.add_argument('--draws', type=int, default=20); ap.add_argument('--max-epochs', type=int, default=TRAIN['max_epochs'])
    ap.add_argument('--patience', type=int, default=TRAIN['patience'])
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"embedding_distance | pandas {pd.__version__} sklearn {library_versions()['sklearn']} device {device}")
    ns = load_notebook()
    markets = args.markets or ns['_discover_markets']()
    seeds = args.seeds or list(ns['SEEDS'])
    root = ns['RESULTS']
    # Phase B pre-flight, with the drift accepted and stamped: the weights were written at an older commit
    with contextlib.redirect_stdout(io.StringIO()) as pf_out:
        pre = ns['phase_b_preflight'](root, markets, seeds, accept_commit_drift=True)
    drifted = pre['commit_drift_accepted']
    tag = '_COMMIT_DRIFT' if drifted else ''
    banner = ("COMMIT DRIFT ACCEPTED: weights written at " f"{pre['artifact_commits']}, statistics computed by code at "
              f"{pre['git_head']!r}. New-code metric on old-code weights; do not merge with a clean result.")
    print('pre-flight:', 'commit drift accepted; ' if drifted else '', f"{len(pre['warns'])} warning(s), 0 failures")
    factories = ns['_build_model_factories']()[1]
    all_units = []

    for market in markets:
        print(f"\n=== {market} ===")
        parts, mu, sd = load_market(market, ns)
        train, valid, test = parts['train'], parts['valid'], parts['test']
        full = np.concatenate([train, valid, test]); L = len(test)
        z = lambda x: (np.asarray(x, dtype=np.float64) - mu) / sd
        encoder, enc_rec = get_encoder(market, windows(z(train)), device, args.max_epochs, args.patience, TRAIN['seed'])
        E_full = encode(encoder, windows(z(full)))
        n_win = L - WINDOW + 1
        E_real = E_full[len(train) + len(valid):]                       # windows lying wholly inside the test block
        assert len(E_real) == n_win
        # ---- null: fixed-length adjacent sliding, exhaustive
        t0 = time.time(); null = sliding_null(E_full, L, n_win)
        B = len(null['sigma']); blocks = len(full) / (2 * L)
        print(f"  null: L={L}, {B} positions (exhaustive), ~{blocks:.1f} independent blocks, {time.time() - t0:.0f}s; "
              f"sigma median {np.median(null['sigma']):.3f}")
        null_summary = {'construction': 'fixed-length adjacent sliding: windows of [s, s+L) vs [s+L, s+2L), every s (stride 1), full real series',
                        'length': int(L), 'n_windows_per_side': int(n_win), 'series_length': int(len(full)), 'exhaustive': True,
                        'n_positions_B': int(B), 'p_value': '(1 + #{null >= observed}) / (B + 1), upper tail',
                        'p_value_floor': 1.0 / (B + 1), 'approx_independent_blocks': float(blocks),
                        'dependence': ('Full enumeration is not independence: the positions overlap almost entirely, and '
                                       f'about {blocks:.0f} independent blocks exist at this length. The calibration fixes resolution, not independence.'),
                        'stats': {k: summarise(v) for k, v in null.items()}}
        d_null = ROOT / 'thesis_results' / 'production' / market / 'ts2vec'
        (d_null / f'null_adjacent_L{L}.json').write_text(json.dumps({**null_summary, 'draws': {k: [float(x) for x in v] for k, v in null.items()}}))
        spec_real = cov_spectrum(E_real)
        c_fid = {'estimable': False, 'computed': False,
                 'reason': ('369 stride-1 windows share 127/128 observations: about '
                            f"{spec_real['n_eff_from_autocorrelation']:.1f} independent observations of a {spec_real['d']}-dimensional "
                            f"distribution (participation ratio {spec_real['participation_ratio']:.1f}); the Fréchet term "
                            'tr sqrtm(S1 S2) is not estimable, and no number is published.'),
                 'real_test_window_embedding_covariance': spec_real}
        real_D2 = sqdist(E_real, E_real); real_G = E_real @ E_real.T; d_emb = E_real.shape[1]
        real_test_z = z(test)

        for seed in seeds:
            print(f"  seed {seed}")
            wdir = root / market / f'seed{seed}' / 'weights'
            with contextlib.redirect_stdout(io.StringIO()):
                models = {m: ns['reconstruct_model'](m, factories[m][0], wdir, train.astype(np.float32).reshape(-1, 1)) for m in MODELS}
            per = {m: [] for m in MODELS + [CONTROL]}
            for k in range(args.draws):
                ds = ns['_draw_seed'](seed, k)
                series = {}
                for m in MODELS:
                    with contextlib.redirect_stdout(io.StringIO()):
                        series[m] = ns['_regenerate'](models[m], L, ds).astype(np.float64)
                # the control: a permutation of the real test block. Draw 0 is the pipeline's own control (default_rng(42)).
                series[CONTROL] = np.random.default_rng(42 if k == 0 else ds).permutation(test)
                for m, x in series.items():
                    Es = encode(encoder, windows(z(x)))
                    st = statistics(real_D2, sqdist(Es, Es), sqdist(E_real, Es), real_G, Es @ Es.T, E_real @ Es.T, d_emb)
                    st.update({f'p_{key}': mc_p(st[key], null[key]) for key in STAT_KEYS})
                    st['draw'] = k; st['draw_seed'] = int(ds if not (m == CONTROL and k == 0) else 42)
                    per[m].append(st)
            res = {}
            for m, rows in per.items():
                r = {'draws': rows}
                for key in ['sigma'] + STAT_KEYS + [f'p_{x}' for x in STAT_KEYS]:
                    v = np.array([row[key] for row in rows])
                    r[key] = {'mean': float(v.mean()), 'sd': float(v.std())}
                for key in STAT_KEYS:
                    r[key]['fraction_p_le_0.05'] = float(np.mean([row[f'p_{key}'] <= 0.05 for row in rows]))
                    r[key]['exceeds_null_p97.5_fraction'] = float(np.mean([row[key] > null_summary['stats'][key]['quantiles']['97.5'] for row in rows]))
                res[m] = r
            means = {key: {m: res[m][key]['mean'] for m in MODELS} for key in STAT_KEYS}
            means_c = {key: {m: res[m][key]['mean'] for m in MODELS + [CONTROL]} for key in STAT_KEYS}
            order = ordering_table(means); order_c = ordering_table(means_c)
            from scipy.stats import kendalltau
            ref = order['rbf_x1']
            tau = {key: float(kendalltau([ref.index(m) for m in MODELS], [order[key].index(m) for m in MODELS])[0]) for key in STAT_KEYS}
            out = {
                'BANNER': banner if drifted else None, 'market': market, 'seed': seed, 'timestamp': datetime.now().isoformat(timespec='seconds'),
                'statistic': 'unbiased MMD^2 (U-statistic) between real test-block window embeddings and synthetic window embeddings',
                'kernels': {'primary': 'Gaussian RBF exp(-||a-b||^2 / (2 sigma^2))', 'secondary_reporting_only': 'polynomial degree 3, (a.b/d + 1)^3',
                            'bandwidth': 'median heuristic (median pairwise Euclidean distance, pooled real + synthetic embeddings)',
                            'bandwidth_sweep_multipliers': list(BANDWIDTH_MULTS)},
                'windows': {'length': WINDOW, 'stride': 1, 'n_real': int(n_win), 'n_synthetic_per_draw': int(n_win), 'standardisation': 'z-score with the train-block mean/std backfilled in _meta.json; no tanh'},
                'ts2vec': {'config': enc_rec['config'], 'training': {k: v for k, v in enc_rec['training'].items() if k != 'loss_by_epoch'},
                           'encoder_file': f'thesis_results/production/{market}/ts2vec/encoder.pt',
                           'tsgbench_reference': 'output_dims=100, batch_size=8, lr=0.001, max_train_length=3000; fit() with no n_epochs (TS2Vec default 200 iterations) and no early stopping. The 300-epoch cap and 10-epoch patience are this job\'s, not TSGBench\'s code.'},
                'null': null_summary, 'c_fid': c_fid,
                'sigma_median_heuristic_real_vs_generators': {m: res[m]['sigma'] for m in MODELS + [CONTROL]},
                'n_draws': args.draws, 'draw_seeds': 'draw k uses _draw_seed(seed, k) = seed + 100000 k; draw 0 replays the recorded series for GARCH/GJR-GARCH; the control draw 0 is default_rng(42)',
                'results': res, 'ordering_by_mean_statistic': order, 'ordering_including_control': order_c,
                'kendall_tau_vs_rbf_x1_ordering_models_only': tau,
                'ordering_identical_across_bandwidth_sweep': bool(all(order[f'rbf_x{c:g}'] == order['rbf_x1'] for c in BANDWIDTH_MULTS)),
                'libraries': library_versions(), 'phase_b_preflight': {'warnings': pre['warns'], 'commit_drift_accepted': drifted, 'git_head': pre['git_head'], 'artifact_commits': pre['artifact_commits']},
            }
            path = root / market / f'seed{seed}' / f'embedding_distance{tag}.json'
            path.write_text(json.dumps(out, indent=1, default=str))
            ctrl = res[CONTROL]['rbf_x1']; best = min(means['rbf_x1'].values())
            print(f"    control rbf mean {ctrl['mean']:.4f} (p<=.05 in {ctrl['fraction_p_le_0.05']:.0%} of draws) | best model {best:.4f} | null p97.5 {null_summary['stats']['rbf_x1']['quantiles']['97.5']:.4f} | order {order['rbf_x1']}")
            all_units.append(path)
    print(f"\nDONE in {(time.time() - t_start) / 60:.1f} min; wrote {len(all_units)} file(s)")


if __name__ == '__main__':
    main()
