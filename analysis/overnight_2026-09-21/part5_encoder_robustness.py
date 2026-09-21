"""Part 5: encoder robustness of the TS2Vec + MMD study. Two further encoder configurations, everything else as in Task 5 v3.

  d320      output_dims = 320, otherwise the main run (hidden 64, depth 10, lr 1e-3, batch 8, up to 300 epochs, patience 10, seed 42).
  tsgbench  TSGBench's literal configuration: output_dims = 100, batch_size = 8, lr 1e-3, max_train_length 3000; upstream TS2Vec.fit() with NO
            epoch count -> TS2Vec's default n_iters (200 if train_data.size <= 100000 else 600; ours is ~490k elements -> 600 iterations),
            no early stopping. Seed 42.

Stages:   train <config> <market>   trains and stores thesis_results/production/<market>/ts2vec_<config>/{encoder.pt,training.json}
          eval  <config>            per market: exhaustive adjacent-sliding null with THAT encoder; the 15 units x 6 series x 20 cached draws
                                    (draws/ from draws_cache.py, identical to the main run's); writes part5_<config>.json
          report                    appends the section to the consolidated report
GPU training is not bit-reproducible (cuDNN); loss trajectories and encoder files are recorded. Phase A is not touched; no generator is retrained.
"""
import sys, json, time, math, random, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
import embedding_distance as ed
from ts2vec import TS2Vec
RES = ROOT / 'thesis_results/production'; OUT = RES / 'overnight_2026-09-21'; REPORT = RES / 'overnight_report_2026-09-21.md'
MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
ORDER = ['TimeGAN', 'GARCH', 'QuantGAN', 'GJR-GARCH', 'CNN-WGAN-GP', ed.CONTROL]
BASE = dict(ed.TS2VEC)
CONFIGS = {'d320': {**BASE, 'output_dims': 320}, 'tsgbench': dict(BASE)}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

def enc_dir(market, cfg): d = RES / market / f'ts2vec_{cfg}'; d.mkdir(parents=True, exist_ok=True); return d

def load_encoder(market, cfg):
    m = TS2Vec(input_dims=1, device=dev, **CONFIGS[cfg]); m.load(str(enc_dir(market, cfg) / 'encoder.pt')); return m

def train(cfg, market):
    ns = ed.load_notebook(); parts, mu, sd = ed.load_market(market, ns)
    W = ed.windows((parts['train'] - mu) / sd); t0 = time.time()
    if cfg == 'd320':
        ed.TS2VEC = CONFIGS['d320']                                # train_encoder reads the module global
        model, info = ed.train_encoder(W, dev, ed.TRAIN['max_epochs'], ed.TRAIN['patience'], ed.TRAIN['seed'], log=lambda s: print(f'[{market}] {s}', flush=True))
        want = dict(ts2vec=CONFIGS['d320'], max_epochs=ed.TRAIN['max_epochs'], patience=ed.TRAIN['patience'], seed=ed.TRAIN['seed'], window=ed.WINDOW, n_train_windows=int(len(W)))
    else:
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42) if torch.cuda.is_available() else None
        model = TS2Vec(input_dims=1, device=dev, **CONFIGS['tsgbench'])
        n_iters_default = 200 if W.size <= 100000 else 600
        loss_log = model.fit(W)                                     # literal: no n_epochs, no n_iters, no early stopping
        info = dict(training_call='TS2Vec.fit(train_windows)  # no n_epochs / n_iters', n_iters_default_applied=n_iters_default, iterations_run=int(model.n_iters),
                    epochs_touched=int(model.n_epochs), loss_by_epoch=[round(float(v), 5) for v in loss_log], final_epoch_mean_loss=float(loss_log[-1]),
                    train_data_size=int(W.size), stop_reason='n_iters reached (default); no early stopping', seconds=round(time.time() - t0, 1))
        want = dict(ts2vec=CONFIGS['tsgbench'], seed=42, window=ed.WINDOW, n_train_windows=int(len(W)))
    model.save(str(enc_dir(market, cfg) / 'encoder.pt'))
    (enc_dir(market, cfg) / 'training.json').write_text(json.dumps(dict(config=want, training=info, libraries=ed.library_versions(), device=dev,
        note='GPU training is not bit-reproducible (cuDNN). Loss trajectory and the encoder file are recorded; embeddings are computed from the saved encoder.'), indent=1))
    print(f'TRAIN_DONE {cfg} {market} {time.time()-t0:.0f}s', {k: v for k, v in info.items() if k != 'loss_by_epoch'}, flush=True)

def evaluate(cfg):
    ns = ed.load_notebook(); t0 = time.time(); out = dict(config=cfg, ts2vec=CONFIGS[cfg], markets={})
    for market in MARKETS:
        parts, mu, sd = ed.load_market(market, ns); train_, valid, test = parts['train'], parts['valid'], parts['test']
        full = np.concatenate([train_, valid, test]); L, N = len(test), len(full); n_win = L - ed.WINDOW + 1; z = lambda x: (np.asarray(x, dtype=np.float64) - mu) / sd
        enc = load_encoder(market, cfg); tr = json.load(open(enc_dir(market, cfg) / 'training.json'))
        E_full = ed.encode(enc, ed.windows(z(full))); E_real = E_full[len(train_) + len(valid):]; d = E_real.shape[1]; assert len(E_real) == n_win
        null = ed.sliding_null(E_full, L, n_win)                    # per-pair median-heuristic RBF x1, poly3, exhaustive
        real_D2, real_G = ed.sqdist(E_real, E_real), E_real @ E_real.T
        obs = {s: {m: [] for m in ORDER} for s in (42, 43, 44)}
        for seed in (42, 43, 44):
            arr = np.load(OUT / 'draws' / f'{market}_seed{seed}.npz')
            for mod in ORDER:
                for k in range(20):
                    Es = ed.encode(enc, ed.windows(z(arr[f'{mod}__{k}']))); D2yy, D2xy = ed.sqdist(Es, Es), ed.sqdist(E_real, Es)
                    st = ed.statistics(real_D2, D2yy, D2xy, real_G, Es @ Es.T, E_real @ Es.T, d); st['D2yy'] = D2yy; st['D2xy'] = D2xy   # kept for the fixed-sigma pass
                    obs[seed][mod].append(st)
        sigma_star = float(np.median([st['sigma'] for s in obs.values() for v in s.values() for st in v]))
        K = np.exp(-ed.sqdist(E_full, E_full) / (2 * sigma_star ** 2)); n_pos = N - 2 * L + 1
        null_fixed = np.array([ed.mmd2_u(K[s:s + n_win, s:s + n_win], K[s + L:s + L + n_win, s + L:s + L + n_win], K[s:s + n_win, s + L:s + L + n_win]) for s in range(n_pos)]); del K
        units = {}
        for seed in (42, 43, 44):
            units[seed] = {}
            for mod in ORDER:
                rows = obs[seed][mod]; s2 = 2 * sigma_star ** 2
                fixed = [float(ed.mmd2_u(np.exp(-real_D2 / s2), np.exp(-r['D2yy'] / s2), np.exp(-r['D2xy'] / s2))) for r in rows]
                units[seed][mod] = dict(rbf=[r['rbf_x1'] for r in rows], poly3=[r['poly3'] for r in rows], sigma=[r['sigma'] for r in rows], fixed=fixed,
                                        p_rbf=[ed.mc_p(r['rbf_x1'], null['rbf_x1']) for r in rows], p_poly3=[ed.mc_p(r['poly3'], null['poly3']) for r in rows], p_fixed=[ed.mc_p(v, null_fixed) for v in fixed])
        q = lambda a: {k: float(np.percentile(a, float(k))) for k in ('50', '97.5')}
        out['markets'][market] = dict(L=int(L), N=int(N), embedding_dim=int(d), training={k: v for k, v in tr['training'].items() if k != 'loss_by_epoch'}, sigma_star=sigma_star,
                                      null=dict(positions=int(len(null['sigma'])), sigma_median=float(np.median(null['sigma'])), rbf=dict(mean=float(null['rbf_x1'].mean()), sd=float(null['rbf_x1'].std()), **q(null['rbf_x1'])),
                                                poly3=dict(mean=float(null['poly3'].mean()), sd=float(null['poly3'].std()), **q(null['poly3']))),
                                      null_fixed=dict(mean=float(null_fixed.mean()), sd=float(null_fixed.std()), **q(null_fixed)), units={str(s): v for s, v in units.items()})
        print(f'EVAL {cfg} {market}: d={d} null rbf mean {null["rbf_x1"].mean():.4f} p97.5 {np.percentile(null["rbf_x1"], 97.5):.4f} sigma null {np.median(null["sigma"]):.2f} obs {np.median([st["sigma"] for s in obs.values() for v in s.values() for st in v]):.2f} ({time.time()-t0:.0f}s)', flush=True)
    out['libraries'] = ed.library_versions(); (OUT / f'part5_{cfg}.json').write_text(json.dumps(out, indent=1)); print('EVAL_DONE', cfg, round(time.time() - t0), 's')

if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'train': train(a[1], a[2])
    elif a[0] == 'eval': evaluate(a[1])
