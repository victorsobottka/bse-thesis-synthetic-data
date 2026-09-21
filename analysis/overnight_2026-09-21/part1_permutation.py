"""Part 1: is TS2Vec permutation-sensitive? Encode-only; the trained encoders are loaded, never retrained.

Probe protocol. The encoder was trained on the TRAIN block, so real train windows are in-sample for it and
trivially separable from shuffled ones (in-period AUC = 1.000, reported as a diagnostic, not a result). The
clean probe uses only blocks the encoder never saw: fit on real-vs-shuffled VALIDATION-block windows, score on
real-vs-shuffled TEST-block windows.
"""
import sys, json, time, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
import embedding_distance as ed
from ts2vec import TS2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
OUT = ROOT / 'thesis_results/production/overnight_2026-09-21'; REPORT = ROOT / 'thesis_results/production/overnight_report_2026-09-21.md'
N_PERM = 20; MARKETS = ['BOVESPA', 'FTSE', 'MOEX', 'NIFTY50', 'SHANGHAI']
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ns = ed.load_notebook()

def load_encoder(market):
    m = TS2Vec(input_dims=1, device=dev, **ed.TS2VEC); m.load(str(ROOT / f'thesis_results/production/{market}/ts2vec/encoder.pt')); return m

def acf_feats(W):
    x = W[:, :, 0].astype(np.float64)
    def ac(a, k):
        a = a - a.mean(1, keepdims=True); return (a[:, k:] * a[:, :-k]).sum(1) / (a * a).sum(1)
    return np.c_[ac(np.abs(x), 1), ac(x ** 2, 1), ac(np.abs(x), 5)]

def probe_auc(Xtr_real, Xtr_shuf, Xte_real, Xte_shuf):
    Xtr = np.vstack([Xtr_real, Xtr_shuf]); ytr = np.r_[np.zeros(len(Xtr_real)), np.ones(len(Xtr_shuf))]
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    Xte = np.vstack([Xte_real, Xte_shuf]); yte = np.r_[np.zeros(len(Xte_real)), np.ones(len(Xte_shuf))]
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))

def in_period_auc(X_real, X_shuf, frac=0.8, embargo=128):
    n = len(X_real); cut = int(n * frac)
    return probe_auc(X_real[:cut - embargo], X_shuf[:cut - embargo], X_real[cut:], X_shuf[cut:])

rows, detail, t0 = [], {}, time.time()
for market in MARKETS:
    parts, mu, sd = ed.load_market(market, ns); train, valid, test = parts['train'], parts['valid'], parts['test']; L = len(test)
    z = lambda x: (np.asarray(x, dtype=np.float64) - mu) / sd
    enc = load_encoder(market)
    Wte, Wva, Wtr = ed.windows(z(test)), ed.windows(z(valid)), ed.windows(z(train))
    E_te, E_va, E_tr = ed.encode(enc, Wte), ed.encode(enc, Wva), ed.encode(enc, Wtr)
    null = json.load(open(ROOT / f'thesis_results/production/{market}/ts2vec/null_adjacent_L{L}.json'))
    nr, npoly = np.array(null['draws']['rbf_x1']), np.array(null['draws']['poly3'])
    D2r, Gr, d = ed.sqdist(E_te, E_te), E_te @ E_te.T, E_te.shape[1]
    ctrl = np.mean([json.load(open(ROOT / f'thesis_results/production/{market}/seed{s}/embedding_distance_COMMIT_DRIFT.json'))['results'][ed.CONTROL]['rbf_x1']['mean'] for s in (42, 43, 44)])
    A_te, A_va, A_tr = acf_feats(Wte), acf_feats(Wva), acf_feats(Wtr)
    R = []
    for k in range(N_PERM):
        x_te = np.random.default_rng(9000 + k).permutation(test); x_va = np.random.default_rng(9700 + k).permutation(valid); x_tr = np.random.default_rng(9500 + k).permutation(train)
        Ws_te, Ws_va, Ws_tr = ed.windows(z(x_te)), ed.windows(z(x_va)), ed.windows(z(x_tr))
        Es_te, Es_va, Es_tr = ed.encode(enc, Ws_te), ed.encode(enc, Ws_va), ed.encode(enc, Ws_tr)
        st = ed.statistics(D2r, ed.sqdist(Es_te, Es_te), ed.sqdist(E_te, Es_te), Gr, Es_te @ Es_te.T, E_te @ Es_te.T, d)
        R.append(dict(perm=k, rbf=st['rbf_x1'], p_rbf=ed.mc_p(st['rbf_x1'], nr), poly3=st['poly3'], p_poly3=ed.mc_p(st['poly3'], npoly), sigma=st['sigma'],
                      auc_valid_to_test=probe_auc(E_va, Es_va, E_te, Es_te), auc_acf_valid_to_test=probe_auc(A_va, acf_feats(Ws_va), A_te, acf_feats(Ws_te)),
                      auc_train_to_test=probe_auc(E_tr, Es_tr, E_te, Es_te), auc_in_period_train=in_period_auc(E_tr, Es_tr)))
    D = pd.DataFrame(R); detail[market] = D.round(6).to_dict('records')
    rows.append(dict(market=market, L=L, n_windows=len(E_te), rbf_mean=D.rbf.mean(), rbf_sd=D.rbf.std(ddof=0), p_rbf_mean=D.p_rbf.mean(), share_p_le_05=float((D.p_rbf <= .05).mean()),
                     poly3_mean=D.poly3.mean(), poly3_sd=D.poly3.std(ddof=0), p_poly3_mean=D.p_poly3.mean(), share_p_poly3_le_05=float((D.p_poly3 <= .05).mean()),
                     null_median=float(np.median(nr)), null_p97_5=float(np.percentile(nr, 97.5)), task5_control_rbf_mean=float(ctrl), sigma_mean=D.sigma.mean(),
                     auc_valid_to_test_mean=D.auc_valid_to_test.mean(), auc_valid_to_test_sd=D.auc_valid_to_test.std(ddof=0), auc_acf_valid_to_test_mean=D.auc_acf_valid_to_test.mean(),
                     auc_train_to_test_mean=D.auc_train_to_test.mean(), auc_in_period_train_mean=D.auc_in_period_train.mean()))
    print(f"{market}: rbf {D.rbf.mean():.4f} (p {D.p_rbf.mean():.3f}) poly3 {D.poly3.mean():.4f} | probe AUC valid->test {D.auc_valid_to_test.mean():.3f} (ACF ref {D.auc_acf_valid_to_test.mean():.3f}); train->test {D.auc_train_to_test.mean():.3f}; in-period(train, in-sample) {D.auc_in_period_train.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)
T = pd.DataFrame(rows)
json.dump({'protocol': dict(n_permutations=N_PERM, permutation_seeds='test block 9000+k, valid block 9700+k, train block 9500+k',
           primary_probe='logistic regression (C=1, standardised) fitted on real-vs-shuffled VALIDATION-block window embeddings, scored on real-vs-shuffled TEST-block windows; the encoder has seen neither block',
           diagnostics='train->test (real train windows are in-sample for the encoder); in-period train (80/20 with a 128-window embargo; in-sample, AUC 1.000 is memorisation, not ordering)',
           reference_probe='same protocol on 3 volatility-clustering features (lag-1 and lag-5 autocorrelation of |r|, lag-1 of r^2): separability achievable in principle',
           mmd='unbiased MMD^2, RBF at the median heuristic and degree-3, p-values against the existing exhaustive adjacent-sliding null'),
           'per_market': rows, 'per_permutation': detail}, open(OUT / 'part1_permutation_sensitivity.json', 'w'), indent=1)
T.to_csv(OUT / 'part1_permutation_sensitivity.csv', index=False)
a, r = T.auc_valid_to_test_mean, T.auc_acf_valid_to_test_mean
md = ['', '## Part 1: is TS2Vec permutation-sensitive?', '',
      'Encoders loaded from `thesis_results/production/<market>/ts2vec/encoder.pt` (no retraining). Real test block versus the same block with its returns permuted, 20 independent permutations, same windowing (128, stride 1), z-scoring and max-pooling as the main run. MMD² is the unbiased statistic; p-values are against the existing exhaustive adjacent-sliding null.', '',
      '| market | RBF MMD² (mean ± sd) | p vs null (mean) | draws p≤.05 | deg-3 MMD² (mean ± sd) | p (deg-3) | null median (RBF) | Task 5 v3 control (RBF) |', '|---|---|---|---|---|---|---|---|']
for x in rows: md.append(f"| {x['market']} | {x['rbf_mean']:.4f} ± {x['rbf_sd']:.4f} | {x['p_rbf_mean']:.3f} | {x['share_p_le_05']:.0%} | {x['poly3_mean']:.4f} ± {x['poly3_sd']:.4f} | {x['p_poly3_mean']:.3f} | {x['null_median']:.4f} | {x['task5_control_rbf_mean']:.4f} |")
md += ['', '**Linear probe** (logistic regression, real vs shuffled windows; AUC 0.5 = not separable). Primary protocol: fitted on the validation block, scored on the test block, so the encoder has seen neither. Reference: the same protocol on three volatility-clustering features (lag-1 and lag-5 autocorrelation of |r|, lag-1 of r²), which shows what separability is achievable in principle.', '',
       '| market | probe AUC on TS2Vec embeddings (valid→test) | ACF-feature reference (valid→test) | diagnostic: train→test | diagnostic: in-period, train block |', '|---|---|---|---|---|']
for x in rows: md.append(f"| {x['market']} | **{x['auc_valid_to_test_mean']:.3f}** ± {x['auc_valid_to_test_sd']:.3f} | {x['auc_acf_valid_to_test_mean']:.3f} | {x['auc_train_to_test_mean']:.3f} | {x['auc_in_period_train_mean']:.3f} |")
md += ['', 'The train-block diagnostics are not used for inference: the encoder was trained on the real train windows, so a probe on them cannot be read as out-of-sample sensitivity to ordering (in-period AUC is 1.000 in every market, a value I cannot attribute to ordering rather than to the encoder having seen those windows). The valid and test blocks were never used for encoder training or early stopping (patience is on the epoch-mean training loss).', '']
REPORT.open('a').write('\n'.join(md) + '\n')
print('\nPART1_DONE', round(time.time() - t0), 's | valid->test probe AUC mean', round(a.mean(), 3), '| ACF reference', round(r.mean(), 3))
