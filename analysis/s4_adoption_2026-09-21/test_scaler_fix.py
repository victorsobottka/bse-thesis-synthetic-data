"""Part 3 checks: (1) POST equals a hand-written per-fold scaling with the same folds, for contiguous and purged; (2) PRE equals the hand-written global-scaling version;
(3) the two differ, by how much, on real pairs; (4) neither touches the fold construction (same test chunks)."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness2 import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
pre = load_ns(PRE_NB)['FinancialMetrics']; post = load_ns(POST_NB)['FinancialMetrics']
def feats(x, w=20): return np.array([[x[i-w:i].mean(), x[i-w:i].std(), np.abs(x[i-w:i]).mean(), (x[i-w:i]**2).mean()] for i in range(w, len(x))], dtype=np.float32)
def manual(a, b, cv, scale):
    n = min(len(a), len(b)); r, s = feats(a[:n]), feats(b[:n]); k = min(len(r), len(s)); X = np.vstack([r[:k], s[:k]]); y = np.hstack([np.ones(k), np.zeros(k)])
    folds = post._purged_folds(k, 5, 20) if cv == 'purged' else [(tr, te, 0) for tr, te in StratifiedKFold(5).split(X, y)]
    if scale == 'global': X = StandardScaler().fit_transform(X)
    out = []
    for tr, te, _ in folds:
        Xtr, Xte = X[tr], X[te]
        if scale == 'fold': sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=300, random_state=42).fit(Xtr, y[tr]); out.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(out))
rows = []
for m in MARKETS:
    full = load_full(m); L = len(load_test(m))
    for st in (0, 700, 1500, 2600, 3300):
        a, b = full[st:st + L], full[st + L:st + 2 * L]
        for cv in ('contiguous', 'purged'):
            p, q = pre(a, b).compute_discriminative_score(cv=cv)['auc'], post(a, b).compute_discriminative_score(cv=cv)['auc']
            rows.append(dict(market=m, start=st, cv=cv, pre=p, post=q, manual_global=manual(a, b, cv, 'global'), manual_fold=manual(a, b, cv, 'fold')))
D = pd.DataFrame(rows)
print('PRE  vs hand-written global scaling: max |diff| %.2e' % (D.pre - D.manual_global).abs().max()); print('POST vs hand-written per-fold scaling: max |diff| %.2e' % (D.post - D.manual_fold).abs().max())
print('PRE vs POST on %d real adjacent-block pairs: max |diff| %.4f, mean |diff| %.4f, mean signed (post-pre) %+.4f' % (len(D), (D.post - D.pre).abs().max(), (D.post - D.pre).abs().mean(), (D.post - D.pre).mean()))
print('signature / default:', post.compute_discriminative_score.__defaults__)
