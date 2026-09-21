"""Part 1 (compute): full-series fixed-length adjacent sliding nulls (approved construction) at both evaluation lengths, under three CV
schemes on the SAME pairs: contiguous (published), purged (new option, embargo = 20), shuffled (diagnostic only). Plus the stationary GARCH-t
reference null (Task 10 recipe: parameters from seed-42 GARCH_params.json, seeds 100000+2k / +2k+1, 300 pairs) under the same schemes.
Limitation (recorded in the report): the sliding-null population spans 20 years; the observed comparison is the test period."""
import sys, time, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
SCHEMES = ('contiguous', 'purged', 'shuffled'); (OUT / 'nulls').mkdir(parents=True, exist_ok=True); t0 = time.time()
meta = {}
for m in MARKETS:
    full, test = load_full(m), load_test(m); N = len(full)
    for tag, L in (('track_a', len(test)), ('walk_forward', N // 6)):
        starts = list(range(0, N - 2 * L + 1)); pairs = [(full[s:s + L], full[s + L:s + 2 * L]) for s in starts]
        for sc in SCHEMES:
            a = np.array(pmap([(x, y, sc) for x, y in pairs])); np.save(OUT / 'nulls' / f'{m}_{tag}_{sc}.npy', a)
        meta[f'{m}/{tag}'] = dict(L=L, N=N, positions=len(starts), independent_blocks=N / (2 * L), starts=[0, starts[-1]])
        print(f'{m} {tag} L={L} positions={len(starts)} done ({time.time()-t0:.0f}s)', flush=True)
json.dump(meta, open(OUT / 'nulls' / 'sliding_meta.json', 'w'), indent=1)
# ---- stationary GARCH-t reference null
ns = load_fm(); GARCHModel = ns['GARCHModel']; GenErr = ns['GenerationSanityError']; B = 300; out = {}
for m in MARKETS:
    rec = json.load(open(RES / m / 'seed42' / 'weights' / 'GARCH_params.json'))
    train = pd.read_parquet(ROOT / f'data/processed_files/train/{m}_train.parquet').to_numpy().astype(np.float32)
    model = GARCHModel.from_record(rec, train); full, test = load_full(m), load_test(m); N = len(full); src = rec['generated_from']
    out[m] = dict(generated_from=src, persistence=rec['constrained_persistence'] if src == 'constrained' else rec['unconstrained_persistence'], nulls={})
    for tag, L in (('track_a', len(test)), ('walk_forward', N // 6)):
        pairs, rejects, k = [], 0, 0
        while len(pairs) < B and k < 4 * B:
            sa, sb = 100000 + 2 * k, 100000 + 2 * k + 1; k += 1
            try: model.seed = sa; a = model.generate(L).flatten(); model.seed = sb; b = model.generate(L).flatten()
            except GenErr: rejects += 1; continue
            pairs.append((a, b))
        out[m]['nulls'][tag] = dict(length=int(L), n_pairs=len(pairs), rejected_by_generation_guard=rejects)
        for sc in SCHEMES:
            aucs = np.array(pmap([(a, b, sc) for a, b in pairs])); np.save(OUT / 'nulls' / f'stationary_{m}_{tag}_{sc}.npy', aucs)
        print(f'stationary {m} {tag} pairs={len(pairs)} rejected={rejects} ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open(OUT / 'nulls' / 'stationary_meta.json', 'w'), indent=1); print('PART1_NULLS_DONE', round(time.time() - t0), 's')
