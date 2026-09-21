"""Part 2 (compute): the full metric set for the five generators (Phase B draws), the shuffled control and the baselines, per draw, against the test block,
through the notebook's own evaluate_multiple_models -> FinancialMetrics (same call as run_evaluate_phase: max_lag=min(50, n//4), the cached legacy AUC null).
The generators' and the control's rows are then compared with the Phase B CSV cell by cell: the same code path must reproduce them exactly."""
import sys, io, json, time, contextlib, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'task11_2026-09-21'))
from harness import *                       # load_fm, ROOT, RES, MARKETS, np, pd
OUT2 = RES / 'baseline_generators'; UNITS = [(m, s) for m in MARKETS for s in (42, 43, 44)]
GENS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; BASES = ['iid', 'sb_r', 'sb_abs']
_ns = None
def init():
    global _ns
    _ns = load_fm()
def unit(u):
    m, s = u; ns = _ns
    real = ns['load_dataset'](ns['TEST_DIR'] / f'{m}_test.parquet').flatten()
    with contextlib.redirect_stdout(io.StringIO()): auc_null = ns['precompute_auc_null'](real, m, ns['RESULTS'] / m)
    ser = np.load(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / 'synthetic_phase_b_COMMIT_DRIFT.npz'); bs = np.load(OUT2 / 'draws' / f'{m}_seed{s}.npz'); frames = []
    for k in range(20):
        synth = {g: ser[f'{g}__draw{k}'] for g in GENS}; synth.update({b: bs[f'{b}__{k}'] for b in BASES})
        with contextlib.redirect_stdout(io.StringIO()):
            df = ns['evaluate_multiple_models'](real, synth, max_lag=min(50, len(real) // 4), auc_null=auc_null)
        df.insert(0, 'draw', k); df.insert(0, 'seed', s); df.insert(0, 'market', m); frames.append(df)
    return pd.concat(frames, ignore_index=True)
if __name__ == '__main__':
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context('fork').Pool(5, initializer=init)  # run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=20 (Phase B's default BLAS threading: arch's GARCH fit is thread-sensitive at 1e-5) as p: res = p.map(unit, UNITS, chunksize=1)
    D = pd.concat(res, ignore_index=True); rank_cols = [c for c in D.columns if c.endswith('_rank')]; D = D.drop(columns=rank_cols)   # ranks are recomputed over the full pool downstream
    D.to_csv(RES / 'baseline_generators' / 'metrics_all_entries_by_draw.csv', index=False)
    D = pd.read_csv(RES / 'baseline_generators' / 'metrics_all_entries_by_draw.csv')      # compare as stored: Phase B's CSV was written from the same float32 metric values
    worst, cells = 0.0, 0
    for m, s in UNITS:
        pb = pd.read_csv(RES / m / f'seed{s}' / 'phase_b_COMMIT_DRIFT' / f'{m}_metrics_phase_b_COMMIT_DRIFT.csv'); num = [c for c in pb.columns if c in D.columns and c not in ('draw', 'model') and pd.api.types.is_numeric_dtype(pb[c]) and not c.endswith('_rank')]
        a = D[(D.market == m) & (D.seed == s) & D.model.isin(GENS + ['CONTROL: shuffled real'])].sort_values(['draw', 'model']).reset_index(drop=True); b = pb.sort_values(['draw', 'model']).reset_index(drop=True)
        assert (a[['draw', 'model']].values == b[['draw', 'model']].values).all()
        d = (a[num] - b[num]).abs(); worst = max(worst, float(np.nanmax(d.to_numpy()))); cells += d.size
        assert (a[num].isna() == b[num].isna()).all().all()
    print(f'{len(D)} rows ({len(UNITS)} units x 20 draws x 9 entries); generators + control vs Phase B CSV: {cells} cells, max |difference| = {worst:g} ({time.time()-t0:.0f}s)')
    json.dump({'cells_compared': cells, 'max_abs_difference': worst}, open(RES / 'baseline_generators' / 'validation_vs_phase_b.json', 'w'))
