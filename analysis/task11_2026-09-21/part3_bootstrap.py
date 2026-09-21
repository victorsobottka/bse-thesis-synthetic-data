"""Part 3 (+ scenario summaries). Reads unit_composites.csv / draw_composites_frame_F.csv.gz. Bootstrap: resample the 15 market-seed units with replacement
(B = 19,999), recompute each model's overall composite (mean over resampled units), take the margin second - first for the POINT-ESTIMATE first and second,
95% percentile interval. Sensitivity (extra, labelled): resample the 5 markets (all 3 seeds each), since seeds of one market share the real data."""
import sys, json
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from harness import *
MODELS = ['TimeGAN', 'QuantGAN', 'CNN-WGAN-GP', 'GARCH', 'GJR-GARCH']; CTRL = 'CONTROL: shuffled real'; SEEDS = (42, 43, 44)
U = pd.read_csv(OUT / 'unit_composites.csv'); F = pd.read_csv(OUT / 'draw_composites_frame_F.csv.gz')
B = 19999; rng = np.random.default_rng(20260921)
UNITS = [(m, s) for m in MARKETS for s in SEEDS]
def matrix(frame, scen):
    d = U[(U.frame == frame) & (U.scenario == scen) & (U.model != CTRL)]
    return np.array([[d[(d.market == m) & (d.seed == s) & (d.model == mod)].composite.iloc[0] for mod in MODELS] for m, s in UNITS])       # (15, 5)
def boot(C):
    ov = C.mean(0); order = np.argsort(ov); a, b = order[0], order[1]
    idx = rng.integers(0, len(C), (B, len(C))); means = C[idx].mean(1); mg = means[:, b] - means[:, a]
    Cm = C.reshape(5, 3, 5); cidx = rng.integers(0, 5, (B, 5)); cm = Cm[cidx].mean((1, 2)); mgc = cm[:, b] - cm[:, a]
    lo, hi = np.percentile(mg, [2.5, 97.5]); clo, chi = np.percentile(mgc, [2.5, 97.5])
    third = order[2]
    return dict(first=MODELS[a], second=MODELS[b], third=MODELS[third], composite_first=ov[a], composite_second=ov[b], margin=ov[b] - ov[a], ci_lo=lo, ci_hi=hi, excludes_zero=bool(lo > 0),
                p_order_reversed=float((mg <= 0).mean()), p_first_stays_first=float((np.argmin(means, 1) == a).mean()),
                cluster_ci_lo=clo, cluster_ci_hi=chi, cluster_excludes_zero=bool(clo > 0), cluster_p_order_reversed=float((mgc <= 0).mean()))
pubU = pd.read_csv(RES / 'overall_performance.csv')
pub_comp = {}
for m in MARKETS:
    for s in SEEDS:
        df = pd.read_csv(RES / m / f'seed{s}' / f'{m}_metrics.csv'); df = df[~df.model.str.startswith('CONTROL')].set_index('model'); pub_comp[(m, s)] = df.composite_rank.reindex(MODELS)
PUB = np.array([pub_comp[u].to_numpy() for u in UNITS]); PUBPOS = np.array([pd.Series(r).rank(method='min').to_numpy() for r in PUB])
rows, changes, seedrows, ctrlrows = [], [], [], []
frames = [('P', 'S0'), ('P', 'S1')] + [('F', s) for s in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5')]
for frame, scen in frames:
    C = matrix(frame, scen); r = boot(C); r.update(frame=frame, scenario=scen); ov = C.mean(0); pos = pd.Series(ov).rank(method='min').to_numpy()
    for mod, o, p in zip(MODELS, ov, pos): r[f'composite_{mod}'] = o; r[f'position_{mod}'] = int(p)
    rows.append(r)
    posU = np.array([pd.Series(x).rank(method='min').to_numpy() for x in C])
    changes.append(dict(frame=frame, scenario=scen, position_changed_vs_published=int((posU != PUBPOS).sum()), composite_value_changed_vs_published=int((np.abs(C - PUB) > 1e-9).sum()),
                        mean_abs_composite_change=float(np.abs(C - PUB).mean()), n_evaluations=int(C.size),
                        units_with_any_position_change=int((posU != PUBPOS).any(1).sum())))
    if frame == 'F':
        C0 = matrix('F', 'S0'); pos0 = np.array([pd.Series(x).rank(method='min').to_numpy() for x in C0]); changes[-1].update(position_changed_vs_F_S0=int((posU != pos0).sum()), composite_value_changed_vs_F_S0=int((np.abs(C - C0) > 1e-9).sum()))
    # seed spread: per market sd over the 3 seeds (ddof=1) of the unit composite; and the overall margin per seed
    Cm = C.reshape(5, 3, 5); sd = Cm.std(axis=1, ddof=1)                                    # (5 markets, 5 models)
    for k, mod in enumerate(MODELS): seedrows.append(dict(frame=frame, scenario=scen, model=mod, seed_sd_mean_over_markets=float(sd[:, k].mean()), seed_sd_max_over_markets=float(sd[:, k].max()), **{f'seed_sd_{m}': float(sd[i, k]) for i, m in enumerate(MARKETS)}))
    per_seed = C.reshape(5, 3, 5).mean(0)                                                   # (3 seeds, 5 models): overall composite by seed
    a, b = MODELS.index(r['first']), MODELS.index(r['second'])
    seedrows.append(dict(frame=frame, scenario=scen, model='MARGIN second-first by seed 42/43/44', seed_sd_mean_over_markets=float(np.std(per_seed[:, b] - per_seed[:, a], ddof=1)), seed_sd_max_over_markets=np.nan,
                         **{f'seed_sd_{m}': np.nan for m in MARKETS}, margins_by_seed=';'.join(f'{v:.3f}' for v in per_seed[:, b] - per_seed[:, a])))
    # control (6-pool)
    d = U[(U.frame == frame) & (U.scenario == scen)]
    cu = np.array([d[(d.market == m) & (d.seed == s) & (d.model == CTRL)].composite_6pool.iloc[0] for m, s in UNITS])
    gu = np.array([[d[(d.market == m) & (d.seed == s) & (d.model == mod)].composite_6pool.iloc[0] for mod in MODELS] for m, s in UNITS])
    gov = gu.mean(0); best = int(np.argmin(gov)); allc = np.append(gov, cu.mean())
    cf = np.array([d[(d.market == m) & (d.seed == s) & (d.model == CTRL)].fidelity_6pool.iloc[0] for m, s in UNITS]); ct = np.array([d[(d.market == m) & (d.seed == s) & (d.model == CTRL)].temporal_6pool.iloc[0] for m, s in UNITS])
    gt = np.array([[d[(d.market == m) & (d.seed == s) & (d.model == mod)].temporal_6pool.iloc[0] for mod in MODELS] for m, s in UNITS])
    cr = dict(frame=frame, scenario=scen, control_fidelity_rank_6pool=float(cf.mean()), control_temporal_rank_6pool=float(ct.mean()), generators_temporal_rank_6pool=float(gt.mean()), control_composite_6pool=float(cu.mean()), best_generator=MODELS[best], best_generator_composite_6pool=float(gov[best]),
              control_position_among_6=int(pd.Series(allc).rank(method='min').iloc[-1]), units_control_beats_best_generator=int((cu < gu.min(1)).sum()),
              units_control_position_1=int((cu < gu.min(1)).sum()), control_minus_best=float(cu.mean() - gov[best]),
              **{f'generator_{mod}_6pool': float(gov[i]) for i, mod in enumerate(MODELS)})
    if frame == 'F':
        fd = F[F.scenario == scen]; piv = fd.pivot_table(index=['market', 'seed', 'draw'], columns='model', values='composite_6pool')
        cr['draw_level_share_control_beats_best_generator'] = float((piv[CTRL] < piv[MODELS].min(axis=1)).mean()); cr['draws'] = int(len(piv))
    ctrlrows.append(cr)
S = pd.DataFrame(rows); CH = pd.DataFrame(changes); SD = pd.DataFrame(seedrows); CT = pd.DataFrame(ctrlrows)
S.to_csv(OUT / 'scenario_summary.csv', index=False); CH.to_csv(OUT / 'rank_changes_vs_published.csv', index=False); SD.to_csv(OUT / 'seed_spread.csv', index=False); CT.to_csv(OUT / 'control_composite.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(S[['frame', 'scenario', 'first', 'second', 'composite_first', 'composite_second', 'margin', 'ci_lo', 'ci_hi', 'excludes_zero', 'p_order_reversed', 'p_first_stays_first', 'cluster_ci_lo', 'cluster_ci_hi', 'cluster_excludes_zero']].round(4).to_string(index=False))
print(S[['frame', 'scenario'] + [f'composite_{m}' for m in MODELS]].round(4).to_string(index=False))
print(CH.round(4).to_string(index=False)); print(CT.round(4).to_string(index=False))
print('published overall:', pubU[~pubU.model.str.startswith('CONTROL')].set_index('model').composite_rank.round(4).to_dict())
