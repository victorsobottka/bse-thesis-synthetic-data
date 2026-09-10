"""Guard against silent notebook regressions from VS Code / Claude Code write conflicts.

REQUIRED : verified work. A failure is a REGRESSION -> blocks the commit.
PENDING  : planned work not yet applied -> reported only.

When a PENDING item is applied and verified, move its line up into REQUIRED.

Run before every commit:  python verify_notebook.py
"""
import ast
import re
import json
import sys

NB = "3_4_integrated_pipeline.ipynb"
nb = json.load(open(NB))
code_cells = [("".join(c["source"]), i)
              for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
ALL = "\n".join(s for s, _ in code_cells)


def executable_source(src: str) -> str:
    """Blank every string literal so docstrings cannot satisfy or trip a check."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


EXEC = "\n".join(executable_source(s) for s, _ in code_cells)

# Smoked_Test_5 had ~158k chars. A wholesale rollback shrinks the file even
# when individual string checks happen to pass, so guard the size too.
MIN_CODE_CHARS = 160_000

REQUIRED = [
    ("main cell unpacks 2 values",
     "all_market_results, overall = run_complete_pipeline()" in ALL),
    ("no 4-value unpack in executable code",
     "trained_models, results, summary, performance" not in EXEC),

    ("r5-7  metric taxonomy",       "FIDELITY_COLS" in ALL and "TEMPORAL_COLS" in ALL),
    ("r5-7  composite_rank",        "composite_rank" in ALL),
    ("r5-7  control excluded",      "is_ctrl" in ALL),

    ("r8    downstream utility",    "compute_downstream_utility" in ALL),
    ("r8    conditional tails",     "compute_conditional_heavy_tails" in ALL),

    ("r9    canonical return dict", "_EMPTY" in ALL),
    ("r9    WF on full series",     "full_series = np.concatenate" in ALL),
    ("r9    WF NOT on test split",  "cfg, test_flat, n_folds" not in EXEC),
    # Whitespace-tolerant: the live models_config aligns keys with padding
    # ("seq_len":      128), which this exact-substring check never matched --
    # it was only ever passing because of a second, differently-formatted
    # models_config in the now-deleted train_all_gans_pooled() dead code.
    ("r9    seq_len in config",
     bool(re.search(r'["\']seq_len["\']\s*:\s*128', ALL))),
    ("r9    zero-batch guard",      "would be evaluated untrained" in ALL),
    ("r9    no deprecated fillna",  "fillna(method=" not in EXEC),

    ("r11   generation guard (all 3 models)",
     ALL.count("High ACF(1) indicates a smooth") == 3),
    ("r17   Hill alpha capped",     "alpha > 20" in ALL),
    ("r17   NaN ranks last",        "na_option='bottom'" in ALL),
    ("r17   own warnings not silenced",
     "category=UserWarning" in ALL),

    (f"code size >= {MIN_CODE_CHARS:,}", len(ALL) >= MIN_CODE_CHARS),
    ("r11   z-score normalisation", "arctanh" in ALL),
    ("r11   Okabe-Ito palette",     "OKABE_ITO" in ALL),
    ("r11   Set2 removed",          "plt.cm.Set2" not in EXEC),
    ("r11   dynamic n_show",        "n_show" in ALL),
    ("r15   TimeGAN Recovery uses Tanh",
     "class Recovery" in ALL and "self.act_out = nn.Tanh()" in ALL),
    ("r15   round-trip assertion present",
     "round trip failed" in ALL),
    ("r19   shared return bounds", "RETURN_ACF1_MAX" in ALL),
    ("r24   no function-local time import",
     not any(l.strip().startswith(("import time", "from time"))
             and l != l.lstrip()
             for l in ALL.split("\n"))),
    ("r24   timing uses _now()", "_now()" in ALL),
    ("r26   metric diagnostics report is called",
     "generate_metric_diagnostics_report(all_market_results)" in EXEC),
    ("r26   config block present",    "GENERATOR_UPDATES" in ALL),
    ("r26   parity assertion",        "Generator-update budgets must be equal" in ALL),
    ("r26   seeds propagated",        "torch.manual_seed(self.seed)" in ALL),
    ("r26   run metadata written",    "pipeline_run_metadata.json" in ALL),
    ("r26   report artifacts centralized", 'REPORTS = ROOT / "reports"' in ALL),
    ("r26   no epochs keys in config",
     not any(k in ALL for k in ('"ae_epochs":', '"joint_epochs":', '"sup_epochs":'))),
    ("r28   stacked series rows",  "series_rows" in ALL),
    ("r28   no 2xN top grid",      "subgridspec(2, n_panels" not in EXEC),

    ("r30   results split by run type",
     'RESULTS / ("smoke" if SMOKE_TEST else "production")' in ALL),
    ("r30   resume/skip logic present",
     "_market_seed_status" in ALL and "FORCE_RERUN" in ALL
     and "run_config.json" in ALL and "discover_market_seed_dirs" in ALL),

    ("r31   exactly one models_config dict",
     ALL.count("models_config = {") == 1),
    ("r31   exactly one parity assertion",
     ALL.count("Generator-update budgets must be equal") == 1),
    ("r31   train_all_gans_pooled removed",
     "train_all_gans_pooled" not in ALL),

    ("r32   GARCH + GJR-GARCH registered",
     "GARCHModel" in ALL and "'GARCH':" in ALL and "'GJR-GARCH':" in ALL),
    ("r32   parity assertion scoped to gradient family",
     "cls.family == 'gradient'" in ALL),
    ("r32   econometric family exempt from parity",
     "cls.family == 'econometric'" in ALL),
    ("r32   reports split by run type",
     'REPORTS / ("smoke" if SMOKE_TEST else "production")' in ALL),

    # r33 -- maximum-likelihood GARCH converged (convergence_flag=0) to
    # persistence exactly on the stationarity boundary on SHANGHAI walk-forward
    # fold 3 and generated returns with Wasserstein ~75 against a run median of
    # 0.0025. The non-convergence guard never fired. See .claude/CLAUDE.md SS6.
    # A strict `>= 1.0` misses a boundary fit that lands 1.7e-13 below 1.
    ("r33   boundary trigger uses a tolerance, not >= 1.0",
     bool(re.search(r"pers\w*\s*>=\s*1\.0\s*-\s*STATIONARITY_TOL", ALL))),
    ("r33   GJR persistence includes gamma/2",
     bool(re.search(r"gamma\[1\]['\"]\]\s*/\s*2", ALL))),
    ("r33   GARCH post-generation guard raises",
     "raise GenerationSanityError" in EXEC and "GENERATION_SD_RATIO_MAX" in EXEC),
    # Every generation call site that skips a failed fold must re-raise the
    # guard, or an explosive series becomes a silently missing fold.
    ("r33   generation guard not swallowed (3 call sites)",
     EXEC.count("except GenerationSanityError") >= 3),

    # r34 -- the published Fin-GAN (Vuletic, Prenzel & Cucuringu 2024,
    # Quantitative Finance) is a different model; ours is named for its
    # architecture. Checked in every cell, markdown included.
    ("r34   CNN-WGAN-GP registered",
     "'CNN-WGAN-GP':" in ALL and "class CNN_WGAN_GP(BaseGAN)" in ALL),
    ("r34   previous model name absent from every cell",
     not any(re.search(r"\bFinGAN|FINGAN", "".join(c["source"]))
             for c in nb["cells"])),

    # r35 -- downstream utility was written per market-seed (n 486-501) under a
    # name promising the pooled test set. Now pooled across markets, once.
    ("r35   downstream utility pooled across markets",
     "def compute_pooled_downstream_utility" in ALL
     and "compute_pooled_downstream_utility(RESULTS, markets, SEEDS)" in EXEC
     and "{output_dir}/pooled_downstream_utility.csv" not in ALL),

    # r36 -- boundary fits are refit under a stationarity constraint, not raised
    # on (decision 2026-09-10; .claude/CLAUDE.md SS6). These fail on a silent
    # regression to raise-and-stop, or to generating from a boundary fit.
    ("r36   two-stage GARCH: refit class tightens arch's own stationarity row",
     "class _StationaryGARCH(_ArchGARCH)" in ALL
     and bool(re.search(r"b\[-1\]\s*=\s*-\(1\.0\s*-\s*self\._delta\)", ALL))),
    ("r36   two-stage GARCH: trigger 1e-6, bound delta 1e-4",
     bool(re.search(r"^STATIONARITY_TOL\s*=\s*1e-6", ALL, re.M))
     and bool(re.search(r"^STATIONARITY_DELTA\s*=\s*1e-4", ALL, re.M))),
    ("r36   two-stage GARCH: constrained refit only when triggered",
     "if self.constraint_applied:" in EXEC and "_StationaryGARCH" in EXEC
     and "self.unconstrained_result = res" in EXEC),
    ("r36   boundary fits no longer raise-and-stop",
     "NonStationaryFitError" not in ALL),
    ("r36   both fits persisted (params JSON + garch_persistence.csv)",
     "def fit_record" in ALL and "def persistence_row" in ALL
     and "**model.fit_record()" in ALL and "garch_persistence.csv" in ALL),

    # r37 -- library versions: the boundary knife-edge flipped between the
    # production environment and a local one, and no version was recorded.
    ("r37   numerical library versions in run metadata",
     all(f"'{lib}':" in ALL for lib in ("arch", "statsmodels", "scipy", "numpy", "pandas"))),
]

PENDING = [

]

print("REQUIRED (a failure here is a regression):")
for label, ok in REQUIRED:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")

print("\nPENDING (planned, not yet applied):")
for label, ok in PENDING:
    print(f"  {'DONE' if ok else ' -- '}  {label}")

regressions = [label for label, ok in REQUIRED if not ok]
done = [label for label, ok in PENDING if ok]

if regressions:
    print(f"\n{len(regressions)} REGRESSION(S) — do not commit:")
    for r in regressions:
        print(f"  - {r}")
    print("\nRecover with:  git checkout <last-good-commit> -- " + NB)
    sys.exit(1)

if done:
    print(f"\n{len(done)} PENDING item(s) now pass — move them into REQUIRED:")
    for d in done:
        print(f"  - {d}")

print(f"\nNo regressions. {len(PENDING) - len(done)} item(s) still pending.")
