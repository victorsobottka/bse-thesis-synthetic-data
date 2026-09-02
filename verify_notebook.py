"""Guard against silent notebook regressions from VS Code / Claude Code write conflicts.

REQUIRED : verified work. A failure is a REGRESSION -> blocks the commit.
PENDING  : planned work not yet applied -> reported only.

When a PENDING item is applied and verified, move its line up into REQUIRED.

Run before every commit:  python verify_notebook.py
"""
import ast
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
    ("r9    seq_len in config",     '"seq_len": 128' in ALL or "'seq_len': 128" in ALL),
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