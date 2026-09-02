"""Guard against silent notebook regressions from VS Code / Claude Code write conflicts."""
import json, sys

nb = json.load(open("3_4_integrated_pipeline.ipynb"))
cells = [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]
src = {i: s for i, (s, t) in enumerate(cells) if t == "code"}
ALL = "\n".join(src.values())

CHECKS = [
    # (label, condition)
    ("main cell unpacks 2 values",
     "all_market_results, overall = run_complete_pipeline()" in ALL),
    ("main cell does NOT unpack 4",
     "trained_models, results, summary, performance" not in ALL),
    ("rounds 5-7: metric taxonomy",      "FIDELITY_COLS" in ALL and "TEMPORAL_COLS" in ALL),
    ("rounds 5-7: composite_rank",       "composite_rank" in ALL),
    ("rounds 5-7: control excluded",     "is_ctrl" in ALL),
    ("round 8:   downstream utility",    "compute_downstream_utility" in ALL),
    ("round 8:   conditional tails",     "compute_conditional_heavy_tails" in ALL),
    ("round 9:   canonical return dict", "_EMPTY" in ALL),
    ("round 9:   WF on full series",     "full_series = np.concatenate" in ALL),
    ("round 9:   seq_len in config",     '"seq_len": 128' in ALL or "'seq_len': 128" in ALL),
    ("round 9:   zero-batch guard",      "would be evaluated untrained" in ALL),
    ("round 9:   no deprecated fillna",  "fillna(method=" not in ALL),
    ("round 11:  z-score normalisation", "arctanh" in ALL),
    ("round 11:  Okabe-Ito palette",     "OKABE_ITO" in ALL),
    ("round 11:  Set2 removed",          "plt.cm.Set2" not in ALL),
    ("round 11:  dynamic n_show",        "n_show" in ALL),
    ("round 11:  generation guard",      "mode collapse" in ALL),
]

failed = [label for label, ok in CHECKS if not ok]
for label, ok in CHECKS:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
if failed:
    print(f"\n{len(failed)} REGRESSION(S) — do not commit:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll checks pass.")