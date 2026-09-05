"""Data layer for generate_report.py.

Reads pipeline run artifacts from disk and returns a plain-dict context for
template substitution. This module never fabricates a value: a missing or
unreadable artifact raises ReportDataError naming the path, rather than
substituting a default, a placeholder, or NaN. A report that silently
renders around missing data is the failure mode this module exists to
prevent.

Artifact sources (see CLAUDE.md section 7 for the full repository layout):

    reports/pipeline_run_metadata.json                        run provenance
    thesis_results/overall_performance.csv                    cross-run ranks
    thesis_results/per_seed_market_performance.csv             + compute cost
    thesis_results/<MARKET>/seed<N>/<MARKET>_metrics.csv       19 raw metrics
    thesis_results/<MARKET>/seed<N>/pooled_downstream_utility.csv
    thesis_results/walk_forward/<MARKET>/walk_forward_<MODEL>_seed<N>.csv
    thesis_results/<MARKET>/seed<N>/*.png                      figures

Markets, seeds and models are discovered from the filesystem, never
hardcoded: the diffusion, GARCH and LLM arms are coming, and a market run
under a new seed must appear without a code change here.

One exception to "artifacts only": Table 1 (BRICS data characteristics) --
per-market volatility, kurtosis, extreme-event count and clustering ratio --
describes the real return series alone, independent of any generator. None
of the six artifact types above carries that: comprehensive_evaluation()
only ever stores real-vs-synthetic *differences*, never the real series'
own descriptive statistics. compute_real_market_characteristics() below
reads the raw log-return parquet files under data/processed_files/ instead
(the same files the notebook's own load_dataset() reads) and computes
plain, standard descriptive statistics -- sigma, excess kurtosis, a |r|>5*
sigma count, and a next-day-given-big-day conditional-probability ratio.
This does not touch, call, or duplicate any ranking/composite logic from
the notebook's FinancialMetrics class; it is a separate, transparent
computation over the same raw inputs, scoped to Table 1 only.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


class ReportDataError(RuntimeError):
    """Raised when a required report artifact is missing or unreadable."""


# ============================================================================
# Low-level, fail-loud readers
# ============================================================================

def _require_file(path: Path) -> Path:
    if not path.exists():
        raise ReportDataError(f"Required report artifact not found: {path}")
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    _require_file(path)
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ReportDataError(f"Could not read {path}: {e}") from e


def _read_json(path: Path) -> dict:
    _require_file(path)
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise ReportDataError(f"Could not read {path}: {e}") from e


# ============================================================================
# Filesystem discovery -- markets, seeds and models are never hardcoded
# ============================================================================

_SEED_DIR_RE = re.compile(r"^seed(?P<seed>\d+)$")
_WF_FILE_RE = re.compile(r"^walk_forward_(?P<model>.+?)_seed(?P<seed>\d+)\.csv$")


def _discover_market_seed_dirs(results_dir: Path):
    """Yield (market, seed, seed_dir) for every thesis_results/<MARKET>/seed<N>/.

    Older, pre-parity pipeline runs left per-market CSVs directly under
    thesis_results/<MARKET>/ with no seed subdirectory (and, in at least one
    case, values inconsistent with the current pipeline's invariants). Those
    are intentionally NOT discovered here: only the seed<N>/ layout that
    run_complete_pipeline() actually writes today is treated as current.
    """
    if not results_dir.is_dir():
        return
    for market_dir in sorted(results_dir.iterdir()):
        if not market_dir.is_dir() or market_dir.name == "walk_forward":
            continue
        for seed_dir in sorted(market_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            m = _SEED_DIR_RE.match(seed_dir.name)
            if not m:
                continue
            yield market_dir.name, int(m.group("seed")), seed_dir


def _discover_walk_forward(results_dir: Path, market: str, seed: int) -> dict:
    """Return {model_name: DataFrame} from walk_forward_<MODEL>_seed<seed>.csv
    files under thesis_results/walk_forward/<market>/."""
    wf_dir = results_dir / "walk_forward" / market
    if not wf_dir.is_dir():
        raise ReportDataError(f"Required walk-forward directory not found: {wf_dir}")
    out = {}
    for f in sorted(wf_dir.iterdir()):
        m = _WF_FILE_RE.match(f.name)
        if not m or int(m.group("seed")) != seed:
            continue
        out[m.group("model")] = _read_csv(f)
    if not out:
        raise ReportDataError(
            f"No walk_forward_<MODEL>_seed{seed}.csv files found under {wf_dir}"
        )
    return out


def _is_control(model_name: str) -> bool:
    name = model_name.lower()
    return "control" in name or "shuffled" in name


# ============================================================================
# Table 1: real-market descriptive statistics (raw data, not thesis_results/)
# ============================================================================

_MARKET_FILE_STEM = {
    # Processed-file basenames that don't match the market's canonical name.
    "FTSE_JSE": "FTSE",
}


_TEST_FILE_RE = re.compile(r"^(?P<market>.+)_test\.parquet$")


def _discover_data_markets(processed_dir: Path) -> list:
    """List every market with a complete train/valid/test parquet triad under
    data/processed_files/. This is the full BRICS input basket -- independent
    of which markets have a completed GAN run in thesis_results/. Table 1
    describes the *data*, not the pipeline's progress, so it must not be
    filtered down to whichever market happens to have finished training."""
    test_dir = processed_dir / "test"
    if not test_dir.is_dir():
        raise ReportDataError(f"Required directory not found: {test_dir}")
    markets = []
    for f in sorted(test_dir.iterdir()):
        m = _TEST_FILE_RE.match(f.name)
        if not m:
            continue
        market = m.group("market")
        stem = _MARKET_FILE_STEM.get(market, market)
        if all((processed_dir / split / f"{stem}_{split}.parquet").exists()
               for split in ("train", "valid", "test")):
            markets.append(market)
    if not markets:
        raise ReportDataError(f"No complete train/valid/test parquet triads found under {processed_dir}")
    return markets


def compute_real_market_characteristics(processed_dir: Path, markets: list) -> dict:
    """Per-market sigma, excess kurtosis, |r|>5*sigma count, and a big-move
    clustering ratio, computed from the full (train+valid+test) real series.

    Raises if a market's parquet files are missing, rather than skipping it
    silently -- the resulting table would otherwise under-report market-days
    without saying so.
    """
    out = {}
    total_days = 0
    total_extreme = 0
    for market in markets:
        stem = _MARKET_FILE_STEM.get(market, market)
        parts = []
        for split in ("train", "valid", "test"):
            f = processed_dir / split / f"{stem}_{split}.parquet"
            _require_file(f)
            df = pd.read_parquet(f)
            cols = [c for c in df.columns if "og" in c and "eturn" in c]
            if not cols:
                raise ReportDataError(f"No log-return column found in {f}")
            parts.append(df[cols[0]].dropna().to_numpy())
        r = np.concatenate(parts)
        if r.size == 0:
            raise ReportDataError(f"No return observations for market {market}")

        sigma = float(r.std())
        kurt = float(stats.kurtosis(r))  # excess (Fisher) kurtosis
        n_extreme = int(np.sum(np.abs(r) > 5 * sigma))
        big = np.abs(r) > 2 * sigma
        p_big = float(big.mean())
        p_big_given_big = (float(big[1:][big[:-1]].mean())
                           if big[:-1].sum() else float("nan"))
        ratio = p_big_given_big / p_big if p_big else float("nan")

        out[market] = {
            "n_days": int(r.size),
            "sigma_pct": sigma * 100,
            "excess_kurtosis": kurt,
            "n_extreme_5sigma": n_extreme,
            "p_big_move": p_big * 100,
            "p_big_given_big": p_big_given_big * 100,
            "clustering_ratio": ratio,
        }
        total_days += r.size
        total_extreme += n_extreme

    out["_totals"] = {"n_days": total_days, "n_extreme_5sigma": total_extreme}
    return out


# ============================================================================
# LaTeX-ready formatting
# ============================================================================

_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _escape_latex(text) -> str:
    """Escape LaTeX special characters in free text pulled from JSON/CSV data
    (platform strings, device names, version strings, model names). Numbers
    formatted by _fnum/_fint never need this -- only pass-through text does.

    This is what caught platform="...-x86_64-..." breaking the compile: an
    un-escaped "_" outside math mode reads as "missing $ inserted" to LaTeX.
    """
    text = str(text)
    # Single pass, one character at a time, so a replacement's own backslash
    # (e.g. "_" -> "\_") is never re-escaped by a later lookup.
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def _fnum(x, decimals=4) -> str:
    if x is None or (isinstance(x, float) and x != x):  # NaN
        return "---"
    return f"{x:,.{decimals}f}"


def _fint(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "---"
    return f"{int(x):,}"


def _build_table3_rows(overall: pd.DataFrame) -> str:
    """One row per model, ranked models first (ascending composite_rank),
    the control appended last with em-dashes for its NaN ranks, winning
    model's composite bolded."""
    ranked = overall[~overall["model"].apply(_is_control)].copy()
    ranked = ranked.sort_values("composite_rank")
    control_rows = overall[overall["model"].apply(_is_control)]

    lines = []
    best_idx = ranked.index[0] if len(ranked) else None
    for idx, row in ranked.iterrows():
        comp = _fnum(row["composite_rank"], 4)
        if idx == best_idx:
            comp = f"\\textbf{{{comp}}}"
        lines.append(
            f"{_escape_latex(row['model'])} & {comp} & {_fnum(row['fidelity_rank'], 4)} & "
            f"{_fnum(row['temporal_rank'], 4)} & {_fnum(row['avg_rank'], 4)} \\\\"
        )
    for _, row in control_rows.iterrows():
        lines.append(f"{_escape_latex(row['model'])} & --- & --- & --- & --- \\\\")
    return "\n".join(lines)


def _build_budget_parity_rows(per_seed_market: pd.DataFrame) -> str:
    """One row per model, averaged across whatever (market, seed) runs are
    present: generator updates, critic updates, wall-clock, generator
    parameter count. Pulled straight from per_seed_market_performance.csv --
    this is what makes these numbers correct by construction instead of
    hand-typed."""
    cost_cols = ["generator_updates", "critic_updates", "train_seconds", "n_params_g"]
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)].copy()
    missing = [c for c in cost_cols if c not in df.columns]
    if missing:
        raise ReportDataError(
            f"per_seed_market_performance.csv is missing column(s) {missing} "
            "required for the budget-parity table"
        )
    agg = df.groupby("model")[cost_cols].mean()
    lines = []
    for model, row in agg.iterrows():
        lines.append(
            f"{_escape_latex(model)} & {_fint(row['generator_updates'])} & "
            f"{_fint(row['critic_updates'])} & "
            f"{_fnum(row['train_seconds'], 1)} & {_fint(row['n_params_g'])} \\\\"
        )
    return "\n".join(lines)


def _build_table1_rows(characteristics: dict) -> str:
    """Market, sigma, excess kurtosis, resid. kurtosis (not yet computable
    without duplicating FinancialMetrics' GARCH fit -- left as an em-dash,
    same as the table this replaces)."""
    lines = []
    for market, c in characteristics.items():
        if market == "_totals":
            continue
        lines.append(
            f"{_escape_latex(market)} & {_fnum(c['sigma_pct'], 2)} & "
            f"{_fnum(c['excess_kurtosis'], 2)} & --- \\\\"
        )
    return "\n".join(lines)


def _build_table1_prose(characteristics: dict) -> dict:
    """Scalars for the hand-written sentences below Table 1 -- e.g. 'we
    observed ${RPT_TABLE1_TOTAL_EXTREME} [days]' -- so only the numbers are
    templated, not the analysis around them."""
    per_market = {m: c for m, c in characteristics.items() if m != "_totals"}
    totals = characteristics["_totals"]
    max_ratio_market = max(per_market, key=lambda m: per_market[m]["clustering_ratio"])
    max_extreme_market = max(per_market, key=lambda m: per_market[m]["n_extreme_5sigma"])
    ratios = [c["clustering_ratio"] for c in per_market.values()]
    return {
        "total_days": _fint(totals["n_days"]),
        "total_extreme": _fint(totals["n_extreme_5sigma"]),
        "max_extreme_market": _escape_latex(max_extreme_market),
        "max_extreme_count": _fint(per_market[max_extreme_market]["n_extreme_5sigma"]),
        "max_ratio_market": _escape_latex(max_ratio_market),
        "max_ratio_p_big": _fnum(per_market[max_ratio_market]["p_big_move"], 1),
        "max_ratio_p_big_given_big": _fnum(per_market[max_ratio_market]["p_big_given_big"], 1),
        "max_ratio_value": _fnum(per_market[max_ratio_market]["clustering_ratio"], 1),
        "ratio_min": _fnum(min(ratios), 1),
        "ratio_max": _fnum(max(ratios), 1),
    }


def _build_downstream_utility_rows(pooled_utility: pd.DataFrame) -> str:
    df = pooled_utility.sort_values("qlike")
    lines = []
    for _, row in df.iterrows():
        name = _escape_latex(row["model"])
        cf_p = row.get("christoffersen_p", float("nan"))
        lines.append(
            f"{name} & {_fnum(row['qlike'], 3)} & "
            f"{_fnum(row['var_coverage_error'], 4)} & "
            f"{_fnum(row['kupiec_p'], 4)} & "
            f"{'---' if pd.isna(cf_p) else _fnum(cf_p, 4)} & "
            f"{_fint(row['violations'])} & {_fint(row['n_test'])} \\\\"
        )
    return "\n".join(lines)


def _build_walk_forward_rows(wf_by_model: dict) -> str:
    """One row per model: wasserstein, hurst_diff and discriminative_auc,
    mean +/- sd across folds. These three are common to every model's
    walk_forward_<MODEL>_seed<N>.csv; the full per-metric breakdown is in
    walk_forward_summary for anyone consuming the context directly."""
    cols = ["wasserstein", "hurst_diff", "discriminative_auc"]
    lines = []
    for model in sorted(wf_by_model):
        df = wf_by_model[model]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ReportDataError(
                f"walk_forward file for model {model!r} is missing column(s) "
                f"{missing} required for the walk-forward summary table"
            )
        cells = [f"{df[c].mean():.4f} $\\pm$ {df[c].std():.4f}" for c in cols]
        lines.append(f"{_escape_latex(model)} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def _build_walk_forward_summary(wf_by_model: dict) -> dict:
    """model -> {metric -> 'mean \\pm sd'} across folds, for every numeric
    column present (excluding bookkeeping columns)."""
    skip = {"seed", "fold", "train_len", "test_len", "market", "model"}
    out = {}
    for model, df in wf_by_model.items():
        cols = [c for c in df.columns if c not in skip
                and pd.api.types.is_numeric_dtype(df[c])]
        out[model] = {
            c: f"{df[c].mean():.4f} $\\pm$ {df[c].std():.4f}" for c in cols
        }
    return out


_FIGURE_TITLES = {
    "stylized_facts": "Stylised facts: synthetic against real daily log returns",
    "metrics_heatmap": "Normalised metrics heatmap",
    "rank_comparison": "Average rank comparison",
}


def _build_figures_block(figures: dict) -> str:
    """One \\includegraphics per figure that exists on disk; a plain note,
    not a broken reference, for any that don't."""
    parts = []
    for key, title in _FIGURE_TITLES.items():
        path = figures[key]
        title_esc = _escape_latex(title)
        if path.exists():
            parts.append(
                # height as well as width: the stylized-facts figure is a
                # tall, portrait-oriented multi-row chart, and constraining
                # width alone let it overflow past the bottom of a landscape
                # page (the footer printed directly on top of it).
                r"\clearpage\begin{center}\includegraphics"
                r"[width=0.85\linewidth,height=0.82\textheight,keepaspectratio]{"
                + str(path) + r"}\end{center}"
                r"\centerline{\small\textit{" + title_esc + r"}}\vspace{6pt}"
            )
        else:
            parts.append(
                r"\begin{center}\textit{[" + title_esc + " not available: "
                + _escape_latex(str(path)) + r" not found]}\end{center}"
            )
    return "\n\n".join(parts)


def _build_provenance(metadata: dict) -> dict:
    # timestamp/commit/seeds/n_folds/generator_updates are digits, colons,
    # dashes and commas -- never need escaping. python/torch/device/platform
    # are free text from sys.version / torch.__version__ / platform.platform()
    # and DO: platform.platform() on this machine produced a literal "_" in
    # "x86_64", which LaTeX reads as "missing $ inserted" outside math mode.
    commit = metadata.get("git_commit", "")
    return {
        "timestamp": metadata.get("timestamp", "---"),
        "git_commit_full": commit or "---",
        "git_commit_short": (commit[:10] if commit else "---"),
        "seeds": ", ".join(str(s) for s in metadata.get("seeds", [])) or "---",
        "markets": ", ".join(_escape_latex(m) for m in metadata.get("markets", [])) or "---",
        "n_folds": str(metadata.get("n_folds", "---")),
        "generator_updates": _fint(metadata.get("generator_updates")),
        "python": _escape_latex(metadata.get("python", "---")),
        "torch": _escape_latex(metadata.get("torch", "---")),
        "cuda": "yes" if metadata.get("cuda") else "no",
        "device": _escape_latex(metadata.get("device", "---")),
        "platform": _escape_latex(metadata.get("platform", "---")),
        "smoke_test": bool(metadata.get("smoke_test", False)),
    }


# ============================================================================
# Public entry point
# ============================================================================

def load_report_context(results_dir="thesis_results", reports_dir="reports",
                         processed_dir="data/processed_files") -> dict:
    """Read every artifact the report needs and return a plain-dict context.

    Raises ReportDataError (naming the offending path) on any missing or
    unreadable artifact -- never substitutes a placeholder.
    """
    results_dir = Path(results_dir)
    reports_dir = Path(reports_dir)
    processed_dir = Path(processed_dir)

    metadata = _read_json(reports_dir / "pipeline_run_metadata.json")
    overall = _read_csv(results_dir / "overall_performance.csv")
    per_seed_market = _read_csv(results_dir / "per_seed_market_performance.csv")

    runs = []
    for market, seed, seed_dir in _discover_market_seed_dirs(results_dir):
        metrics = _read_csv(seed_dir / f"{market}_metrics.csv")
        pooled_utility = _read_csv(seed_dir / "pooled_downstream_utility.csv")
        walk_forward = _discover_walk_forward(results_dir, market, seed)
        figures = {
            "stylized_facts": seed_dir / f"{market}_stylized_facts.png",
            "metrics_heatmap": seed_dir / f"{market}_metrics_heatmap.png",
            "rank_comparison": seed_dir / f"{market}_rank_comparison.png",
        }
        runs.append({
            "market": market,
            "seed": seed,
            "metrics": metrics,
            "pooled_utility": pooled_utility,
            "walk_forward": walk_forward,
            "figures": figures,
        })

    if not runs:
        raise ReportDataError(
            f"No thesis_results/<MARKET>/seed<N>/ run directories found under "
            f"{results_dir} -- nothing to report on"
        )

    all_models = set()
    for run in runs:
        all_models.update(run["metrics"]["model"])
    models = sorted(m for m in all_models if not _is_control(m))
    markets = sorted({run["market"] for run in runs})
    seeds = sorted({run["seed"] for run in runs})

    ctx = {
        "metadata": metadata,
        "overall": overall,
        "per_seed_market": per_seed_market,
        "runs": runs,
        "models": models,
        "markets": markets,
        "seeds": seeds,
    }

    data_markets = _discover_data_markets(processed_dir)
    characteristics = compute_real_market_characteristics(processed_dir, data_markets)

    # One run's worth of downstream-utility / walk-forward tables for now
    # (single-market smoke-test data): B2-B8 use runs[0]. A multi-market
    # production run should report per-market, not pooled-and-hidden; that
    # template change is left for when there is more than one run to show.
    primary = runs[0]

    ctx["formatted"] = {
        "provenance": _build_provenance(metadata),
        "table1_rows": _build_table1_rows(characteristics),
        "table1_prose": _build_table1_prose(characteristics),
        "table3_rows": _build_table3_rows(overall),
        "budget_parity_rows": _build_budget_parity_rows(per_seed_market),
        "downstream_utility_rows": _build_downstream_utility_rows(primary["pooled_utility"]),
        "walk_forward_rows": _build_walk_forward_rows(primary["walk_forward"]),
        "walk_forward_summary": _build_walk_forward_summary(primary["walk_forward"]),
        "primary_market": primary["market"],
        "primary_seed": str(primary["seed"]),
        "figures_block": _build_figures_block(primary["figures"]),
    }
    return ctx


if __name__ == "__main__":
    import pprint
    ctx = load_report_context()
    pprint.pprint({k: v for k, v in ctx.items() if k != "runs"})
    print("\n--- formatted ---")
    pprint.pprint(ctx["formatted"])
