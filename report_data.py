"""Data layer for generate_report.py.

Reads pipeline run artifacts from disk and returns a plain-dict context for
template substitution. This module never fabricates a value: a missing or
unreadable artifact raises ReportDataError naming the path, rather than
substituting a default, a placeholder, or NaN. A report that silently
renders around missing data is the failure mode this module exists to
prevent.

Artifact sources (see CLAUDE.md section 7 for the full repository layout).
thesis_results/ is split production/ vs smoke/ (run_complete_pipeline picks
one based on SMOKE_TEST); the default results_dir below is production/, and
that is the only tree meant to back a real report:

    reports/pipeline_run_metadata.json                            run provenance
    thesis_results/production/overall_performance.csv             cross-run ranks
    thesis_results/production/per_seed_market_performance.csv      + compute cost
    thesis_results/production/<MARKET>/seed<N>/<MARKET>_metrics.csv  19 raw metrics
    thesis_results/production/pooled_downstream_utility.csv       pooled TSTR backtest
      (runs before 2026-09-10 have <MARKET>/seed<N>/pooled_downstream_utility.csv
       instead -- one market's backtest each, despite the name; see _load_downstream)
    thesis_results/production/walk_forward/<MARKET>/walk_forward_<MODEL>_seed<N>.csv
    thesis_results/production/<MARKET>/seed<N>/*.png               figures

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


def discover_market_seed_dirs(results_dir: Path):
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


def _discover_walk_forward(results_dir: Path, market: str, seed: int,
                           run_models: set, stale: list) -> dict:
    """Return {model_name: DataFrame} from walk_forward_<MODEL>_seed<seed>.csv
    files under thesis_results/walk_forward/<market>/, for the models this
    (market, seed) run actually evaluated.

    The directory is shared by every run that ever wrote to it, and a rerun
    overwrites its own files without deleting anyone else's. A model that has
    since been renamed or removed (walk_forward_<OLD>_seed42.csv beside a new
    walk_forward_<NEW>_seed42.csv) would otherwise be read as an extra
    model and its old folds mixed into this run's walk-forward tables. So the
    model list comes from the run's own metrics CSV; files for any other model
    are skipped and recorded in `stale` so the caller can say so, never used.
    A model the run evaluated but has no walk-forward file for is an error.
    """
    wf_dir = results_dir / "walk_forward" / market
    if not wf_dir.is_dir():
        raise ReportDataError(f"Required walk-forward directory not found: {wf_dir}")
    wanted = {m for m in run_models if not _is_control(m)}
    out = {}
    for f in sorted(wf_dir.iterdir()):
        m = _WF_FILE_RE.match(f.name)
        if not m or int(m.group("seed")) != seed:
            continue
        if m.group("model") not in wanted:
            stale.append(str(f))
            continue
        out[m.group("model")] = _read_csv(f)
    missing = sorted(wanted - set(out))
    if missing:
        raise ReportDataError(
            f"No walk-forward file for model(s) {missing} evaluated in "
            f"{market} seed {seed}: expected walk_forward_<MODEL>_seed{seed}.csv under {wf_dir}"
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


def _best_composite(overall: pd.DataFrame) -> str:
    ranked = overall[~overall["model"].apply(_is_control)]
    if ranked.empty:
        return "---"
    return _fnum(ranked["composite_rank"].min(), 3)


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


# Caption per figure. These are full captions, not labels: the survival panel
# in particular is unreadable without being told that a straight line on
# log-log axes is a power-law tail and that its slope is the tail index --
# which is what makes tail_index_diff visually checkable rather than a number
# the reader has to take on trust.
_FIGURE_TITLES = {
    "stylized_facts": (
        "Stylised facts: every series against real daily log returns. The top "
        "block is one full-width row per series --- real, each generator, and "
        "the shuffled control --- on a single shared y-scale; the x-axis is a "
        "common index, not a common time, since real and synthetic are "
        "independent unconditional draws and vertical alignment between rows "
        "carries no correspondence. The tail panel plots the empirical "
        "survival function $P(|r| > x)$ on log-log axes: a power-law tail is a "
        "straight line there, and its slope is the tail index $\\alpha$, so a "
        "model whose line runs parallel to the real series matches the tail "
        "exponent while one that falls away below it under-produces extremes. "
        "Lines and axis are cut at roughly $1/n$, because the largest few "
        "order statistics are single-point estimates of an extreme "
        "probability; the trim affects the drawn line only, never a metric."),
    "metrics_heatmap": (
        "Normalised metrics heatmap, one column per metric and one row per "
        "series. Colour is within-metric normalised, so it compares models on "
        "a metric and never metrics with each other."),
    "rank_comparison": (
        "Average rank across all ranked metrics. Shown for comparability "
        "only: avg\\_rank is not the selection criterion, because an "
        "unweighted mean over metrics is won by the shuffled control. Model "
        "selection uses composite\\_rank."),
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
                # page (the footer printed directly on top of it). It grew
                # from five rows to seven when the two econometric baselines
                # landed, so it is taller still than when this was written.
                r"\clearpage\begin{center}\includegraphics"
                r"[width=0.85\linewidth,height=0.78\textheight,keepaspectratio]{"
                + str(path) + r"}\end{center}"
                # Captions run to several sentences now, so they are set as a
                # justified paragraph rather than a centred one-liner.
                r"\begin{center}\begin{minipage}{0.88\linewidth}"
                r"\small\textit{" + title_esc + r"}\end{minipage}\end{center}"
                r"\vspace{6pt}"
            )
        else:
            parts.append(
                r"\begin{center}\textit{[" + title_esc + " not available: "
                + _escape_latex(str(path)) + r" not found]}\end{center}"
            )
    return "\n\n".join(parts)


_LIBRARIES = ("arch", "statsmodels", "scipy", "numpy", "pandas")


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
        # Numerical libraries: recorded because the GARCH boundary knife-edge
        # flipped between environments. Runs before 2026-09-10 lack them.
        "libraries": (" $\\cdot$ ".join(
            f"{lib} {_escape_latex(metadata[lib])}" for lib in _LIBRARIES if metadata.get(lib))
            or "not recorded in this run's metadata"),
        "n_markets": str(len(metadata.get("markets", []))),
        "n_seeds": str(len(metadata.get("seeds", []))),
        # TimeGAN pre-training is reported separately from the parity budget,
        # so the report needs it as its own number rather than folded in.
        "ae_steps": _fint(_timegan_step(metadata, "ae_steps")),
        "sup_steps": _fint(_timegan_step(metadata, "sup_steps")),
    }


def _timegan_step(metadata: dict, key: str):
    cfg = metadata.get("models_config", {}) or {}
    for name, params in cfg.items():
        if "timegan" in str(name).lower() and key in params:
            return params[key]
    return float("nan")


# ============================================================================
# Model families
#
# The benchmark now spans two families that are not comparable on training
# budget: gradient-trained generators (TimeGAN/QuantGAN/CNN-WGAN-GP), which take a
# specified number of generator updates, and econometric generators
# (GARCH/GJR-GARCH), fitted by maximum likelihood, for which a "generator
# update" has no analogue. The split is read from the data -- a model with no
# generator_updates recorded is econometric -- never from a hardcoded name
# list, so a diffusion or LLM arm lands in the right family without a code
# change here.
# ============================================================================

def _model_families(per_seed_market: pd.DataFrame) -> dict:
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)]
    if "generator_updates" not in df.columns:
        raise ReportDataError(
            "per_seed_market_performance.csv has no generator_updates column; "
            "cannot separate gradient-trained from econometric models"
        )
    out = {}
    for model, grp in df.groupby("model"):
        out[str(model)] = ("gradient" if grp["generator_updates"].notna().any()
                           else "econometric")
    return out


def _ranked(overall: pd.DataFrame) -> pd.DataFrame:
    return overall[~overall["model"].apply(_is_control)].sort_values("composite_rank")


# ============================================================================
# Headline result: ranks, per-market-seed wins, and fit cost in one table
# ============================================================================

def _win_counts(per_seed_market: pd.DataFrame, rank_col: str) -> dict:
    """model -> number of (market, seed) cells this model wins on rank_col.

    Counted over cells, not averaged over models: 'GJR-GARCH wins 3 of 15'
    is a claim about how often it is best, which a mean rank cannot make.
    """
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)].copy()
    for col in ("market", "seed", rank_col):
        if col not in df.columns:
            raise ReportDataError(
                f"per_seed_market_performance.csv is missing column {col!r} "
                f"required for {rank_col} win counts"
            )
    df = df.dropna(subset=[rank_col])
    if df.empty:
        raise ReportDataError(f"No non-NaN {rank_col} values to count wins over")
    winners = df.loc[df.groupby(["market", "seed"])[rank_col].idxmin(), "model"]
    return {str(k): int(v) for k, v in winners.value_counts().items()}


def _n_cells(per_seed_market: pd.DataFrame) -> int:
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)]
    return int(df.groupby(["market", "seed"]).ngroups)


def _fmt_seconds(x) -> str:
    """Fit cost spans four orders of magnitude here (0.017 s for GJR-GARCH,
    465 s for QuantGAN). A fixed number of decimals either prints '0.0' for
    the econometric models or six digits of noise for the GANs, so the
    precision follows the magnitude."""
    if x is None or (isinstance(x, float) and x != x):
        return "---"
    if x < 1:
        return f"$<1$"
    return f"{x:,.0f}"


def _build_headline_rows(overall: pd.DataFrame, per_seed_market: pd.DataFrame,
                         families: dict) -> str:
    """The report's lead table: composite/fidelity/temporal rank, composite
    wins out of all (market, seed) cells, family, and mean fit seconds."""
    wins = _win_counts(per_seed_market, "composite_rank")
    cost = (per_seed_market[~per_seed_market["model"].apply(_is_control)]
            .groupby("model")["train_seconds"].mean())
    ranked = _ranked(overall)
    n = _n_cells(per_seed_market)
    lines = []
    best_idx = ranked.index[0] if len(ranked) else None
    for idx, row in ranked.iterrows():
        model = str(row["model"])
        comp = _fnum(row["composite_rank"], 3)
        if idx == best_idx:
            comp = f"\\textbf{{{comp}}}"
        fam = "econometric" if families.get(model) == "econometric" else "gradient"
        lines.append(
            f"{_escape_latex(model)} & {fam} & {comp} & "
            f"{_fnum(row['fidelity_rank'], 3)} & {_fnum(row['temporal_rank'], 3)} & "
            f"{wins.get(model, 0)}\\,/\\,{n} & "
            f"{_fmt_seconds(cost.get(model, float('nan')))} \\\\"
        )
    return "\n".join(lines)


def _build_win_prose(per_seed_market: pd.DataFrame, families: dict) -> dict:
    """Scalars for the sentences that report the family split, so the claim
    'the econometric family takes 8 of 15' is counted, never typed."""
    n = _n_cells(per_seed_market)
    out = {"n_cells": str(n)}
    for label, col in (("composite", "composite_rank"),
                       ("temporal", "temporal_rank"),
                       ("fidelity", "fidelity_rank")):
        wins = _win_counts(per_seed_market, col)
        econ = sum(v for m, v in wins.items() if families.get(m) == "econometric")
        grad = sum(v for m, v in wins.items() if families.get(m) != "econometric")
        out[f"{label}_wins"] = ", ".join(
            f"{_escape_latex(m)} {v}" for m, v in
            sorted(wins.items(), key=lambda kv: (-kv[1], kv[0])))
        out[f"{label}_econ"] = str(econ)
        out[f"{label}_grad"] = str(grad)
    return out


# ============================================================================
# Budget sensitivity: who wins among the gradient-trained models alone
# ============================================================================

def _build_gan_only_prose(per_seed_market: pd.DataFrame, families: dict) -> dict:
    """Win counts restricted to the gradient-trained family.

    Needed because the question "does the budget you evaluate at decide the
    winner?" is a question about the GANs, and the econometric baselines are
    not on a budget at all. The composite_rank values compared here were
    computed across ALL models present, so this is the ordering of the three
    GANs within the five-model ranking -- not a re-ranking among three, which
    would require recomputing ranks and is deliberately not done here (ranking
    logic belongs to the pipeline, not to the report's data layer).
    """
    grad = [m for m, f in families.items() if f == "gradient"]
    df = per_seed_market[per_seed_market["model"].isin(grad)].dropna(
        subset=["composite_rank"])
    if df.empty:
        raise ReportDataError("No gradient-trained models with a composite_rank")
    winners = df.loc[df.groupby(["market", "seed"])["composite_rank"].idxmin(), "model"]
    counts = winners.value_counts()
    n = int(df.groupby(["market", "seed"]).ngroups)
    return {
        "n_cells": str(n),
        "counts": ", ".join(f"{_escape_latex(str(m))} {v}\\,/\\,{n}"
                            for m, v in counts.items()),
        "leader": _escape_latex(str(counts.idxmax())),
        "leader_wins": str(int(counts.max())),
        "n_models": str(len(grad)),
    }


# ============================================================================
# Compute versus performance
# ============================================================================

def _build_compute_rows(overall: pd.DataFrame, per_seed_market: pd.DataFrame,
                        families: dict) -> str:
    """Parameter count, mean fit seconds and composite rank, ordered by cost.

    Econometric rows carry their real fitted-parameter count and an explicit
    'n/a' for generator updates rather than a fabricated update count: an ML
    fit has no gradient-step analogue, and printing one would invent a number
    the run never produced.
    """
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)].copy()
    for col in ("n_params_g", "train_seconds", "generator_updates"):
        if col not in df.columns:
            raise ReportDataError(
                f"per_seed_market_performance.csv is missing {col!r}, required "
                "for the compute-versus-performance table"
            )
    agg = df.groupby("model")[["n_params_g", "train_seconds", "generator_updates"]].mean()
    comp = overall.set_index("model")["composite_rank"]
    lines = []
    for model, row in agg.sort_values("train_seconds").iterrows():
        model = str(model)
        econ = families.get(model) == "econometric"
        lines.append(
            f"{_escape_latex(model)} & {'econometric' if econ else 'gradient'} & "
            f"{_fint(row['n_params_g'])} & "
            f"{'n/a' if econ else _fint(row['generator_updates'])} & "
            f"{_fnum(row['train_seconds'], 3) if row['train_seconds'] < 1 else _fnum(row['train_seconds'], 1)} & "
            f"{_fnum(comp.get(model, float('nan')), 3)} \\\\"
        )
    return "\n".join(lines)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _build_compute_prose(per_seed_market: pd.DataFrame, overall: pd.DataFrame,
                         families: dict) -> dict:
    """The claims the text makes about who ranks where, and at what cost.

    Every model named in a sentence is found here from the data -- the
    composite winner, the family leaders, the largest model -- rather than
    typed into the template. The parameter and wall-clock ratios are both taken
    against the *same* large model, so a sentence cannot pair one model's
    parameter count with a different model's training time.
    """
    df = per_seed_market[~per_seed_market["model"].apply(_is_control)]
    agg = df.groupby("model")[["n_params_g", "train_seconds"]].mean()
    ranked = _ranked(overall)
    ov = ranked.set_index("model")
    best_model = str(ranked.iloc[0]["model"])
    biggest = str(agg["n_params_g"].idxmax())
    fid_leader = str(ov["fidelity_rank"].idxmin())
    tmp_leader = str(ov["temporal_rank"].idxmin())

    below = [lab for lab, col in (("composite", "composite_rank"),
                                  ("fidelity", "fidelity_rank"),
                                  ("temporal", "temporal_rank"))
             if ov.loc[biggest, col] > ov.loc[best_model, col]]
    if len(below) == 3:
        below_txt = "on composite, on fidelity and on temporal"
    elif below:
        below_txt = "on " + " and ".join(below) + " only"
    else:
        below_txt = "on none of composite, fidelity or temporal"

    # Composite position of each econometric model, and whether the leverage
    # term (GJR over plain GARCH) bought rank. Located by family and name so
    # the verdict follows the data rather than asserting an ordering.
    order = [str(m) for m in ranked["model"]]
    econ = [m for m in order if families.get(m) == "econometric"]
    positions = " and ".join(f"{_escape_latex(m)} {_ordinal(order.index(m) + 1)}"
                             for m in econ) + f" of {len(order)}"
    gjr = next((m for m in econ if "gjr" in m.lower()), None)
    plain = next((m for m in econ if "gjr" not in m.lower()), None)
    if gjr and plain:
        leverage = ("which is consistent with the leverage term earning its place"
                    if order.index(gjr) < order.index(plain) else
                    "so this run does not show the leverage term earning its place")
    else:
        leverage = "so no leverage comparison is available in this run"

    return {
        "best_model": _escape_latex(best_model),
        "best_params": _fint(agg.loc[best_model, "n_params_g"]),
        "best_seconds": _fnum(agg.loc[best_model, "train_seconds"], 3),
        "best_family": families.get(best_model, "---"),
        "biggest_model": _escape_latex(biggest),
        "biggest_params": _fint(agg.loc[biggest, "n_params_g"]),
        "biggest_seconds": _fnum(agg.loc[biggest, "train_seconds"], 0),
        "biggest_below": below_txt,
        "param_ratio": _fint(agg.loc[biggest, "n_params_g"]
                             / max(agg.loc[best_model, "n_params_g"], 1)),
        "time_ratio": _fint(agg.loc[biggest, "train_seconds"]
                            / max(agg.loc[best_model, "train_seconds"], 1e-12)),
        "fidelity_leader": _escape_latex(fid_leader),
        "temporal_leader": _escape_latex(tmp_leader),
        "temporal_leader_params": _fint(agg.loc[tmp_leader, "n_params_g"]),
        "econ_positions": positions,
        "leverage_verdict": leverage,
    }


# ============================================================================
# Shuffled-control audit, measured across every (market, seed)
# ============================================================================

# A metric counts as permutation-invariant in practice when the shuffled
# control's value is negligible against what the best real generator achieves
# on the same metric. An absolute epsilon cannot do this job: the metrics span
# many orders of magnitude (mean_diff ~1e-4, resid_kurtosis_diff ~1e+0), so any
# fixed threshold either misses the invariant metrics or sweeps in ordering-
# sensitive ones. The ratio criterion is also what the claim actually means --
# "the control scores this metric as if it were the real data" -- and on this
# run it separates cleanly: every invariant metric lands below 1e-6, every
# other metric above 0.34, with nothing in between to make the cut arbitrary.
_PERM_INVARIANT_RATIO = 1e-3

# The four metrics on which the control is not perfect by construction but is
# still competitive -- the finding that the 2026 run replaced the old
# "seven of fourteen" framing with. Ordered as reported.
_CONTROL_AUDIT_METRICS = [
    ("acf_returns_mae",  r"ACF$(r)$ MAE"),
    ("acf_squared_mae",  r"ACF$(r^2)$ MAE"),
    ("acf_absolute_mae", r"ACF$(|r|)$ MAE"),
    ("hurst_diff",       r"Hurst diff"),
]


def _pooled_metrics(runs: list) -> pd.DataFrame:
    """Every per-(market, seed) metrics CSV stacked into one frame, so a
    'mean across all 15 market-seeds' is computed over the runs actually on
    disk rather than over whichever one happens to be first."""
    frames = []
    for run in runs:
        d = run["metrics"].copy()
        d["market"] = run["market"]
        d["seed"] = run["seed"]
        frames.append(d)
    if not frames:
        raise ReportDataError("No per-market metrics CSVs to pool")
    return pd.concat(frames, ignore_index=True)


def _control_name(pooled: pd.DataFrame) -> str:
    names = [m for m in pooled["model"].unique() if _is_control(m)]
    if not names:
        raise ReportDataError(
            "No shuffled-control row found in the per-market metrics CSVs; "
            "the control audit is the report's central result and cannot be "
            "rendered without it"
        )
    return str(names[0])


def _build_perm_invariance(pooled: pd.DataFrame) -> dict:
    """Count and name the metrics the control scores exactly at the optimum.

    Counted, not asserted: the headline 'nine of nineteen' is the number of
    computed metrics whose control value is zero to within 1e-12, recomputed
    from this run rather than carried forward from the run that first found
    it.
    """
    ctrl = _control_name(pooled)
    metric_cols = [c for c in pooled.columns
                   if c not in ("model", "market", "seed")
                   and not c.endswith("_rank")
                   and pd.api.types.is_numeric_dtype(pooled[c])]
    cmeans = pooled[pooled["model"] == ctrl][metric_cols].mean()
    bmeans = pooled[~pooled["model"].apply(_is_control)].groupby("model")[metric_cols].mean().min()
    invariant, sensitive = [], []
    for c in metric_cols:
        best = abs(bmeans[c])
        ratio = abs(cmeans[c]) / best if best else float("inf")
        (invariant if ratio < _PERM_INVARIANT_RATIO else sensitive).append((c, ratio))
    gap_hi = max((r for _, r in invariant), default=float("nan"))
    gap_lo = min((r for _, r in sensitive), default=float("nan"))
    return {
        "control_name": _escape_latex(ctrl),
        "n_metrics": str(len(metric_cols)),
        "n_invariant": str(len(invariant)),
        "n_sensitive": str(len(sensitive)),
        "invariant_list": ", ".join(_escape_latex(c) for c, _ in invariant),
        # The separation itself, so the report can say the cut is not a
        # judgement call: the largest invariant ratio and the smallest
        # ordering-sensitive one, several orders of magnitude apart.
        "gap_hi": f"{gap_hi:.1e}",
        "gap_lo": _fnum(gap_lo, 2),
    }


def _build_control_audit_rows(pooled: pd.DataFrame, models: list) -> str:
    """One row per ordering-sensitive metric on which the control is
    competitive: the control's mean, then every model's mean, with the
    models that beat the control marked.

    This is the table that replaces the old 'the control scores seven of
    fourteen metrics perfectly' framing. The perfect-score metrics are
    permutation-invariant by construction and say nothing new; these four
    are ordering-sensitive by construction, and the control still beats
    real generators on them.
    """
    ctrl = _control_name(pooled)
    means = pooled.groupby("model")[[m for m, _ in _CONTROL_AUDIT_METRICS]].mean()
    if ctrl not in means.index:
        raise ReportDataError(f"Control row {ctrl!r} missing from pooled metric means")
    lines = []
    for metric, label in _CONTROL_AUDIT_METRICS:
        if metric not in means.columns:
            raise ReportDataError(
                f"Metric {metric!r} not present in the per-market metrics CSVs; "
                "the shuffled-control audit table cannot be built"
            )
        c = means.loc[ctrl, metric]
        cells = [label, f"\\textbf{{{_fnum(c, 4)}}}"]
        beaten = []
        for model in models:
            if model not in means.index:
                cells.append("---")
                continue
            v = means.loc[model, metric]
            # Bold the models that clear the control: the claim being made is
            # "which real generators beat a shuffled deck", so that comparison
            # has to be readable off the row itself.
            cells.append(f"\\textbf{{{_fnum(v, 4)}}}" if v < c else _fnum(v, 4))
            if v >= c:
                beaten.append(model)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines)


def _build_control_audit_prose(pooled: pd.DataFrame, models: list,
                               families: dict) -> dict:
    """The sentences around the audit table: which models the control beats
    on each metric, and by what margin. Counted per metric, because the
    answer differs per metric -- that difference is the point."""
    ctrl = _control_name(pooled)
    means = pooled.groupby("model")[[m for m, _ in _CONTROL_AUDIT_METRICS]].mean()
    out = {}
    for metric, _ in _CONTROL_AUDIT_METRICS:
        c = means.loc[ctrl, metric]
        beaten = [m for m in models if m in means.index and means.loc[m, metric] >= c]
        clears = [m for m in models if m in means.index and means.loc[m, metric] < c]
        margins = [c - means.loc[m, metric] for m in clears]
        out[metric] = {
            "control": _fnum(c, 4),
            "beaten": ", ".join(_escape_latex(m) for m in beaten) or "none",
            "n_beaten": str(len(beaten)),
            "clears": ", ".join(_escape_latex(m) for m in clears) or "none",
            "n_clears": str(len(clears)),
            "n_gans_beaten": str(sum(1 for m in beaten
                                     if families.get(m) != "econometric")),
            "margin_min": _fnum(min(margins), 4) if margins else "---",
            "margin_max": _fnum(max(margins), 4) if margins else "---",
        }
    return out


# ============================================================================
# Walk-forward: AUC against the empirical null, and the fold-length effect
# ============================================================================

# Measured over 15 real-vs-real half-splits; see Methods. Rank is on
# |AUC - 0.5|, never on raw AUC.
AUC_NULL_MEAN = 0.506
AUC_NULL_SD = 0.084


def _pooled_walk_forward(runs: list) -> pd.DataFrame:
    frames = []
    for run in runs:
        for model, df in run["walk_forward"].items():
            d = df.copy()
            d["model"] = model
            d["market"] = run["market"]
            frames.append(d)
    if not frames:
        raise ReportDataError("No walk-forward CSVs to pool")
    return pd.concat(frames, ignore_index=True)


def _build_wf_auc_rows(wf: pd.DataFrame) -> str:
    """Per model: mean +/- sd discriminative AUC across every fold of every
    (market, seed), and its distance from the empirical null in null sd
    units -- the only reading of an AUC that is interpretable, since the
    null is 0.506 +/- 0.084 rather than exactly 0.5."""
    if "discriminative_auc" not in wf.columns:
        raise ReportDataError(
            "walk-forward CSVs have no discriminative_auc column")
    lines = []
    for model, grp in wf.groupby("model"):
        mean = grp["discriminative_auc"].mean()
        sd = grp["discriminative_auc"].std()
        z = abs(mean - AUC_NULL_MEAN) / AUC_NULL_SD
        lines.append(
            f"{_escape_latex(str(model))} & {_fnum(mean, 3)} $\\pm$ {_fnum(sd, 3)} & "
            f"{_fnum(abs(mean - 0.5), 3)} & {_fnum(z, 2)} & {_fint(len(grp))} \\\\"
        )
    return "\n".join(lines)


def _build_wf_auc_prose(wf: pd.DataFrame) -> dict:
    """How many models' fold-averaged AUC lies more than 1.96 null sd from the
    empirical null.

    Counted, because "every model is distinguishable" is exactly the kind of
    sentence that reads true and is not: in this run some generators sit
    inside the band.
    """
    z = {}
    above = 0
    for model, grp in wf.groupby("model"):
        mean = grp["discriminative_auc"].mean()
        z[str(model)] = abs(mean - AUC_NULL_MEAN) / AUC_NULL_SD
        above += int(mean > AUC_NULL_MEAN)
    beyond = [m for m, v in sorted(z.items(), key=lambda kv: -kv[1]) if v > 1.96]
    within = [m for m, v in sorted(z.items(), key=lambda kv: kv[1]) if v <= 1.96]
    return {
        "n_models": str(len(z)),
        "n_beyond": str(len(beyond)),
        "beyond": ", ".join(_escape_latex(m) for m in beyond) or "none",
        "within": ", ".join(_escape_latex(m) for m in within) or "none",
        "n_above": str(above),
    }


def _build_fold_effect_rows(wf: pd.DataFrame) -> str:
    """Per fold: the training-length range across markets and the pooled mean
    AUC. Both move together by construction of the walk-forward schedule --
    the confound is stated in the text, not hidden by omitting one column."""
    for col in ("fold", "train_len", "discriminative_auc"):
        if col not in wf.columns:
            raise ReportDataError(f"walk-forward CSVs have no {col!r} column")
    lines = []
    for fold, grp in wf.groupby("fold"):
        lines.append(
            f"{_fint(fold)} & {_fint(grp['train_len'].min())}--{_fint(grp['train_len'].max())} & "
            f"{_fnum(grp['discriminative_auc'].mean(), 3)} & "
            f"{_fnum(grp['discriminative_auc'].std(), 3)} \\\\"
        )
    return "\n".join(lines)


def _build_fold_effect_prose(wf: pd.DataFrame) -> dict:
    """Whether the fold effect is monotonic is checked, not assumed: the
    pooled AUC falls overall from fold 0 to the last fold but does not fall
    at every step, and a report that called it monotonic would be making a
    claim the folds contradict."""
    per_fold = wf.groupby("fold").agg(
        auc=("discriminative_auc", "mean"),
        lo=("train_len", "min"),
        hi=("train_len", "max"),
    ).sort_index()
    aucs = per_fold["auc"].to_numpy()
    monotonic = bool((aucs[1:] <= aucs[:-1]).all())
    rises = [int(f) for f, prev, cur in
             zip(per_fold.index[1:], aucs[:-1], aucs[1:]) if cur > prev]
    return {
        "first_fold": _fint(per_fold.index[0]),
        "last_fold": _fint(per_fold.index[-1]),
        "first_auc": _fnum(aucs[0], 3),
        "last_auc": _fnum(aucs[-1], 3),
        "first_len": f"{_fint(per_fold['lo'].iloc[0])}--{_fint(per_fold['hi'].iloc[0])}",
        "last_len": f"{_fint(per_fold['lo'].iloc[-1])}--{_fint(per_fold['hi'].iloc[-1])}",
        "monotonic": "monotonic" if monotonic else "not monotonic",
        "rise_folds": ", ".join(str(f) for f in rises) or "none",
        "n_folds_measured": _fint(len(per_fold)),
    }


def _build_wf_dispersion_rows(wf: pd.DataFrame) -> str:
    """Wasserstein and Hurst across folds, mean and median side by side.

    Median as well as mean deliberately: one model's fold-level Wasserstein
    mean in this run is three orders of magnitude above its own median,
    driven by a handful of folds (see the outlier note in Results). A mean
    alone would report that as the model's typical behaviour.
    """
    lines = []
    for model, grp in wf.groupby("model"):
        cells = []
        for col in ("wasserstein", "hurst_diff"):
            if col not in grp.columns:
                raise ReportDataError(f"walk-forward CSVs have no {col!r} column")
            cells.append(f"{_fnum(grp[col].mean(), 4)}")
            cells.append(f"{_fnum(grp[col].median(), 4)}")
        lines.append(f"{_escape_latex(str(model))} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def _build_wf_outliers(wf: pd.DataFrame, k: int = 3) -> dict:
    """The folds whose Wasserstein is furthest above the pooled median.

    Reported rather than filtered. Removing them would change a measured
    result; naming them lets a reader see that the affected mean is one
    market's fold, not the model's general behaviour.
    """
    med = wf["wasserstein"].median()
    top = wf.nlargest(k, "wasserstein")
    rows = []
    for _, r in top.iterrows():
        rows.append(
            f"{_escape_latex(str(r['model']))} & {_escape_latex(str(r['market']))} & "
            f"{_fint(r['seed'])} & {_fint(r['fold'])} & {_fnum(r['wasserstein'], 2)} & "
            f"{_fnum(r['discriminative_auc'], 3)} \\\\"
        )
    worst = top.iloc[0]
    return {
        "rows": "\n".join(rows),
        "median": _fnum(med, 5),
        "worst_model": _escape_latex(str(worst["model"])),
        "worst_market": _escape_latex(str(worst["market"])),
        "worst_fold": _fint(worst["fold"]),
        "worst_value": _fnum(worst["wasserstein"], 1),
        "worst_auc": _fnum(worst["discriminative_auc"], 3),
        "ratio": _fint(worst["wasserstein"] / med) if med else "---",
    }


# ============================================================================
# Tail index spread, and where TimeGAN actually finishes last
# ============================================================================

def _build_tail_spread_rows(pooled: pd.DataFrame, models: list) -> str:
    """tail_index_diff per model: mean, sd, and the full observed range
    across (market, seed). The range is the point -- for every model it is
    wider than the gaps between the models' means."""
    if "tail_index_diff" not in pooled.columns:
        raise ReportDataError("tail_index_diff missing from per-market metrics CSVs")
    lines = []
    g = pooled[pooled["model"].isin(models)].groupby("model")["tail_index_diff"]
    for model in models:
        if model not in g.groups:
            continue
        s = g.get_group(model)
        lines.append(
            f"{_escape_latex(model)} & {_fnum(s.mean(), 3)} & {_fnum(s.std(), 3)} & "
            f"{_fnum(s.min(), 3)}--{_fnum(s.max(), 3)} \\\\"
        )
    return "\n".join(lines)


def _build_tail_spread_prose(pooled: pd.DataFrame, models: list) -> dict:
    g = pooled[pooled["model"].isin(models)].groupby("model")["tail_index_diff"]
    means, sds = g.mean(), g.std()
    others = [m for m in models if m != means.idxmax()]
    widest = str(sds.idxmax())
    # The comparison the text makes: is the within-model spread larger than
    # the between-model spread? Computed, so the sentence cannot drift from
    # the data.
    between = float(means.max() - means.min())
    within_max = float(sds.max())
    return {
        "worst_model": _escape_latex(str(means.idxmax())),
        "worst_mean": _fnum(means.max(), 2),
        "others_range": f"{_fnum(min(means[m] for m in others), 2)}--"
                        f"{_fnum(max(means[m] for m in others), 2)}",
        "widest_model": _escape_latex(widest),
        "widest_sd": _fnum(sds.max(), 2),
        "between_spread": _fnum(between, 2),
        "within_max": _fnum(within_max, 2),
        "spread_verdict": ("exceeds" if within_max > between else "does not exceed"),
    }


def _build_worst_model_prose(pooled: pd.DataFrame, models: list,
                             fidelity_cols: list, temporal_cols: list) -> dict:
    """Where the bottom-ranked model is actually worst.

    'Worst on every metric' is a strong claim and this run does not support
    it: the bottom model is worst on every temporal metric but not on every
    fidelity one. The counts are computed so the text states the true scope.
    """
    metric_cols = [c for c in pooled.columns
                   if c not in ("model", "market", "seed")
                   and not c.endswith("_rank")
                   and pd.api.types.is_numeric_dtype(pooled[c])]
    means = pooled[pooled["model"].isin(models)].groupby("model")[metric_cols].mean()
    worst_by_metric = means.idxmax()
    counts = worst_by_metric.value_counts()
    model = str(counts.idxmax())
    present_t = [c for c in temporal_cols if c in metric_cols]
    present_f = [c for c in fidelity_cols if c in metric_cols]
    return {
        "model": _escape_latex(model),
        "n_worst_all": str(int((worst_by_metric == model).sum())),
        "n_all": str(len(metric_cols)),
        "n_worst_temporal": str(sum(worst_by_metric[c] == model for c in present_t)),
        "n_temporal": str(len(present_t)),
        "n_worst_fidelity": str(sum(worst_by_metric[c] == model for c in present_f)),
        "n_fidelity": str(len(present_f)),
        # Each exception carries the model that is worse there: the exceptions
        # are not all one model, and a single name would misattribute some.
        "not_worst": ", ".join(
            f"{_escape_latex(c)} ({_escape_latex(str(worst_by_metric[c]))})"
            for c in metric_cols if worst_by_metric[c] != model
        ) or "none",
    }


def _build_resid_kurt_prose(pooled: pd.DataFrame, families: dict) -> dict:
    """resid_kurtosis_diff for the econometric models, the GANs and the control.

    Backs the circularity paragraph with values rather than an adjective: if
    the GARCH generators were winning this metric by being GARCH models, they
    would sit near zero and beat the control. Whether they do is computed.
    """
    if "resid_kurtosis_diff" not in pooled.columns:
        raise ReportDataError("resid_kurtosis_diff missing from per-market metrics CSVs")
    means = pooled.groupby("model")["resid_kurtosis_diff"].mean()
    ctrl = _control_name(pooled)
    econ = sorted(m for m, f in families.items() if f == "econometric" and m in means.index)
    grad = [m for m, f in families.items() if f == "gradient" and m in means.index]
    econ_min = float(min(means[m] for m in econ)) if econ else float("nan")
    return {
        "econ_values": ", ".join(f"{_escape_latex(m)} {_fnum(means[m], 2)}" for m in econ),
        "gan_range": (f"{_fnum(min(means[m] for m in grad), 2)}--"
                      f"{_fnum(max(means[m] for m in grad), 2)}") if grad else "---",
        "control": _fnum(means[ctrl], 2),
        "vs_control": ("neither scores better than the shuffled control on it"
                       if econ and econ_min > means[ctrl] else
                       "at least one scores better than the shuffled control on it"),
    }


# ============================================================================
# Post-generation guard: mode-collapse firings, read from the run log
# ============================================================================

_GUARD_RE = re.compile(
    r"\[WARNING\]\s+(?P<model>\S+?):\s+generated mean=(?P<mean>-?\d+\.\d+),"
    r"\s+std=(?P<std>-?\d+\.\d+),\s+ACF\(1\)=(?P<acf>-?\d+\.\d+)")


def _build_guard_firings(reports_dir: Path) -> dict:
    """Summarise every [WARNING] post-generation guard firing in the run log.

    The run log is stdout+stderr for the run, written next to the report. It
    is the only artifact that records the guard, and unlike the metrics CSVs
    it is not tracked in git -- so a fresh clone can lack it. Rather than
    fabricate the numbers or drop the finding silently, an absent log yields
    available=False and the report prints an explicit note in place of the
    table. Nothing here is defaulted to a plausible value.
    """
    logs = sorted(reports_dir.glob("run_log_*.log"))
    if not logs:
        return {"available": False, "path": _escape_latex(str(reports_dir / "run_log_*.log"))}
    log = logs[-1]
    text = log.read_text(errors="replace")
    hits = list(_GUARD_RE.finditer(text))
    if not hits:
        return {"available": True, "n": "0", "models": "none", "path": _escape_latex(log.name),
                "acf_abs_mean": "---", "acf_min": "---", "acf_max": "---",
                "std_min": "---", "std_max": "---", "rows": "", **_diag_summary(text)}
    diag = _diag_summary(text)
    acf = np.array([float(h.group("acf")) for h in hits])
    std = np.array([float(h.group("std")) for h in hits])
    models = pd.Series([h.group("model") for h in hits]).value_counts()
    rows = "\n".join(
        f"{_escape_latex(str(m))} & {_fint(v)} \\\\" for m, v in models.items())
    return {
        "available": True,
        "path": _escape_latex(log.name),
        "n": str(len(hits)),
        "models": ", ".join(f"{_escape_latex(str(m))} ({v})" for m, v in models.items()),
        "n_models": str(len(models)),
        "acf_abs_mean": _fnum(float(np.abs(acf).mean()), 3),
        "acf_min": _fnum(float(acf.min()), 3),
        "acf_max": _fnum(float(acf.max()), 3),
        "std_min": _fnum(float(std.min()), 4),
        "std_max": _fnum(float(std.max()), 4),
        "rows": rows,
        **diag,
    }


_DIAG_LATENT_RE = re.compile(r"\[DIAG\] latent: mean=-?\d+\.\d+ std=-?\d+\.\d+ saturated=(?P<sat>\d+\.\d+)%")
_DIAG_RECON_RE = re.compile(r"\[DIAG\] reconstruction: std=(?P<got>\d+\.\d+) vs target std=(?P<want>\d+\.\d+)")


def _diag_summary(text: str) -> dict:
    """Worst latent saturation and worst reconstruction-sd mismatch over every
    [DIAG] line in the run log -- the evidence that withdrew the tanh-saturation
    explanation of TimeGAN's collapse, read rather than quoted."""
    sat = [float(m.group("sat")) for m in _DIAG_LATENT_RE.finditer(text)]
    rec = [abs(float(m.group("got")) - float(m.group("want")))
           for m in _DIAG_RECON_RE.finditer(text)]
    return {
        "diag_n_latent": _fint(len(sat)),
        "diag_sat_max": _fnum(max(sat), 1) if sat else "---",
        "diag_n_recon": _fint(len(rec)),
        "diag_recon_max_diff": _fnum(max(rec), 4) if rec else "---",
    }


# ============================================================================
# Design-decision evidence computed from the raw return series
#
# Same standing exception as Table 1: these describe the real data alone,
# independent of any generator, and no per-model artifact carries them. They
# are plain descriptive computations over the same parquet files the notebook
# reads -- they do not touch, call, or duplicate any ranking logic.
# ============================================================================

def _read_market_returns(processed_dir: Path, market: str) -> np.ndarray:
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
    return np.concatenate(parts)


def _build_kurtosis_divergence(processed_dir: Path, markets: list) -> dict:
    """Sample excess kurtosis against block length, per market.

    Evidence for the decision to make kurtosis_diff descriptive-only. With
    Hill alpha below 4 the population kurtosis does not exist, so the sample
    statistic has nothing to converge to: it grows with the window instead.
    Measured on disjoint blocks of increasing length, so each estimate is
    independent -- on nested prefixes the longer windows reuse the shorter
    ones' data and the growth would be partly an artefact of that overlap.

    Every market is shown rather than one chosen example: the effect is
    general, and picking the market where it looks strongest would be
    choosing the evidence to fit the claim.
    """
    block_ns = (250, 500, 1000, 2000)
    per_market = {}
    for market in markets:
        r = _read_market_returns(processed_dir, market)
        vals = {}
        for n in block_ns:
            if len(r) < 2 * n:
                continue
            blocks = [stats.kurtosis(r[i:i + n]) for i in range(0, len(r) - n + 1, n)]
            vals[n] = float(np.mean(blocks))
        per_market[market] = vals
    used_ns = [n for n in block_ns if all(n in v for v in per_market.values())]
    rows = []
    for market, vals in per_market.items():
        rows.append(_escape_latex(market) + " & "
                    + " & ".join(_fnum(vals[n], 2) for n in used_ns) + r" \\")
    lo, hi = used_ns[0], used_ns[-1]
    grew = sum(1 for v in per_market.values() if v[hi] > v[lo])
    return {
        "rows": "\n".join(rows),
        "header": " & ".join("$n=" + f"{n:,}".replace(",", "{,}") + "$"
                             for n in used_ns),
        "n_cols": str(len(used_ns)),
        # The tabular column spec has to match the number of block lengths
        # that survived the length check, or LaTeX silently misaligns the row.
        "colspec": "r" * len(used_ns),
        "n_min": _fint(lo), "n_max": _fint(hi),
        "n_grew": str(grew), "n_markets": str(len(per_market)),
        "ratio_min": _fnum(min(v[hi] / v[lo] for v in per_market.values()), 1),
        "ratio_max": _fnum(max(v[hi] / v[lo] for v in per_market.values()), 1),
    }


def _build_minmax_evidence(processed_dir: Path, market: str) -> dict:
    """What min-max normalisation does to a heavy-tailed return series --
    evidence for using z-score + tanh(z/3) instead.

    Under min-max to [-1, 1] a single extreme day fixes both endpoints, so the
    bulk of the series is compressed into a narrow band that is not centred at
    zero. The generator then has to spend capacity learning that offset before
    it can learn any dynamics. Measured on the market with the heaviest tail,
    which is where the effect is largest and therefore where the design choice
    is actually load-bearing.
    """
    r = _read_market_returns(processed_dir, market)
    lo, hi = float(r.min()), float(r.max())
    mm = 2 * (r - lo) / (hi - lo) - 1
    p1, p99 = np.percentile(mm, [1, 99])
    return {
        "market": _escape_latex(market),
        "median": _fnum(float(np.median(mm)), 3),
        "occupied_pct": _fnum(float((p99 - p1) / 2 * 100), 1),
        "min_ret_pct": _fnum(lo * 100, 1),
        "max_ret_pct": _fnum(hi * 100, 1),
    }


def _pick_extreme_market(characteristics: dict) -> str:
    per = {m: c for m, c in characteristics.items() if m != "_totals"}
    return max(per, key=lambda m: per[m]["excess_kurtosis"])


# ============================================================================
# Downstream utility, aggregated across every run on disk
# ============================================================================

def _load_downstream(results_dir: Path, runs: list) -> dict:
    """Downstream-utility rows, and which of the two on-disk layouts they came from.

      'pooled'     -- results_dir/pooled_downstream_utility.csv: one backtest
                      per model per seed over every market's concatenated test
                      set. Written by the pipeline from 2026-09-10.
      'per-market' -- <MARKET>/seed<N>/pooled_downstream_utility.csv: one
                      backtest per market-seed at that market's n. Written
                      before 2026-09-10 under a name promising pooling it did
                      not do; still read, so reports on those runs build, and
                      described in the text as per-market throughout.

    The pooled file wins when both exist: a rerun into an old tree overwrites
    every per-seed directory it touches but does not delete the old per-seed
    CSVs, and those must not be mixed into the new result.
    """
    pooled_path = results_dir / "pooled_downstream_utility.csv"
    cols = ["model", "seed", "qlike", "var_coverage_error", "kupiec_p", "violations", "n_test"]
    if pooled_path.exists():
        du = _read_csv(pooled_path)
        missing = [c for c in cols if c not in du.columns]
        if missing:
            raise ReportDataError(f"{pooled_path} is missing column(s) {missing}")
        du["market"] = "pooled"
        return {"mode": "pooled", "frame": du}
    frames = []
    for run in runs:
        if run["pooled_utility"] is None:
            raise ReportDataError(
                f"No downstream-utility results: neither {pooled_path} nor the "
                f"legacy per-seed file for {run['market']} seed {run['seed']} exists")
        d = run["pooled_utility"].copy()
        d["market"] = run["market"]
        d["seed"] = run["seed"]
        frames.append(d)
    du = pd.concat(frames, ignore_index=True)
    missing = [c for c in cols if c not in du.columns]
    if missing:
        raise ReportDataError(f"legacy downstream-utility CSVs missing column(s) {missing}")
    return {"mode": "per-market", "frame": du}


# A QLIKE of the right order is around -8; a positive one means a variance
# forecast near zero on some observation drove r^2/sigma^2 up by orders of
# magnitude. Used only to COUNT and NAME such cells, never to drop them.
_QLIKE_DEGENERATE = 0.0


def _signed(x, decimals=2) -> str:
    """A signed number for text mode: a bare '-' sets as a hyphen."""
    s = _fnum(abs(x), decimals)
    return f"$-${s}" if x < 0 else s


def _build_downstream_pooled_rows(downstream: dict) -> str:
    """Per model: median and mean QLIKE, and median coverage error, Kupiec p,
    violations and n, over every downstream row present (one per seed when
    pooled, one per market-seed in the legacy layout).

    Median first, deliberately: QLIKE averages log(sigma^2) + r^2/sigma^2, so
    one variance forecast near zero sends a single cell to four or five
    figures, and a mean across cells then reports that cell rather than the
    model. Both are shown and the degenerate cells are counted per model, so
    nothing is dropped and nothing is hidden.
    """
    du = downstream["frame"]
    agg = du.groupby("model").agg(
        qlike_med=("qlike", "median"), qlike_mean=("qlike", "mean"),
        cov=("var_coverage_error", "median"), kup=("kupiec_p", "median"),
        viol=("violations", "median"), n=("n_test", "median"),
    ).sort_values("qlike_med")
    bad = du[du["qlike"] > _QLIKE_DEGENERATE].groupby("model").size()
    lines = []
    for model, row in agg.iterrows():
        n_bad = int(bad.get(model, 0))
        lines.append(
            f"{_escape_latex(str(model))} & {_fnum(row['qlike_med'], 3)} & "
            f"{_fnum(row['qlike_mean'], 1)} & {_fnum(row['cov'], 4)} & "
            f"{_fnum(row['kup'], 3)} & {_fint(row['viol'])} & {_fint(row['n'])} & "
            f"{n_bad if n_bad else '---'} \\\\"
        )
    return "\n".join(lines)


def _build_downstream_text(downstream: dict, wfo: dict, garch_audit_row: str) -> dict:
    """Every sentence about downstream utility whose truth depends on the
    layout on disk or on what the rows contain: the degenerate-cell block,
    the note on n, the Methods sentence, the Extended Data audit rows and the
    Limitations bullet. Built here so the template never asserts pooling the
    artifacts do not have, or a mean-versus-median gap the data does not show.
    """
    du, pooled = downstream["frame"], downstream["mode"] == "pooled"
    unit = "model-seed" if pooled else "model-market-seed"

    # How often the shuffled control's QLIKE beats a genuine generator's,
    # within each backtest (one per seed pooled, one per market-seed per-market).
    # Counted over control-versus-generator pairs, not over all model pairs.
    _beats = _pairs = 0
    for _, _g in du.groupby(["seed"] if pooled else ["market", "seed"]):
        _c = _g[_g["model"].apply(_is_control)]["qlike"].dropna()
        _o = _g[~_g["model"].apply(_is_control)]["qlike"].dropna()
        if len(_c):
            _pairs += len(_o)
            _beats += int((float(_c.iloc[0]) < _o).sum())
    ctrl_sentence = (
        f"The shuffled control's QLIKE beats a genuine generator's in {_beats} of {_pairs} "
        f"control-versus-generator comparisons ({(100 * _beats / _pairs) if _pairs else 0:.0f}\\%)."
        if _pairs else "")
    bad = du[du["qlike"] > _QLIKE_DEGENERATE].sort_values("qlike", ascending=False)
    n_bad, n_total = len(bad), len(du)

    st = du.groupby("model")["qlike"].agg(["mean", "median"])
    gap = st["mean"] - st["median"]
    # The model named in the "mean is uninformative" sentence must be one whose
    # gap actually comes from degenerate cells, or the sentence misattributes.
    gap_pool = gap[gap.index.isin(bad["model"].unique())] if n_bad else gap
    gap_model = str(gap_pool.idxmax())
    gap_fails = int((bad["model"] == gap_model).sum())

    n_vals = sorted(int(v) for v in du["n_test"].dropna().unique())
    n_min, n_max = n_vals[0], n_vals[-1]
    n_txt = _fint(n_min) if n_min == n_max else f"{_fint(n_min)}--{_fint(n_max)}"
    meets = "meets" if n_min >= 600 else "does not meet"
    power = (r"close to the $n\approx2{,}480$ at which Kupiec power was measured at $\approx99\%$"
             if n_min >= 2000 else
             r"below the $n\approx2{,}480$ at which Kupiec power was measured at $\approx99\%$ "
             r"(it is $\approx56\%$ at $n\approx496$)")
    out = {}

    if n_bad:
        rows = "\n".join(
            f"{_escape_latex(str(r['model']))} & {_escape_latex(str(r['market']))} & "
            f"{_fint(r['seed'])} & {_fnum(r['qlike'], 0)} & {_fint(r['violations'])} & "
            f"{_fint(r['n_test'])} \\\\" for _, r in bad.iterrows())
        out["fail_block"] = (
            r"\textbf{Median and mean are both shown, and they disagree.} "
            f"{n_bad} of {n_total} {unit} cells returned a positive QLIKE:" "\n\n"
            "\\smallskip\n{\\footnotesize\n\\begin{tabular}{@{}llrrrr@{}}\n\\toprule\n"
            "Model & Market & Seed & QLIKE & Viol. & $n$ \\\\\n\\midrule\n"
            + rows + "\n\\bottomrule\n\\end{tabular}}\n\n\\smallskip\n"
            r"QLIKE averages $\log\sigma_t^2 + r_t^2/\sigma_t^2$ over observations, so a variance "
            r"forecast near zero on a single observation sends that term toward infinity, and one "
            r"such cell then dominates any mean taken across cells. \textbf{The mean column is "
            r"therefore uninformative here:} "
            f"{_escape_latex(gap_model)}'s mean of {_signed(st.loc[gap_model, 'mean'], 2)} against "
            f"a median of {_signed(st.loc[gap_model, 'median'], 2)} is an artefact of {gap_fails} "
            f"degenerate cell{'s' if gap_fails != 1 else ''}, not a summary. "
            r"The cells are reported, not removed --- removing them would change a measured "
            r"result --- and the median is the column to read.")
    else:
        out["fail_block"] = (
            r"\textbf{Median and mean are both shown.} "
            f"No {unit} cell returned a positive QLIKE in this run; the largest gap between a "
            f"model's mean and median QLIKE is {_fnum(gap.max(), 3)} ({_escape_latex(gap_model)}).")

    if pooled:
        n_mk = int(du["n_markets"].max()) if "n_markets" in du.columns else 0
        mk_names = (_escape_latex(str(du["markets"].dropna().iloc[0]).replace(" ", ", "))
                    if "markets" in du.columns else "---")
        n_sd = int(du["seed"].nunique())
        s_ = "s" if n_sd != 1 else ""
        bnd = max(n_mk - 1, 0)
        caveat = (f"Concatenating {n_mk} markets means one GARCH filter crosses {bnd} market "
                  f"boundar{'y' if bnd == 1 else 'ies'} within {n_txt} pooled observations.")
        out["n_note"] = (
            f"Each backtest runs once per seed over the pooled test set: the real test series of "
            f"all {n_mk} markets ({mk_names}) concatenated in market order, with each model's "
            f"per-market synthetic draws concatenated in the same order. $n$ is {n_txt} per "
            f"backtest, over {n_sd} seed{s_}, so each row above summarises {n_sd} pooled "
            f"backtest{s_}. That $n$ {meets} the " r"$n\gtrsim600$" " at which QLIKE ranks "
            f"stably, and is " + power + ". " + caveat + " " + ctrl_sentence)
        # Kept shorter than the per-market variant: the Methods continuation page
        # has no slack, and this sentence is the one whose length varies by layout.
        out["methods_note"] = (
            r"The pipeline therefore computes it once per seed over every market's concatenated "
            f"test set: $n$ is {n_txt} here (Results, Table~7), which {meets} the "
            r"$n\gtrsim600$ threshold, at the disclosed cost of the variance filter crossing "
            f"{bnd} market boundar{'y' if bnd == 1 else 'ies'}.")
        out["limitation"] = (
            r"\textbf{Downstream QLIKE outliers.} "
            + (f"{n_bad} of {n_total} pooled backtests returned a positive QLIKE; reported with "
               "the median beside the mean, not removed." if n_bad else
               "No pooled backtest returned a degenerate QLIKE in this run."))
        pool_row = (
            r"\auditrow{FIXED}{Amber}{ABg}%" "\n"
            r"  {Downstream utility pooled across markets}%" "\n"
            f"  {{One file, one backtest per model per seed, $n$ of {n_txt} over {n_mk} markets. "
            r"Before 2026-09-10 the pipeline wrote one file per market-seed under this name, at "
            r"that market's $n$ (486--501) --- below QLIKE's $n\gtrsim600$ stability threshold, "
            r"where the shuffled control's QLIKE beat a genuine generator's in 33 of 75 "
            r"control-versus-generator comparisons (44\%) --- a prior measurement on those files.}")
    else:
        per_market = du.groupby("market")["n_test"].max()
        n_files = int(du.groupby(["market", "seed"]).ngroups)
        out["n_note"] = (
            r"The files are named \texttt{pooled\_downstream\_utility.csv}, but each carries one "
            f"market's test set: $n$ ranges from {_fint(n_min)} to {_fint(n_max)} across {n_files} "
            f"files, combining to {_fint(per_market.sum())} across the {len(per_market)} markets. "
            f"What is reported above is therefore a summary of {n_files} per-market backtests, "
            r"\textbf{not} a single pooled backtest at the combined $n$ --- these artifacts predate "
            r"the pipeline's pooled computation. The distinction is not cosmetic: the per-market $n$ "
            r"is " + power + r", so the per-market $p$-values above are descriptive, and QLIKE is the "
            r"primary downstream signal. " + ctrl_sentence)
        out["methods_note"] = (
            r"The design intent is therefore pooled computation. The pipeline now does this --- "
            r"once per seed over every market's concatenated test set --- but \textbf{these "
            r"artifacts predate that change}: they are per-market at $n$ of "
            f"{_fint(n_min)}--{_fint(n_max)} (Results, Table~7), below the " r"$n\gtrsim600$ "
            r"threshold despite the file name, and are described as per-market throughout. Kupiec "
            r"power is $\approx56\%$ at $n\approx496$ against $\approx99\%$ at $n\approx2{,}480$, so "
            r"per-market $p$-values are descriptive.")
        out["limitation"] = (
            r"\textbf{Downstream utility per market, not pooled.} These artifacts predate the "
            r"pipeline's pooled computation (Methods), and "
            f"{n_bad} of {n_total} cells returned a degenerate QLIKE, reported with the median "
            r"beside the mean.")
        pool_row = (
            r"\auditrow{FIXED IN PIPELINE}{Amber}{ABg}%" "\n"
            r"  {\texttt{pooled\_downstream\_utility.csv} was not pooled}%" "\n"
            f"  {{Each file in these artifacts carries one market's test set ($n$ of "
            f"{_fint(n_min)}--{_fint(n_max)}, combining to {_fint(per_market.sum())} across "
            f"{len(per_market)} markets), because the evaluation function that wrote it only ever "
            r"sees one market. QLIKE's stability threshold is $n\gtrsim600$ and Kupiec power at "
            r"$n\approx496$ is $\approx56\%$ against $\approx99\%$ pooled, so the difference is "
            r"material: " + ctrl_sentence + r" \textbf{Fix:} the pipeline now records each market-seed's inputs and pools "
            r"them across markets once per seed into a single file. These artifacts predate it and "
            r"are described as per-market throughout.}")

    qlike_row = (
        (r"\auditrow{REPORTED}{Indigo}{IBg}%" "\n"
         f"  {{Downstream QLIKE degenerate in {n_bad} of {n_total} cells}}%" "\n"
         r"  {QLIKE averages $\log\sigma_t^2 + r_t^2/\sigma_t^2$, so a variance forecast near zero "
         r"on a single observation sends one term toward infinity. "
         f"{_escape_latex(gap_model)}'s mean of {_signed(st.loc[gap_model, 'mean'], 2)} against a "
         f"median of {_signed(st.loc[gap_model, 'median'], 2)} is an artefact of {gap_fails} such "
         r"cell(s), not a summary. The cells are reported with the median beside the mean and are "
         r"not removed, since removing them would change a measured result.}")
        if n_bad else
        (r"\auditrow{OK}{FGreen}{GBg}%" "\n"
         r"  {Downstream QLIKE: no degenerate cells}%" "\n"
         f"  {{No {unit} cell returned a positive QLIKE in this run; median and mean are both "
         r"reported regardless.}"))
    out["audit_rows"] = "\n\n".join([garch_audit_row, qlike_row, pool_row])
    return out


def _sci(x: float, digits: int = 1) -> str:
    """Scientific notation for LaTeX math, e.g. $1.7\\times10^{-13}$. A Python
    '1.7e-13' set inside $...$ turns the exponent's hyphen into a binary minus
    with operator spacing ('1.7e − 13')."""
    if x == 0:
        return "$0$"
    mant, exp = f"{x:.{digits}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def _window_key(w: str):
    """Sort 'main' before 'fold 0', 'fold 1', ... numerically."""
    m = re.match(r"fold (\d+)$", str(w))
    return (1, int(m.group(1))) if m else (0, 0)


def _build_persistence(runs: list, wfo: dict) -> dict:
    """The integrated-boundary finding: how many econometric fits reached
    persistence 1, where, and that the constraint was applied to exactly those.

    Read from <MARKET>/seed<N>/garch_persistence.csv (one row per econometric
    fit: the main fit and every walk-forward fold). The consistency of
    constraint_applied with the recorded unconstrained persistence is checked,
    not assumed -- a disagreement means the artifact cannot support the
    sentence "applied to exactly those", and the build stops.
    """
    frames = [r["persistence"] for r in runs if r.get("persistence") is not None]
    why = (
        r"Near-unit persistence in daily equity returns is a documented consequence of structural "
        r"breaks in the unconditional variance being absorbed as persistence (Lamoureux \& "
        r"Lastrapes 1990\tcite{42}; Mikosch \& St\u{a}ric\u{a} 2004\tcite{43}). SHANGHAI over "
        r"2006--2026 spans the 2007 bubble and crash, 2015 and COVID: three volatility regimes in one "
        r"training window. At $\alpha+\beta=1$ the conditional variance is a martingale with no "
        r"unconditional variance to revert to, so simulated paths random-walk without an anchor --- "
        r"the mechanism behind a walk-forward Wasserstein of 74.8 against a run median of 0.0025 on "
        r"SHANGHAI fold~3 in the run that preceded this procedure."
        "\n\n\\smallskip\n"
        r"\textbf{Two-stage fit.} Stage~1 fits by maximum likelihood without further constraint and "
        r"always records persistence ($\alpha+\beta$, plus $\gamma/2$ for GJR-GARCH --- \texttt{arch}'s "
        r"own definition). Stage~2 runs only when stage~1 lands within $10^{-6}$ of 1: it refits under "
        r"persistence $\le 1-\delta$, $\delta=10^{-4}$, and output is generated from the constrained "
        r"parameters; both parameter sets are stored with the fit. The refit tightens the right-hand "
        r"side of \texttt{arch}'s own stationarity constraint and re-optimises \texttt{arch}'s own "
        r"likelihood, so nothing is reimplemented, and on stationary windows it reproduces the "
        r"unconstrained fit. $\delta$ sits in an observed gap: across 110 refits, 6 were within "
        r"$1.7\times10^{-13}$ of 1 and every other fit at least $1.23\times10^{-4}$ below it, so the "
        r"bound binds only on fits that were already at the boundary."
        "\n\n\\smallskip\n"
        r"\textbf{Why at fit time.} Whether a boundary fit explodes is arbitrary. In the preceding run, "
        r"GARCH exploded on SHANGHAI fold~3 for all three seeds while GJR-GARCH, on the same window, did "
        r"not; local refits reversed it, 200 of 200 simulated paths against 0 of 200. Rejecting "
        r"explosive draws after generation would keep whichever draws happened to survive; "
        r"constraining the fit removes the knife-edge. A post-generation guard still raises if "
        r"output sd leaves $1/10$--$10\times$ the training sd. It is not guaranteed silent: at "
        r"persistence 0.9999 one simulated path in 200 per boundary window exceeded $10\times$ in "
        r"local refits (max $20.8\times$), though the pipeline's seeds stayed within $1.7\times$. "
        r"The refit counts and the knife-edge are prior measurements recorded in the repository's "
        r"standing constraints, not artifacts of this run.")

    if not frames:
        left = (r"Persistence was not recorded in these artifacts: they predate two-stage fitting, "
                r"so the number of fits that reached the integrated boundary cannot be read from them.")
        audit = (
            r"\auditrow{FIXED IN PIPELINE}{Amber}{ABg}%" "\n"
            r"  {GARCH fits at the integrated boundary generated explosive paths}%" "\n"
            f"  {{In these artifacts {wfo['worst_model']} on {wfo['worst_market']} walk-forward fold "
            f"{wfo['worst_fold']} gives Wasserstein {wfo['worst_value']} against a pooled median of "
            f"{wfo['median']}, with discriminative AUC {wfo['worst_auc']}. Cause: maximum likelihood "
            r"converged normally to persistence on the stationarity boundary, where the conditional "
            r"variance is a martingale. \textbf{Fix:} two-stage fitting --- persistence recorded "
            r"unconstrained, a constrained refit (persistence $\le1-10^{-4}$) only when it reaches 1 to "
            r"within $10^{-6}$, output generated from the constrained parameters --- plus a "
            r"post-generation guard. These artifacts predate both.}")
        return {"available": False, "left": left, "right": why, "audit_row": audit}

    df = pd.concat(frames, ignore_index=True)
    need = ["market", "seed", "window", "model", "unconstrained_persistence",
            "constraint_applied", "constrained_persistence"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ReportDataError(f"garch_persistence.csv missing column(s) {missing}")
    tol = float(df["stationarity_tol"].iloc[0]) if "stationarity_tol" in df.columns else 1e-6
    applied = df["constraint_applied"].astype(str).str.lower().isin(["true", "1"])
    hit = df["unconstrained_persistence"] >= 1.0 - tol
    if int((hit != applied).sum()):
        raise ReportDataError(
            f"garch_persistence.csv: constraint_applied disagrees with unconstrained persistence "
            f"(tolerance {tol:g}) in {int((hit != applied).sum())} row(s); the report cannot state "
            f"that the constraint was applied to exactly the boundary fits")

    keys = ["market", "window", "model"]
    n_rows, n_distinct, n_seeds = len(df), df.groupby(keys).ngroups, df["seed"].nunique()
    n_hit, hit_df = int(hit.sum()), df[hit]
    n_hit_distinct = hit_df.groupby(keys).ngroups if n_hit else 0
    s_ = "s" if n_seeds != 1 else ""
    left = (
        r"Every econometric fit records its unconstrained persistence as a result: the main fit on "
        r"each training split and one per walk-forward fold, for both GARCH variants. "
        f"Across {n_rows} fits ({n_distinct} market--window--model combinations $\\times$ {n_seeds} "
        f"seed{s_}; a fit does not depend on the seed), ")
    if n_hit:
        rows = []
        for (mk, w, mdl), g in sorted(hit_df.groupby(keys), key=lambda kv: (kv[0][0], _window_key(kv[0][1]), kv[0][2])):
            rows.append(
                f"{_escape_latex(mk)} & {_escape_latex(w)} & {_escape_latex(mdl)} & "
                f"{g['unconstrained_persistence'].iloc[0]:.10f} & "
                f"{g['constrained_persistence'].iloc[0]:.10f} & {g['seed'].nunique()} \\\\")
        mk_hit = ", ".join(_escape_latex(m) for m in sorted(hit_df["market"].unique()))
        rest = df[~hit]["unconstrained_persistence"]
        gap_hit = float((1.0 - hit_df["unconstrained_persistence"]).abs().max())
        pc = hit_df["constrained_persistence"].astype(float)
        llc = hit_df["loglikelihood_change"].astype(float) if "loglikelihood_change" in df.columns else None
        left += (
            f"\\textbf{{{n_hit} reached the integrated boundary}} --- persistence within $10^{{-6}}$ of 1 "
            f"--- in {n_hit_distinct} combination{'s' if n_hit_distinct != 1 else ''}, on {mk_hit}. "
            r"The constraint was applied to exactly those fits and to no other."
            "\n\n\\smallskip\n{\\footnotesize\n\\begin{tabular}{@{}lllrrc@{}}\n\\toprule\n"
            "Market & Window & Model & Unconstrained & Constrained & Seeds \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\n\\smallskip\n"
            f"Boundary fits sit within {_sci(gap_hit)} of 1"
            + (f"; the largest persistence among the other {len(rest)} fits is {rest.max():.8f} "
               f"($1-p$ = {_sci(1 - rest.max(), 2)})" if len(rest) else "")
            + f". Constrained persistence is {pc.min():.10f}--{pc.max():.10f}"
            + (f", at a log-likelihood change of {llc.min():+.4f} to {llc.max():+.4f}" if llc is not None else "")
            + ".")
        hit_list = "; ".join(
            f"{_escape_latex(mk)} {_escape_latex(w)} {_escape_latex(mdl)}"
            for (mk, w, mdl), _ in sorted(hit_df.groupby(keys), key=lambda kv: (kv[0][0], _window_key(kv[0][1]), kv[0][2])))
        audit_body = (
            f"{n_hit} of {n_rows} econometric fits in this run reached persistence within $10^{{-6}}$ of 1 "
            f"({hit_list}) and were refit under persistence $\\le1-10^{{-4}}$; output was generated from "
            r"the constrained parameters and both parameter sets are stored (Results, Table~8).")
    else:
        left += (r"\textbf{none reached the integrated boundary}, and the constraint was applied to no "
                 f"fit. The largest unconstrained persistence is {df['unconstrained_persistence'].max():.8f}.")
        audit_body = (f"None of the {n_rows} econometric fits in this run reached the integrated "
                      r"boundary, so no constrained refit was needed.")
    audit = (
        r"\auditrow{FIXED}{Amber}{ABg}%" "\n"
        r"  {GARCH fits at the integrated boundary: two-stage fitting}%" "\n"
        "  {" + audit_body + r" An earlier version raised and stopped the run on such fits, which would "
        r"have kept SHANGHAI from completing on any seed; it was replaced because stopping the "
        r"benchmark is not a treatment of the fit. The post-generation guard still raises outside "
        r"$1/10$--$10\times$ the training sd.}")
    return {"available": True, "left": left, "right": why, "audit_row": audit,
            "n_rows": n_rows, "n_hit": n_hit}


def _build_control_audit_header(models: list) -> dict:
    """Header and column spec for the control-audit table, from the same model
    list the rows iterate. A typed header silently mislabels every column the
    moment the sorted model order changes -- which a rename does."""
    return {
        "header": "Metric & Control & " + " & ".join(_escape_latex(m) for m in models),
        "colspec": "l" + "c" * (len(models) + 1),
    }


# ============================================================================
# Public entry point
# ============================================================================

def load_report_context(results_dir="thesis_results/production",
                         reports_dir="reports/production",
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
    stale_walk_forward = []   # files for models not in their run -- skipped, reported
    for market, seed, seed_dir in discover_market_seed_dirs(results_dir):
        metrics = _read_csv(seed_dir / f"{market}_metrics.csv")
        # Pre-2026-09-10 layout only; newer runs write one pooled file at the root.
        _legacy_du = seed_dir / "pooled_downstream_utility.csv"
        pooled_utility = _read_csv(_legacy_du) if _legacy_du.exists() else None
        _persist_path = seed_dir / "garch_persistence.csv"
        persistence = _read_csv(_persist_path) if _persist_path.exists() else None
        walk_forward = _discover_walk_forward(results_dir, market, seed,
                                              set(metrics["model"]), stale_walk_forward)
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
            "persistence": persistence,
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
        "stale_walk_forward": stale_walk_forward,
    }

    data_markets = _discover_data_markets(processed_dir)
    characteristics = compute_real_market_characteristics(processed_dir, data_markets)

    # Table families that the ranking uses. Declared here only so the report
    # can say which metric belongs to which family; the ranking itself is the
    # notebook's, and this module never recomputes a rank.
    fidelity_cols = ["mean_diff", "std_diff", "wasserstein", "energy_distance",
                     "quantile_mse", "tail_index_diff", "extreme_events_diff"]
    temporal_cols = ["acf_returns_mae", "acf_absolute_mae", "acf_squared_mae",
                     "hurst_diff", "resid_kurtosis_diff",
                     "discriminative_auc_dist", "discriminative_auc_absz"]

    families = _model_families(per_seed_market)
    pooled = _pooled_metrics(runs)
    wf = _pooled_walk_forward(runs)

    # The min-max evidence is measured on the heaviest-tailed market, chosen
    # by the data rather than named in the text: that is where a scale set by
    # one extreme day bites hardest, so it is where the normalisation choice
    # is load-bearing. The kurtosis-divergence evidence uses every market,
    # since the claim there is that the effect is general.
    heaviest_tail_market = _pick_extreme_market(characteristics)

    # The primary run still backs the single-run figures. Every table below is
    # pooled across all runs on disk instead -- with 5 markets x 3 seeds there
    # is no longer any reason to report one market's numbers as the result.
    primary = runs[0]
    wf_outliers = _build_wf_outliers(wf)
    downstream = _load_downstream(results_dir, runs)
    persistence = _build_persistence(runs, wf_outliers)

    ctx["formatted"] = {
        "provenance": _build_provenance(metadata),
        "table1_rows": _build_table1_rows(characteristics),
        "table1_prose": _build_table1_prose(characteristics),
        "table3_rows": _build_table3_rows(overall),
        "best_composite": _best_composite(overall),
        "budget_parity_rows": _build_budget_parity_rows(per_seed_market),

        # Headline result and the family split
        "families": families,
        "headline_rows": _build_headline_rows(overall, per_seed_market, families),
        "win_prose": _build_win_prose(per_seed_market, families),

        # Compute versus performance
        "gan_only_prose": _build_gan_only_prose(per_seed_market, families),
        "compute_rows": _build_compute_rows(overall, per_seed_market, families),
        "compute_prose": _build_compute_prose(per_seed_market, overall, families),

        # Shuffled-control audit
        "perm_invariance": _build_perm_invariance(pooled),
        "control_audit_rows": _build_control_audit_rows(pooled, models),
        "control_audit_header": _build_control_audit_header(models),
        "control_audit_prose": _build_control_audit_prose(pooled, models, families),

        # Walk-forward
        "wf_auc_rows": _build_wf_auc_rows(wf),
        "wf_auc_prose": _build_wf_auc_prose(wf),
        "resid_kurt_prose": _build_resid_kurt_prose(pooled, families),
        "fold_effect_rows": _build_fold_effect_rows(wf),
        "fold_effect_prose": _build_fold_effect_prose(wf),
        "wf_dispersion_rows": _build_wf_dispersion_rows(wf),
        "wf_outliers": wf_outliers,

        # Tail index and the bottom-ranked model
        "tail_spread_rows": _build_tail_spread_rows(pooled, models),
        "tail_spread_prose": _build_tail_spread_prose(pooled, models),
        "worst_model_prose": _build_worst_model_prose(
            pooled, models, fidelity_cols, temporal_cols),

        # Post-generation guard
        "guard": _build_guard_firings(reports_dir),

        # Design-decision evidence from the raw series
        "kurtosis_divergence": _build_kurtosis_divergence(
            processed_dir, data_markets),
        "minmax_evidence": _build_minmax_evidence(
            processed_dir, heaviest_tail_market),

        # Downstream utility: the pooled file if the run wrote one, else the
        # legacy per-market files, described as such (see _load_downstream)
        "downstream_mode": downstream["mode"],
        "downstream_rows": _build_downstream_pooled_rows(downstream),
        "persistence": persistence,
        "downstream_text": _build_downstream_text(downstream, wf_outliers, persistence["audit_row"]),

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
