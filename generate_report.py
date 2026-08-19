"""
Comprehensive PDF report — Synthetic Financial Time Series Generation.
Run from the project root:  python generate_report.py
Feynman technique: every concept is explained from first principles.
Outputs to reports/report_YYYY-MM-DD.pdf  (versioned, date-stamped).
"""

import os
import textwrap
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

_DATE      = datetime.date.today().isoformat()          # e.g. 2026-07-20
OUTPUT_PDF = f"reports/report_{_DATE}.pdf"

# ── Palette ──────────────────────────────────────────────────────────────────
C = {
    'navy':   '#0D2B6B', 'blue':   '#1565C0', 'sky':    '#1976D2',
    'teal':   '#00695C', 'green':  '#2E7D32', 'amber':  '#E65100',
    'red':    '#B71C1C', 'purple': '#6A1B9A', 'grey':   '#424242',
    'light':  '#F5F5F5', 'white':  '#FFFFFF',
    'TG': '#E65100', 'QG': '#1565C0', 'FG': '#2E7D32',
}
PAGE = (11.69, 8.27)   # A4 landscape


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def header_bar(fig, title, subtitle=''):
    bar = fig.add_axes([0, 0.91, 1, 0.09])
    bar.set_facecolor(C['navy'])
    bar.axis('off')
    bar.text(0.015, 0.62, title, color='white', fontsize=14, fontweight='bold',
             va='center', transform=bar.transAxes)
    if subtitle:
        bar.text(0.015, 0.18, subtitle, color='#90CAF9', fontsize=8.5,
                 va='center', transform=bar.transAxes)
    bar.text(0.985, 0.5, 'Synthetic Financial Time Series  ·  UPC Research',
             color='#7986CB', fontsize=7.5, va='center', ha='right',
             transform=bar.transAxes)


def footer(fig, page_num, total=13):
    ft = fig.add_axes([0, 0, 1, 0.03])
    ft.set_facecolor('#E8EAF6')
    ft.axis('off')
    ft.text(0.5, 0.5, f'Page {page_num} of {total}', ha='center', va='center',
            color=C['grey'], fontsize=7.5, transform=ft.transAxes)


def tblock(ax, x, y, lines, fs=8.5, color=C['grey'], dy=0.032, clip=True):
    """Draw a list of strings top-down; returns the y after the last line."""
    for ln in lines:
        ax.text(x, y, ln, transform=ax.transAxes, fontsize=fs,
                color=color, va='top', clip_on=clip)
        y -= dy
    return y


def titled_col(ax, y, title, color, lines, fs=8.3, dy=0.031, gap=0.012):
    """
    Draw a bold coloured title then body lines.
    Returns y position after the block.
    """
    ax.text(0.01, y, title, transform=ax.transAxes,
            fontsize=10, fontweight='bold', color=color, va='top', clip_on=True)
    y -= 0.035
    for ln in lines:
        ax.text(0.025, y, ln, transform=ax.transAxes, fontsize=fs,
                color=C['grey'], va='top', clip_on=True)
        y -= dy
    return y - gap


def section_box(ax, x, y, w, h, title, lines, tc=None, fs=8.2):
    """
    Titled coloured-border box that clips content to stay inside.
    tc = title colour (defaults to navy).
    Title strip height: 0.060.  Content starts 0.018 below the strip bottom.
    dy is computed dynamically so all lines share the available height.
    """
    if tc is None:
        tc = C['navy']
    from matplotlib.patches import FancyBboxPatch

    # outer border
    ax.add_patch(FancyBboxPatch((x, y - h), w, h,
                 boxstyle='round,pad=0.008', linewidth=1.3,
                 edgecolor=tc, facecolor='#FAFAFA',
                 transform=ax.transAxes, clip_on=True))
    # title strip
    strip_h = 0.060
    ax.add_patch(FancyBboxPatch((x, y - strip_h), w, strip_h,
                 boxstyle='round,pad=0.008', linewidth=0,
                 edgecolor='none', facecolor=tc,
                 transform=ax.transAxes, clip_on=True))
    # title text (vertically centred in strip)
    ax.text(x + 0.010, y - strip_h / 2, title,
            transform=ax.transAxes, fontsize=8.8, fontweight='bold',
            color='white', va='center', clip_on=True)

    # content lines — compute dy dynamically
    top_margin = 0.018
    bot_margin = 0.010
    content_h  = h - strip_h - top_margin - bot_margin
    n_lines    = max(len(lines), 1)
    dy = min(content_h / (n_lines + 0.5), 0.038)   # never overly spaced

    cy = y - strip_h - top_margin
    for ln in lines:
        if cy < (y - h + bot_margin):
            break
        ax.text(x + 0.012, cy, ln, transform=ax.transAxes, fontsize=fs,
                color=C['grey'], va='top', clip_on=True)
        cy -= dy


# ─────────────────────────────────────────────────────────────────────────────
# Page 1 — Title
# ─────────────────────────────────────────────────────────────────────────────

def page_title(pdf):
    fig = plt.figure(figsize=PAGE)
    fig.patch.set_facecolor(C['navy'])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(C['navy']); ax.axis('off')

    ax.text(0.5, 0.88, 'Synthetic Financial Time Series Generation',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=26, fontweight='bold', color='white')
    ax.text(0.5, 0.78, 'GAN Benchmark:  TimeGAN  ·  QuantGAN  ·  FinGAN',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=16, color='#90CAF9')
    ax.text(0.5, 0.70, 'BRICS Emerging Market Indices',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=13, color='#BBDEFB')
    ax.axhline(y=0.62, xmin=0.08, xmax=0.92, color='#5C6BC0', linewidth=1.5)

    meta = [
        ('Collaboration',  'Universitat Politècnica de Catalunya (UPC)'),
        ('Document type',  'Project Status & In-Depth Technical Review'),
        ('Purpose',        'Paper publication preparation'),
        ('Models covered', 'TimeGAN (RNN) · QuantGAN (TCN-WGAN-GP) · FinGAN (CNN-WGAN-GP)'),
        ('Markets',        'BOVESPA · FTSE JSE · MSCI · NIFTY50 · SHANGHAI COMPOSITE'),
    ]
    y = 0.56
    for label, value in meta:
        ax.text(0.20, y, f'{label}:', ha='right', va='center',
                transform=ax.transAxes, fontsize=10.5,
                color='#7986CB', fontweight='bold')
        ax.text(0.22, y, value, ha='left', va='center',
                transform=ax.transAxes, fontsize=10.5, color='#E8EAF6')
        y -= 0.072

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 2 — Project Overview  (fixed spacing)
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Project Status Overview',
               'Research context · data pipeline · model status · open tasks')
    footer(fig, 2)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis('off')

    # Row 1 — two wide boxes (y=0.97, h=0.30)
    section_box(ax, 0.00, 0.97, 0.475, 0.30, 'Research Goal', [
        '• Benchmark three GAN architectures for synthetic financial log-return generation',
        '• Evaluate on 5 BRICS emerging-market indices (multi-market generalisation)',
        '• Systematically validate hyperparameters via grid search for publication',
        '• Compare stylized-fact reproduction: fat tails, volatility clustering, long memory',
        '• Prepare peer-reviewed paper in collaboration with UPC',
    ], tc=C['navy'])

    section_box(ax, 0.525, 0.97, 0.475, 0.30, 'Data Pipeline', [
        '• Source: 5 CSV files — BOVESPA, FTSE JSE, MSCI, NIFTY50, SHANGHAI',
        '• Feature: daily log return  r_t = ln(P_t / P_{t-1})',
        '• No outlier removal applied to log returns (deliberate — see note below)',
        '• Temporal 80 / 10 / 10 split (no shuffle) — saved as Parquet files',
        '• GAN training: windows of 128 steps pooled across all 5 markets',
        '  (temporal order preserved within each window; windows shuffled across markets)',
    ], tc=C['sky'])

    # Row 2 — three narrow boxes (y=0.63, h=0.27)
    section_box(ax, 0.00, 0.63, 0.305, 0.27, 'TimeGAN', [
        '• Yoon et al. (2019), NeurIPS',
        '• 4-phase training',
        '• GRU encoder / decoder',
        '• BCE loss + moment matching',
        '• MinMax [0,1] normalisation',
        '• Current: epochs = 5 (too low)',
    ], tc=C['TG'])

    section_box(ax, 0.347, 0.63, 0.305, 0.27, 'QuantGAN', [
        '• Wiese et al. (2020), Quant. Finance',
        '• WGAN-GP + TCN (causal dilated)',
        '• Generator: FC → upsample → TCN',
        '• Discriminator: TCN → pool → MLP',
        '• Adam betas=(0, 0.9)',
        '• n_critic=3,  lambda_gp=10',
    ], tc=C['QG'])

    section_box(ax, 0.695, 0.63, 0.305, 0.27, 'FinGAN', [
        '• Custom CNN-WGAN-GP',
        '• Generator: FC → 3× ConvTranspose1d',
        '• Critic: Conv1d + LayerNorm',
        '• seq_len must be divisible by 8',
        '• Adam betas=(0, 0.9)',
        '• n_critic=3,  lambda_gp=10',
    ], tc=C['FG'])

    # Row 3 — two wide boxes (y=0.32, h=0.28)
    section_box(ax, 0.00, 0.32, 0.475, 0.28, 'Pipeline Status', [
        '• [DONE]  Reproducibility seeds added to all GAN classes',
        '• [DONE]  Deprecated fillna() API fixed across all classes',
        '• [DONE]  Grid search (grid_search_gan) implemented and wired',
        '• [DONE]  FinancialMetrics rewritten with time-series-aware metrics',
        '• [TODO]  Re-run pipeline to regenerate results with updated metrics',
        '• [TODO]  Increase TimeGAN epochs (5 is far below the original paper)',
    ], tc=C['green'])

    section_box(ax, 0.525, 0.32, 0.475, 0.28, 'Key References', [
        '• Yoon et al. (2019). Time-series GAN. NeurIPS 32.',
        '• Wiese et al. (2020). Quant GANs. Quantitative Finance.',
        '• Gulrajani et al. (2017). Improved WGAN. NeurIPS.',
        '• Ang et al. (2023). TSGBench. PVLDB 17(3).',
        '• Meldrum et al. (2025). Synthetic Data for Finance. arXiv:2510.26076.',
        '• Takahashi & Mizuno (2025). Diffusion for Finance. Quant. Finance 25(10).',
    ], tc=C['purple'])

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 3 — The Problem: Financial Time Series from Scratch
# ─────────────────────────────────────────────────────────────────────────────

def page_problem(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'The Problem — From First Principles: What Are Financial Time Series?',
               'Understanding the data before understanding the models')
    footer(fig, 3)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('What is a Financial Time Series?', C['navy'], [
            'A financial time series is simply a sequence of numbers measured over time.',
            'For a stock or index: P_1, P_2, P_3, ..., P_T where P_t is the closing price',
            'on day t.',
            '',
            'Example: the BOVESPA index on five consecutive days might be:',
            '  [124,500  →  126,200  →  125,800  →  128,100  →  127,600]',
            '',
            'Problem with prices directly: prices grow over time (have a trend), so',
            'comparing "day 1 volatility" to "day 1000 volatility" is misleading.',
            'Statistical models struggle with non-stationary data.',
        ]),
        ('Why Log Returns? (Building the Feature)', C['blue'], [
            'Instead of prices, we compute the log return for each day:',
            '',
            '  r_t = ln(P_t / P_{t-1})  =  ln(P_t) - ln(P_{t-1})',
            '',
            'Why the logarithm?',
            '',
            '1. Symmetry:  a 10% gain then 10% loss ≠ zero in simple returns.',
            '   Log returns are additive:  r_{1→3} = r_{1→2} + r_{2→3}  (exactly).',
            '',
            '2. Approximate percentage change:  for small moves,',
            '   ln(P_t/P_{t-1}) ≈ (P_t - P_{t-1}) / P_{t-1}  (simple return)',
            '',
            '3. Stationarity:  log returns fluctuate around zero with no long-run trend,',
            '   making them much easier to model statistically.',
            '',
            'From our example:  r_2 = ln(126,200 / 124,500) ≈ +0.0135  (+1.35%)',
        ]),
        ('The First Discovery: Returns Are NOT Normal', C['red'], [
            'The simplest assumption: r_t ~ N(μ, σ²) independently.',
            'This is the foundation of many classic financial models (Black-Scholes, etc.).',
            '',
            'The problem: this is empirically WRONG in the tails.',
            '',
            'If BOVESPA daily returns were Gaussian (μ≈0, σ≈1.2%):',
            '  A day with |r| > 3σ ≈ 3.6%  should occur ~0.27% of days → ~1 day/year.',
            '  Reality: such days occur 3–5× more often (heavy tails / kurtosis >> 3).',
            '',
            'A "5σ event" (a very large crash):',
            '  Under Gaussian: once every ~14,000 years.',
            '  Reality: several times per decade.',
            '',
            'This is why the 2008 financial crisis was called "25-sigma" by some banks',
            '— it was considered probabilistically impossible under Gaussian models.',
        ]),
    ]

    right = [
        ('What Are Stylized Facts?', C['teal'], [
            'Cont (2001) coined the term "stylized facts": statistical properties that',
            'appear consistently across virtually all financial markets and time periods.',
            '',
            'They are the fingerprint of financial returns. Any generative model that',
            'claims to produce realistic synthetic returns MUST reproduce them.',
            '',
            'The five most important:',
            '',
            '  (1) Heavy tails:  extreme events far more frequent than Gaussian predicts.',
            '',
            '  (2) Volatility clustering:  "turbulent periods beget more turbulence".',
            '       If |r_t| is large today, |r_{t+1}|, |r_{t+2}| tend to also be large.',
            '       Visually: a time series of |returns| shows "bursts" of high activity.',
            '',
            '  (3) Long memory in |returns|:  the correlation between |r_t| and |r_{t+k}|',
            '       decays very slowly — still positive even at lag k = 100 days.',
            '',
            '  (4) Near-zero autocorrelation of r_t itself:  you cannot predict tomorrow\'s',
            '       direction from today\'s return (weak-form market efficiency).',
            '',
            '  (5) Negative skewness for equity indices:  crashes are more extreme than',
            '       rallies (asymmetric distribution).',
        ]),
        ('Why Generate Synthetic Data?', C['purple'], [
            'If we already have real data, why generate synthetic data?',
            '',
            '  1. Data augmentation:  deep learning models for finance need large',
            '     datasets; real data is scarce (markets close each day).',
            '',
            '  2. Stress testing:  banks need to test risk models under extreme',
            '     scenarios that may not exist in historical data.',
            '',
            '  3. Privacy and sharing:  real financial data may be proprietary.',
            '     Synthetic data with the same statistical properties can be shared',
            '     freely for research.',
            '',
            '  4. Scenario generation:  generate many plausible futures for',
            '     portfolio optimisation and backtesting strategies.',
            '',
            'The challenge: synthetic returns must reproduce stylized facts.',
            'A Gaussian generator fails immediately (no heavy tails, no clustering).',
            'This is what motivates the use of deep generative models (GANs).',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 4 — What is a GAN? (From Scratch)
# ─────────────────────────────────────────────────────────────────────────────

def page_gan_intro(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'The Solution Candidate: What is a Generative Adversarial Network?',
               'From the intuition (counterfeiter and detective) to the mathematics')
    footer(fig, 4)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('The Core Idea — A Two-Player Game', C['navy'], [
            'Goodfellow et al. (2014) proposed a clever framework:',
            'train TWO neural networks that compete against each other.',
            '',
            'GENERATOR (G):  takes random noise z as input → outputs a fake sample x̃.',
            '  Goal: produce fake samples so realistic that the Discriminator is fooled.',
            '',
            'DISCRIMINATOR (D):  takes a sample (real or fake) → outputs a number in [0,1].',
            '  Goal: correctly classify real samples (output → 1) from fake ones (output → 0).',
            '',
            'They train simultaneously:',
            '  • D gets better at spotting fakes → G must produce better fakes.',
            '  • G gets better at fooling D → D must become more discerning.',
            '',
            'This is the counterfeiter-and-detective analogy:',
            '  Counterfeiter (G) starts making crude fake banknotes.',
            '  Detective (D) easily spots them.',
            '  The counterfeiter studies the detective\'s feedback and improves.',
            '  Eventually, the counterfeiter\'s notes are indistinguishable from real.',
            '',
            'For our problem: G generates fake log returns; D judges whether they',
            'look like real market returns.',
        ]),
        ('The Mathematical Formulation', C['blue'], [
            'Let p_data(x) = true distribution of real financial returns.',
            'Let p_g(x)    = distribution of samples generated by G.',
            '',
            'The GAN training is a minimax game (Goodfellow et al., 2014):',
            '',
            '  min_G  max_D  V(D,G)',
            '    = E_{x~p_data} [log D(x)]  +  E_{z~p_z} [log(1 - D(G(z)))]',
            '',
            'Reading this equation piece by piece:',
            '  • E_{x~p_data}[log D(x)]:  D should output values close to 1',
            '    (log(1)=0, maximum) on real data → D is rewarded for being right.',
            '  • E_{z~p_z}[log(1-D(G(z)))]:  D should output values close to 0',
            '    (log(1-0)=0) on fake data → G is penalised when D spots its fakes.',
            '',
            'G wants to MINIMISE V → maximise D(G(z)) → fool D.',
            'D wants to MAXIMISE V → correctly classify both real and fake.',
            '',
            'At the theoretical optimum (Nash equilibrium):',
            '  D*(x) = p_data(x) / (p_data(x) + p_g(x)) = 0.5   for all x',
            '  This means p_g = p_data — generator has perfectly learned the real distribution.',
        ]),
    ]

    right = [
        ('The Critical Problem: Training Instability', C['red'], [
            'In practice, GAN training is notoriously difficult. Two main failure modes:',
            '',
            'PROBLEM 1 — Mode Collapse:',
            '  G learns to output only a few samples that reliably fool D,',
            '  ignoring the diversity of the real distribution.',
            '  Analogy: the counterfeiter only produces 10-euro notes, not 20, 50, 100.',
            '',
            'PROBLEM 2 — Vanishing gradients:',
            '  The GAN loss V uses Jensen-Shannon (JS) divergence internally.',
            '  When p_data and p_g have disjoint support (early in training),',
            '  JS divergence = log(2) (a constant) — no gradient information for G.',
            '',
            '  Analogy: if the counterfeiter\'s fakes are SO BAD that the detective',
            '  immediately says "fake" without any useful feedback on HOW to improve,',
            '  the counterfeiter makes no progress.',
            '',
            'What does "disjoint support" mean?',
            '  If real returns cluster around [-0.05, +0.05] and G produces outputs',
            '  in [-10, +10], there is no overlap → the JS distance gives no useful',
            '  gradient telling G which direction to move.',
        ]),
        ('What Makes GANs Suitable for Financial Returns?', C['teal'], [
            'Unlike parametric models (GARCH, EVT), a GAN:',
            '',
            '  • Makes NO distributional assumption about the shape of returns.',
            '    It learns the distribution implicitly from data.',
            '',
            '  • Can model arbitrarily complex joint distributions p(r_1, r_2, ..., r_T),',
            '    capturing not just the marginal distribution but also temporal dependencies',
            '    (volatility clustering, long memory).',
            '',
            '  • Scales to multivariate settings (multiple assets) without specifying',
            '    a copula or correlation model in advance.',
            '',
            'Three architecture choices used in this project:',
            '',
            '  TimeGAN:    uses GRU (Gated Recurrent Unit) networks — a type of',
            '              recurrent network designed for sequential data.',
            '',
            '  QuantGAN:   uses TCN (Temporal Convolutional Network) with causal',
            '              dilated convolutions — a faster parallelisable alternative.',
            '',
            '  FinGAN:     uses CNN deconvolution — upsamples from a noise vector',
            '              to a full time series via strided transposed convolutions.',
        ]),
        ('What Does the Discriminator Actually "See"?', C['purple'], [
            'D is trained on SEQUENCES of returns, not individual points.',
            'A window of length T = 128: [r_1, r_2, ..., r_128].',
            '',
            'D learns to recognise patterns such as:',
            '  "Real returns have bursts of high volatility followed by calm periods."',
            '  "Real returns rarely stay at the same volatility level for 50 steps."',
            '',
            'G must learn to produce sequences that exhibit these same patterns.',
            'This is what allows GANs to potentially reproduce volatility clustering',
            'and other temporal stylized facts — unlike models that generate each',
            'timestep independently.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 5 — WGAN-GP: Fixing the Training Problem
# ─────────────────────────────────────────────────────────────────────────────

def page_wgan(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Better Training: The Wasserstein GAN and Gradient Penalty',
               'Arjovsky et al. (2017) ICML · Gulrajani et al. (2017) NeurIPS')
    footer(fig, 5)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('A Better Way to Measure "Distance" Between Distributions', C['navy'], [
            'The core insight of WGAN: the original GAN\'s failure comes from using',
            'the WRONG measure of how different two distributions are.',
            '',
            'The JS divergence (used implicitly in the original GAN) has a fatal flaw:',
            '  If p_data and p_g have disjoint support,  JS(p_data, p_g) = log(2)',
            '  This is a CONSTANT — its derivative w.r.t. G\'s parameters is ZERO.',
            '  G receives no gradient signal and cannot learn.',
            '',
            'Enter the Wasserstein-1 distance (also called "Earth Mover\'s distance"):',
            '',
            '  W(p, q) = inf_{γ ∈ Π(p,q)}  E_{(x,y)~γ} [ |x - y| ]',
            '',
            'Think of it with a physical analogy:',
            '  Imagine p_data as a pile of sand shaped like the real distribution',
            '  and p_g as a pile of sand shaped like the fake distribution.',
            '  W is the minimum total "work" needed to reshape one pile into the other',
            '  — where work = (mass moved) × (distance moved).',
            '',
            'Why is this better?',
            '  Even when distributions don\'t overlap at all,',
            '  W tells you HOW FAR APART they are and in which direction.',
            '  G always receives a meaningful gradient → no vanishing gradient problem.',
        ]),
        ('The Kantorovich-Rubinstein Duality — Making W Computable', C['blue'], [
            'The inf over all joint distributions γ in W(p,q) is intractable directly.',
            'The key mathematical result (Kantorovich-Rubinstein duality):',
            '',
            '  W(p_data, p_g) = sup_{||f||_L ≤ 1}  E_{x~p_data}[f(x)] - E_{x~p_g}[f(x)]',
            '',
            'where the sup is taken over all 1-Lipschitz functions f.',
            '',
            '1-Lipschitz means:  |f(x) - f(y)| ≤ |x - y|  for all x, y',
            '  (the function cannot change faster than the input changes).',
            '',
            'In the WGAN, the "Discriminator" D plays the role of f.',
            'We relabel it "Critic" (it outputs an unbounded real number, not a probability).',
            '',
            'New WGAN training objective:',
            '  min_G  max_{||D||_L ≤ 1}  E_{x~p_data}[D(x)] - E_{z~p_z}[D(G(z))]',
            '',
            'G minimises: wants -E[D(G(z))] to be small → D(G(z)) large → fools D.',
            'D maximises: wants E[D(x_real)] - E[D(x_fake)] to be large.',
            '  This approximates the Wasserstein distance between p_data and p_g.',
        ]),
    ]

    right = [
        ('Gradient Penalty: How to Enforce the Lipschitz Constraint', C['teal'], [
            'The original WGAN clipped weights to enforce Lipschitz (crude, slow).',
            'WGAN-GP (Gulrajani et al., 2017) uses a smoother penalty instead:',
            '',
            '  GP = λ · E_{x̂ ~ p_{x̂}} [ (||∇_{x̂} D(x̂)||_2  -  1)² ]',
            '',
            'where:',
            '  x̂ = ε · x_real + (1-ε) · x_fake    (a random interpolation)',
            '  ε ~ U[0,1]    (random mixing coefficient)',
            '  λ = 10        (penalty strength, standard value)',
            '',
            'Reading the penalty:',
            '  ||∇D(x̂)||_2 is the norm of D\'s gradient at the interpolated point.',
            '  For a 1-Lipschitz function, this norm should equal 1 everywhere.',
            '  The penalty pushes the gradient norm toward 1 throughout training.',
            '',
            'Full training loss (for each step):',
            '  L_D = E[D(x̃)] - E[D(x)]  +  λ · GP      (D maximises -L_D)',
            '  L_G = -E[D(G(z))]                         (G minimises L_G)',
            '',
            'n_critic = 3:  D is updated 3 times for each G update.',
            '  Ensures D stays near its optimum (approximates W well) before',
            '  G uses the gradient signal. This is critical for stability.',
        ]),
        ('Adam Optimiser Settings for WGAN-GP', C['amber'], [
            'Recommended by Gulrajani et al. (2017):  Adam(lr=1e-4, betas=(0, 0.9))',
            '',
            'Standard Adam uses betas=(0.9, 0.999):',
            '  β1 = 0.9 → exponential moving average of GRADIENT (momentum).',
            '  β2 = 0.999 → exponential moving average of squared gradient (RMS scaling).',
            '',
            'Why β1 = 0 for WGAN-GP?',
            '  Momentum accumulates past gradients. In adversarial training,',
            '  gradients reverse direction frequently (G and D take turns).',
            '  Momentum from the previous step can push in the wrong direction,',
            '  causing oscillations. Setting β1=0 disables momentum accumulation.',
            '  Only the current gradient (scaled by β2 RMS) is used.',
        ]),
        ('Why No BatchNorm in the WGAN-GP Critic?', C['red'], [
            'BatchNorm normalises activations across the batch:',
            '  output = (x - mean_batch) / std_batch * γ + β',
            '',
            'Problem: when computing the gradient penalty on interpolated points x̂,',
            '  the gradient ||∇D(x̂)|| depends on the batch statistics.',
            '  BatchNorm introduces a dependency on OTHER SAMPLES in the batch,',
            '  corrupting the gradient norm calculation.',
            '',
            'Solution (both QuantGAN and FinGAN): use LayerNorm instead.',
            '  LayerNorm normalises per-sample (across features, not across the batch):',
            '  output_i = (x_i - mean_i) / std_i * γ + β',
            '',
            'Each sample is normalised independently → gradient penalty is correct.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 6 — TimeGAN: Making the GAN Time-Aware
# ─────────────────────────────────────────────────────────────────────────────

def page_timegan(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'TimeGAN: Making the GAN Time-Aware',
               'Yoon, Jarrett & van der Schaar (2019) NeurIPS 32 — building on GRU + GAN')
    footer(fig, 6)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('The Problem a Plain GAN Cannot Solve', C['red'], [
            'A basic GAN generates each window independently from noise.',
            'It can learn to match the MARGINAL distribution (heavy tails, right scale),',
            'but it has no mechanism to enforce that consecutive timesteps within a window',
            'exhibit temporal patterns like volatility clustering.',
            '',
            'Analogy: asking someone to write a convincing novel by choosing',
            'each word at random from a realistic vocabulary. The individual words',
            'might all be real English, but the sequence makes no coherent sense.',
        ]),
        ('What is a GRU? (From Scratch)', C['blue'], [
            'A GRU (Gated Recurrent Unit) is a neural network designed for sequences.',
            'It processes one input at a time while maintaining a "hidden state" h_t,',
            'like a reader maintaining a running summary of a book.',
            '',
            'At each step t, the GRU receives:',
            '  • x_t   — the current input (e.g. the return on day t)',
            '  • h_{t-1} — the hidden state summarising all previous inputs',
            '',
            'And produces:',
            '  • h_t   — updated hidden state',
            '',
            'Internally, two "gates" control the information flow:',
            '  Reset gate r_t = σ(W_r · [h_{t-1}, x_t])    — how much of h_{t-1} to forget',
            '  Update gate z_t = σ(W_z · [h_{t-1}, x_t])   — how much of old h to keep',
            '  Candidate h̃_t = tanh(W · [r_t ∘ h_{t-1}, x_t])',
            '  New state  h_t = (1-z_t) ∘ h_{t-1} + z_t ∘ h̃_t',
            '',
            'Key property: h_t is a compressed summary of all inputs so far,',
            'carrying information about temporal patterns across many timesteps.',
        ]),
        ('TimeGAN\'s Key Innovation: Embedding Space', C['navy'], [
            'TimeGAN trains the GAN NOT in the raw return space X,',
            'but in a learned LATENT EMBEDDING SPACE H.',
            '',
            'Why? Raw returns are noisy and high-dimensional in the temporal sense.',
            'A smooth latent space H is easier to model with a GAN.',
            '',
            'The embedding also enforces temporal structure:',
            'the Supervisor network is trained to predict H_{t+1} from H_t,',
            'embedding the Markov dynamics of the data into the latent space.',
            'G then generates sequences in H that follow these learned dynamics.',
        ]),
    ]

    right = [
        ('The Four-Phase Training Algorithm', C['teal'], [
            'Phase 1 — Autoencoder Training  (epochs // 2 steps)',
            '  Embedder e(·): X → H    (GRU: raw returns → latent representation)',
            '  Recovery r(·): H → X̃   (GRU: latent → reconstructed returns)',
            '  Loss: L_R = ||X - r(e(X))||²',
            '  Goal: learn a faithful compressed representation of the return sequences.',
            '',
            'Phase 2 — Supervisor Training  (epochs // 2 steps)',
            '  Supervisor s(·): H → Ĥ  (GRU: predict next latent state)',
            '  Loss: L_S = ||H_{t+1} - s(H_t)||²  (step-ahead prediction in latent space)',
            '  Goal: embed the temporal dynamics of real returns into the latent space.',
            '  Generator g(·) is co-trained here to have outputs in the same latent space.',
            '',
            'Phase 3 — Joint Adversarial Training  (epochs steps)',
            '  Generator: Z ~ N(0,I) → g(Z) = Ê → s(Ê) = Ĥ → r(Ĥ) = X̃',
            '  Discriminator: operates on the hidden states H (not raw returns)',
            '',
            '  Generator loss:',
            '    L_U = -E[log D(s(g(Z)))]          (adversarial: fool discriminator)',
            '    L_S = ||H_{t+1} - s(H_t)||²        (maintain temporal dynamics)',
            '    L_V = |μ_H - μ_{Ĥ}| + |σ_H - σ_{Ĥ}|   (moment matching)',
            '    L_G = L_U + 100·sqrt(L_S) + 100·L_V',
            '',
            '  Discriminator loss:',
            '    L_D = BCE(D(H_real), 1) + BCE(D(H_fake), 0)',
        ]),
        ('Critical Limitation: Training Budget', C['red'], [
            'With the current setting of epochs = 5:',
            '  Phase 1 (autoencoder) runs for 5 // 2 = 2 gradient steps.',
            '  Phase 2 (supervisor)  runs for 5 // 2 = 2 gradient steps.',
            '',
            '2 gradient steps CANNOT train a GRU to convergence.',
            'The embedding space H is meaningless → Phase 3 trains on garbage.',
            '',
            'Yoon et al. (2019) used 5,000–10,000 iterations per phase.',
            '',
            'Consequence: TimeGAN\'s poor ranking (avg_rank=2.36 vs QuantGAN\'s 1.79)',
            'is almost certainly due to this budget issue, not a fundamental architectural',
            'weakness. A fair comparison requires epochs ≥ 50–100.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 7 — QuantGAN & FinGAN: Convolutional Approaches
# ─────────────────────────────────────────────────────────────────────────────

def page_conv_models(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'QuantGAN & FinGAN: Convolutional Approaches to Time Series Generation',
               'Building from convolutions to causal dilated TCNs to WGAN-GP generators')
    footer(fig, 7)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('What is a Convolution? (From Scratch)', C['navy'], [
            'A 1D convolution slides a small "filter" (kernel) over a sequence,',
            'computing a weighted sum at each position:',
            '',
            '  out[t] = sum_{k=0}^{K-1}  w_k · in[t - k]',
            '',
            'where w_0, ..., w_{K-1} are learnable weights (kernel size K).',
            '',
            'Think of it like a magnifying glass that scans the time series,',
            'highlighting patterns of a specific shape and length.',
            '',
            'Causal constraint — looking only into the past:',
            '  out[t] = sum_{k=0}^{K-1} w_k · in[t - k]',
            '  Position t can only see inputs at t, t-1, ..., t-(K-1).',
            '  No "future leakage" — important for financial time series modelling.',
            '',
            'Dilated convolution — skip steps to see further back cheaply:',
            '  out[t] = sum_{k=0}^{K-1} w_k · in[t - k·d]    (dilation d)',
            '  With d=1: looks at consecutive positions  [t, t-1, t-2].',
            '  With d=2: looks at every other position   [t, t-2, t-4].',
            '  With d=4:                                  [t, t-4, t-8].',
            '',
            'Stacking layers with dilation 1, 2, 4, 8, ... exponentially increases',
            'the receptive field (how far back a neuron can "see") without adding',
            'many parameters. For K=3, L=4 layers: receptive field = 1 + 2*(1+2+4+8) = 31.',
        ]),
        ('QuantGAN — TCN-based WGAN-GP (Wiese et al. 2020)', C['QG'], [
            'Key innovation: replace GRU with TCN (parallelisable, no vanishing gradients).',
            '',
            'Generator architecture:',
            '  z (noise_dim=100)  →  Linear  →  reshape (C, T/4)',
            '  →  ConvTranspose1d (stride=2, ×2 upsampling)',
            '  →  TCN (3 residual blocks, dilations 1/2/4)',
            '  →  Conv1d(1)  →  Tanh  →  x̃ (T×1 synthetic returns)',
            '',
            'Discriminator (Critic) architecture:',
            '  x (T×1)  →  permute  →  TCN (3 blocks, dilations 1/2/4)',
            '  →  AdaptiveAvgPool1d(1)  →  Linear(128)  →  LeakyReLU  →  Linear(1)',
            '',
            'Why AdaptiveAvgPool?  Averages the entire temporal dimension into',
            '1 number per channel → size-agnostic, captures global sequence statistics.',
            '',
            'Training: WGAN-GP as described on page 5.',
            '  n_critic=3: D updated 3× per G update.',
            '  lambda_gp=10: gradient penalty weight.',
        ]),
    ]

    right = [
        ('FinGAN — CNN Deconvolution WGAN-GP', C['FG'], [
            'A different convolutional approach: use transposed convolutions to',
            'directly "upsample" from a noise vector to a full time series.',
            '',
            'Transposed convolution (ConvTranspose1d):',
            '  The "reverse" of a strided convolution.',
            '  Inserts zeros between inputs, then applies a convolution.',
            '  With stride=2: every input step produces 2 output steps.',
            '  Three such layers: T/8 → T/4 → T/2 → T (8× upsampling).',
            '  This is why seq_len must be divisible by 8.',
            '',
            'Generator architecture:',
            '  z (noise_dim=100)  →  Linear+BN  →  reshape (C, T/8)',
            '  →  ConvTranspose1d(C/2, stride=2)  →  BN  →  LeakyReLU',
            '  →  ConvTranspose1d(C/4, stride=2)  →  BN  →  LeakyReLU',
            '  →  ConvTranspose1d(1, stride=2)    →  Tanh  →  x̃',
            '',
            'Critic (discriminator) architecture:',
            '  x (T×1)  →  Conv1d(C)   →  LayerNorm  →  LeakyReLU',
            '  →  Conv1d(2C, stride=2)  →  LayerNorm  →  LeakyReLU',
            '  →  Conv1d(4C, stride=2)  →  LayerNorm  →  LeakyReLU',
            '  →  Flatten  →  Linear(128)  →  Linear(1)',
            '',
            'LayerNorm (not BatchNorm) in the critic — as explained on page 5.',
        ]),
        ('QuantGAN vs FinGAN vs TimeGAN: Comparison', C['purple'], [
            'Architectural philosophy:',
            '  TimeGAN:   sequential (step-by-step); explicit temporal dynamics via GRU.',
            '             Most faithful to the sequential nature of time series.',
            '  QuantGAN:  parallel (full window at once); long-range via dilation.',
            '             Faster training; larger receptive field without depth.',
            '  FinGAN:    parallel; global upsampling from a single noise vector.',
            '             Simpler; fewer temporal inductive biases.',
            '',
            'Normalisation:',
            '  TimeGAN: MinMax [0,1] (required for sigmoid output activations).',
            '  QuantGAN & FinGAN: MinMax [-1,1] (required for Tanh output).',
            '',
            'Prior results (before metrics were updated):',
            '  QuantGAN: avg_rank = 1.79  (best overall)',
            '  FinGAN:   avg_rank = 1.85  (very close)',
            '  TimeGAN:  avg_rank = 2.36  (worst — likely due to epochs=5, not architecture)',
            '',
            'After increasing TimeGAN epochs to a fair budget (50–100),',
            'the ranking may change significantly.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 8 — Stylized Facts Reference Table
# ─────────────────────────────────────────────────────────────────────────────

def page_stylized_facts(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'What We Measure: The 10 Stylized Facts of Financial Returns',
               'Cont (2001) Quantitative Finance · Mandelbrot (1963) · Ding, Granger & Engle (1993)')
    footer(fig, 8)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.86])
    ax.axis('off')

    # Column left edges (axes coords 0-1).
    # Widths: #=0.022, Fact=0.178, Plain English=0.240, Math=0.240, Metric=0.320
    col_x    = [0.000, 0.022, 0.200, 0.440, 0.680]
    # Conservative wrap widths in chars (7.2pt proportional font, ~13 chars/inch):
    # axes widths * 11.22" * 13 chars/inch, then subtract ~15% for padding/var width
    wrap_ch  = [3,     26,    38,    38,    50]
    fs_body  = 7.2
    fs_hdr   = 8.0
    line_h   = 0.019   # axes units per line (≈ 0.135" @ 7.2pt with 1.3× leading)
    row_pad  = 0.007   # top+bottom padding inside each row

    def wcell(text, w):
        """Split on manual newlines first, then wrap each segment."""
        out = []
        for seg in text.split('\n'):
            if not seg.strip():
                out.append('')
            else:
                out.extend(textwrap.wrap(seg, width=w) or [''])
        return out

    rows_raw = [
        ['#', 'Stylized Fact', 'In Plain English', 'Mathematical Statement', 'Our Metric'],
        ['1', 'Heavy tails',
         'Extreme events occur far more often than a bell curve predicts.',
         'P(|r|>x) ~ x^{-α}, α≈3–5 (Pareto)\nkurtosis(r) >> 3',
         'Kurtosis diff · tail-index diff'],
        ['2', 'Near-zero autocorrelation',
         "Knowing today's direction gives no useful information about tomorrow's.",
         'Corr(r_t, r_{t+k}) ≈ 0  for k≥1\n(weak-form market efficiency)',
         'ACF returns MAE'],
        ['3', 'Volatility clustering',
         'Large moves tend to be followed by large moves (Mandelbrot 1963).',
         'Corr(|r_t|, |r_{t+k}|) > 0\nfor k = 1, ..., 100+',
         'ACF |returns| MAE · ARCH-LM stat diff'],
        ['4', 'Long memory in volatility',
         'The clustering effect persists for hundreds of trading days.',
         'ACF(|r_t|) ~ k^{-β}, β∈(0,1)\nHurst exponent H > 0.5',
         'Hurst exponent diff'],
        ['5', 'Gain / loss asymmetry',
         'Crashes are sharper and more extreme than equivalent-size rallies.',
         'skewness(r) < 0  for equity indices;\nleft tail heavier than right',
         'Skewness diff'],
        ['6', 'ARCH effects',
         'Return variance is not constant — it changes over time.',
         'Var(r_t | F_{t-1}) = σ²_t  time-varying;\nnot a constant σ²',
         'ARCH-LM test (Engle 1982)'],
        ['7', 'Conditional heavy tails',
         'After removing time-varying variance, residuals are still non-Gaussian.',
         'ε_t = r_t/σ_t  still has kurtosis >> 3\n(GARCH standardised residuals)',
         'Kurtosis diff on GARCH residuals (ext.)'],
        ['8', 'Extreme events',
         'Very large days occur far more frequently than Gaussian probability predicts.',
         '% of days with |r|>2σ exceeds\n5% expected under N(0,1)',
         'Extreme events diff'],
        ['9', 'Distributional match',
         'The full shape of the return distribution — not just its tails — must be reproduced.',
         'W_1(p_g, p_data) ≈ 0\nF̂_syn ≈ F̂_real  (quantile-by-quantile)',
         'Wasserstein · Quantile MSE · Energy distance'],
        ['10', 'Indistinguishability',
         'A classifier trained to separate real from synthetic should perform at chance level.',
         'AUC(classifier) → 0.5\n(p_g = p_data at GAN optimum)',
         'Discriminative AUC'],
    ]

    # Pre-wrap every cell
    wrapped = [[wcell(cell, w) for cell, w in zip(row, wrap_ch)] for row in rows_raw]

    # Variable row heights based on max lines in that row
    def rh(wr):
        return max(len(wc) for wc in wr) * line_h + 2 * row_pad

    heights = [rh(wr) for wr in wrapped]
    total   = sum(heights)
    # Scale to fill 93% of axis height so rows are evenly spaced and page is well used
    scale   = 0.93 / total
    heights = [h * scale for h in heights]
    lh      = line_h * scale
    rp      = row_pad * scale

    y = 0.97
    for r_i, (wr, row_h) in enumerate(zip(wrapped, heights)):
        is_hdr = r_i == 0
        bg = C['navy'] if is_hdr else ('#EEF2F7' if r_i % 2 == 0 else '#FAFAFA')
        fc = 'white' if is_hdr else C['grey']
        fw = 'bold' if is_hdr else 'normal'
        fs = fs_hdr if is_hdr else fs_body

        ax.add_patch(plt.Rectangle((0, y - row_h), 1, row_h,
                     transform=ax.transAxes, facecolor=bg,
                     edgecolor='#CFD8DC', linewidth=0.3, clip_on=True))

        for ci, (wc, cx) in enumerate(zip(wr, col_x)):
            ty = y - rp
            for ln in wc:
                ax.text(cx + 0.005, ty, ln,
                        transform=ax.transAxes, fontsize=fs,
                        color=fc, va='top', fontweight=fw, clip_on=True)
                ty -= lh

        y -= row_h

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 9 — Why Classic Tests Fail (i.i.d. Problem)
# ─────────────────────────────────────────────────────────────────────────────

def page_metrics_iid(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Evaluation Metrics Part 1 — Why Standard Statistical Tests Fail',
               'The i.i.d. assumption: what it means, why financial series violate it, mathematical consequences')
    footer(fig, 9)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('What Does i.i.d. Mean?', C['navy'], [
            'A sample {X_1, X_2, ..., X_n} is i.i.d. (independent and identically distributed) if:',
            '',
            '  • Identically distributed: every X_i has the same distribution F.',
            '  • Independent: knowing X_1, ..., X_{t-1} gives NO information about X_t.',
            '',
            'Coin flips are i.i.d.: the 100th flip is unaffected by the previous 99.',
            '',
            'Financial returns are NOT i.i.d.:',
            '  • The size of |r_{t+1}| is correlated with |r_t| (volatility clustering).',
            '  • Large shocks cluster together — ARCH effects mean Var(r_t|past) ≠ const.',
            '  • The autocorrelation of |r_t| is still positive at lag 100+ (long memory).',
            '',
            'Most classical statistical tests are DERIVED under the i.i.d. assumption.',
            'Applying them to time series requires understanding the consequences.',
        ]),
        ('The Kolmogorov-Smirnov Test — What It Really Tests', C['blue'], [
            'The two-sample KS statistic:',
            '',
            '  D_{n,m} = sup_x  | F̂_n(x) - Ĝ_m(x) |',
            '',
            'F̂_n, Ĝ_m are empirical CDFs of two samples of size n and m.',
            '',
            'Under H_0 (same distribution) AND i.i.d. observations:',
            '',
            '  sqrt(nm/(n+m)) · D_{n,m}  →  K  (Kolmogorov distribution)',
            '',
            'The p-value is derived from this limiting distribution.',
            '',
            'For dependent series, the CLT underlying this limit changes:',
            '',
            '  Var(n^{-1} sum_t X_t) = (σ²/n) · (1 + 2·sum_{k=1}^∞ ρ_k)',
            '',
            'For financial |returns|, sum_{k=1}^{100} ρ_{|r|}(k) ≈ 5 to 15:',
            '',
            '  n_eff = n / (1 + 2·sum_k ρ_k)  ≈  n/11  to  n/31',
            '',
            'Consequence: the test uses n but the effective information is only n_eff.',
            '  → The test statistic is INFLATED relative to the null distribution.',
            '  → P-values are TOO SMALL (anti-conservative).',
            '  → We over-reject H_0: "not the same distribution" even when the',
            '    MARGINAL distributions match, because temporal dependence is mistaken',
            '    for distributional difference.',
        ]),
    ]

    right = [
        ('Welch\'s t-test — A Narrower Failure', C['amber'], [
            'Welch\'s t-test checks H_0: μ_real = μ_synthetic:',
            '',
            '  t = (X̄ - Ȳ) / sqrt(s²_X/n + s²_Y/m)',
            '',
            'For i.i.d. data: s²_X/n consistently estimates Var(X̄) = σ²_X/n.',
            '',
            'For dependent data: the correct estimator is:',
            '',
            '  Var(X̄) = (σ²_X/n) · (1 + 2·sum_{k=1}^∞ ρ_k(X))',
            '',
            'For log returns, ρ_k(r_t) ≈ 0 for k ≥ 1,',
            '  so Welch\'s t-test is approximately valid for testing mean equality.',
            '',
            'However, it tests ONLY the mean — providing no information about',
            'tails, volatility clustering, or any other stylized fact.',
            '',
            'Using Welch\'s t-test as the primary evaluation metric (as was done',
            'in the original thesis for LLM evaluation) tells us almost nothing',
            'about whether the synthetic data is financially realistic.',
        ]),
        ('What Can the KS Test Still Tell Us?', C['teal'], [
            'Despite its limitation, the KS test is not useless:',
            '',
            '  • It tests the MARGINAL distribution F(x) = P(r ≤ x).',
            '  • The marginal distribution is well-defined even for stationary',
            '    dependent series (strict stationarity is sufficient).',
            '  • A GAN that produces the wrong scale, wrong sign, or obviously',
            '    non-financial-looking returns will fail the KS test.',
            '',
            'Strategy used in this project:',
            '',
            '  (a) Report KS statistic alongside the explicit caveat about n_eff.',
            '      Note: the p-value should not be used for formal inference.',
            '',
            '  (b) Use Wasserstein distance W_1(p_g, p_data) as the primary',
            '      distributional metric. W_1 is a proper metric on distributions',
            '      and is more stable under temporal dependence than KS p-values.',
            '',
            '  (c) Add time-series-aware tests (ARCH-LM, Hurst, discriminative AUC)',
            '      to capture what KS and Welch cannot test.',
        ]),
        ('Why Pointwise MSE/MAE Are Wrong', C['red'], [
            'The evaluation previously used:',
            '  MSE = mean_t (r_real(t) - r_synthetic(t))²',
            '',
            'This compares the i-th real return to the i-th synthetic return.',
            'But real and synthetic are INDEPENDENTLY generated time series.',
            'There is no meaningful alignment between their temporal indices.',
            '',
            'Analogy: measuring the "error" between two random walks by',
            'comparing step 1 to step 1, step 2 to step 2, etc.',
            'This measures nothing about distribution similarity.',
            '',
            'Fix: use quantile MSE (compares sorted distributions) and',
            'energy distance (proper metric on the space of distributions).',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 10 — New Time-Series-Aware Metrics
# ─────────────────────────────────────────────────────────────────────────────

def page_metrics_new(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Evaluation Metrics Part 2 — New Time-Series-Aware Tests Added',
               'ARCH-LM · Hurst exponent · Energy distance · Discriminative AUC · Quantile MSE')
    footer(fig, 10)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('1. ARCH-LM Test — Does the GAN Reproduce Volatility Clustering?', C['red'], [
            'Problem it solves: the KS test cannot detect whether synthetic returns',
            'have volatility clustering. A GAN could pass KS (correct marginal) but',
            'generate i.i.d. returns — no clustering.',
            '',
            'What it tests:  H_0 = no ARCH effects (homoskedasticity).',
            '',
            'Procedure (Engle 1982):',
            '  (1) Regress squared returns on their own lags:',
            '      ε̂²_t = α_0 + α_1·ε̂²_{t-1} + ... + α_q·ε̂²_{t-q} + v_t',
            '  (2) Compute LM = n · R²  where R² is from this regression.',
            '  (3) Under H_0:  LM ~ χ²(q)',
            '  (4) If p < 0.05: reject H_0 → ARCH effects present.',
            '',
            'How we use it:',
            '  Real BRICS returns: consistently show ARCH (p << 0.05).',
            '  Synthetic from a good GAN: should also show ARCH.',
            '  Our metric: |LM_real - LM_synthetic|  →  0 is ideal.',
            '  We also check arch_reproduced: does the YES/NO match?',
        ]),
        ('2. Hurst Exponent — Does the GAN Reproduce Long Memory?', C['purple'], [
            'Problem it solves: detects whether |returns| have the slow ACF decay',
            '(long memory, stylized fact 4) that real financial series exhibit.',
            '',
            'Physical intuition: the Hurst exponent H characterises the scaling',
            'behaviour of the "rescaled range" R/S of a time series.',
            '  E[R_n / S_n]  ~  c · n^H',
            '',
            'where R_n = range of partial sums,  S_n = standard deviation.',
            '',
            'Interpretation:',
            '  H = 0.5 → Brownian motion (independent increments, no memory).',
            '  H > 0.5 → persistent long memory (trends reinforce themselves).',
            '  H < 0.5 → mean-reverting (anti-persistent).',
            '',
            'For financial log returns r_t: H ≈ 0.5 (efficient market).',
            'For |r_t| (absolute returns): H ≈ 0.6–0.7 (long memory).',
            '',
            'A GAN that generates clustered volatility correctly should',
            'produce |synthetic returns| with H significantly above 0.5.',
            '',
            'Our metric: |H_real - H_synthetic|  →  0 is ideal.',
            'Ref: Hurst (1951); Ding, Granger & Engle (1993).',
        ]),
    ]

    right = [
        ('3. Energy Distance — A Proper Distributional Metric', C['blue'], [
            'The energy distance between distributions P and Q (Székely & Rizzo 2004):',
            '',
            '  E(P,Q) = 2·E||X-Y|| - E||X-X\'|| - E||Y-Y\'||',
            '',
            'where X,X\'~P and Y,Y\'~Q are independent copies.',
            '',
            'Plain English: it measures the average distance between a random sample',
            'from P and one from Q, corrected for within-distribution spread.',
            '',
            'Key properties:',
            '  • E(P,Q) ≥ 0, with equality iff P = Q. A true metric on distributions.',
            '  • Works on any dimension; does not require a density function.',
            '  • Does not assume independence: uses marginal expectations.',
            '',
            'Estimated using n=min(500,N) subsampled pairs (O(n²) cost, fast in practice).',
            '',
            'Advantage over Wasserstein: simpler to estimate in practice;',
            'advantage over KS: a proper metric, not just a supremum distance.',
        ]),
        ('4. Discriminative AUC — The Holistic Score', C['teal'], [
            'The gold-standard evaluation for GANs: can a classifier tell real from fake?',
            '',
            'Procedure (TSGBench, Ang et al. 2023):',
            '  1. Extract rolling-window features from both real and synthetic:',
            '     (mean, std, skewness, kurtosis, |r| mean, |r| max) per window.',
            '  2. Label real windows 0, synthetic windows 1.',
            '  3. Train logistic regression classifier.',
            '  4. Report AUC from 5-fold cross-validation.',
            '',
            'AUC = 0.5 → classifier performs at chance → distributions indistinguishable.',
            'AUC = 1.0 → trivially separable → GAN has failed.',
            '',
            'Why is this the best single summary?',
            '  It captures distributional AND temporal structure simultaneously.',
            '  The rolling-window features encode local time-series patterns.',
            '  A GAN that has the right marginal but wrong temporal structure',
            '  will still show AUC > 0.5 (windows look different in feature space).',
        ]),
        ('5. Quantile MSE — Proper Distributional Shape Comparison', C['amber'], [
            'Motivation: pointwise MSE(real, synthetic) is meaningless (no alignment).',
            '',
            'Quantile MSE compares the empirical quantile functions:',
            '',
            '  QMSE = (1/K) · sum_{k=1}^K  (Q_real(α_k) - Q_syn(α_k))²',
            '',
            'where α_k = k/(K+1), K=99 (1st to 99th percentile).',
            '',
            'Plain English: sort the real returns and the synthetic returns separately.',
            'Compare the i-th smallest real return to the i-th smallest synthetic return.',
            'This is the only meaningful way to compare two unaligned distributions.',
            '',
            'QMSE = 0 iff F̂_real = F̂_syn (identical empirical distributions).',
            'Equivalent to measuring the L2 distance between the CDFs.',
            '',
            'Advantage: captures shape differences at every percentile,',
            'including tails (5th, 1st percentile) which are crucial for finance.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 11 — Code Audit
# ─────────────────────────────────────────────────────────────────────────────

def page_audit(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Code Audit: Issues Found, Mathematical Justification & Changes Applied',
               '3_4_integrated_pipeline.ipynb — reviewed and updated July 2026')
    footer(fig, 11)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis('off')

    STATUS_W = 0.062
    items = [
        ('FIXED', C['red'], '#FFEBEE',
         'KS / Welch tests assumed i.i.d. — CRITICAL',
         'n_eff = n/(1+2Σρ_k) << n for financial series (sum of ACF can be 5–15 for |returns|). '
         'This inflates the KS statistic, making p-values anti-conservative. Fix: added ARCH-LM '
         'test, Hurst exponent, energy distance, and discriminative AUC. KS retained with caveat; '
         'Wasserstein used as the primary distributional metric.'),
        ('FIXED', C['red'], '#FFEBEE',
         'Pointwise MSE/MAE had no semantic meaning',
         'MSE compared r_real(t) to r_syn(t) but real and synthetic are independently generated: '
         'no temporal alignment exists between their indices. Fix: replaced with Quantile MSE '
         '(L2 distance between empirical quantile functions) and Energy distance (Székely & Rizzo '
         '2004 — a proper metric on the space of distributions).'),
        ('DECISION', C['navy'], '#E8EAF6',
         'No outlier clipping applied to log returns — deliberate methodological choice',
         'The preprocessing notebook (cell 25) computed 0.5th/99.9th percentile bounds on raw '
         'price columns but the clip line was commented out and was never applied to log returns. '
         'This is kept intentionally: clipping at the 0.5th/99.5th percentile removes exactly the '
         'observations that determine kurtosis, tail index, and the "% of |r|>2σ" metric — all of '
         'which appear in the evaluation table. Truncating the tail and then reporting that models '
         'reproduce tails is circular. Ref: Adams et al. (2019, Financial Management) show '
         'winsorizing/trimming can worsen rather than fix distributional misfit. LeBaron\'s EVT '
         'work further shows BRICS markets have systematically fatter tails than developed ones — '
         'the very property motivating the BRICS choice. Raw returns are therefore kept intact.'),
        ('FIXED', C['amber'], '#FFF3E0',
         'Ljung-Box test used deprecated statsmodels API',
         'acorr_ljungbox(..., return_df=False) returns a DataFrame in statsmodels >= 0.13, not a '
         'tuple. Accessing lb[1][-1] raised TypeError. Fix: return_df=True, lags=[n] (list syntax), '
         'accessed via .iloc[-1]["lb_pvalue"].'),
        ('FIXED', C['amber'], '#FFF3E0',
         'ResidualBlock class defined but never used in FinGAN cell',
         'class ResidualBlock(nn.Module) was defined at the top of the FinGAN cell but neither '
         'FinGAN_Generator nor FinGAN_Critic references it anywhere. Dead code creates confusion '
         'for readers who may think residual connections are active. Fix: class removed entirely.'),
        ('FIXED', C['amber'], '#FFF3E0',
         'Grid search score_cols included pointwise MSE and MAE',
         'The composite ranking used to select hyperparameter configurations included "mse" and '
         '"mae" — which as described above compare unaligned series (meaningless). Fix: removed '
         'mse/mae from score_cols; added hurst_diff, energy_distance, arch_stat_diff.'),
        ('OK', C['green'], '#E8F5E9',
         'WGAN-GP gradient penalty computation is correct (QuantGAN & FinGAN)',
         'x̂ = ε·x_real + (1-ε)·x_fake; x̂.requires_grad_(True); gradients computed w.r.t. x̂ via '
         'torch.autograd.grad with create_graph=True. fake.detach() ensures fake does not '
         'accumulate spurious gradients. This correctly implements Gulrajani et al. (2017) Eq. 3.'),
        ('OK', C['green'], '#E8F5E9',
         'Adam betas=(0.0, 0.9) for WGAN-GP — correct',
         'β1=0 disables first-moment (momentum) accumulation. In adversarial training, gradients '
         'change direction frequently; momentum from the previous step destabilises training. '
         'β2=0.9 retains RMS adaptive scaling. Recommended explicitly by Gulrajani et al. (2017).'),
        ('CONCERN', C['amber'], '#FFFDE7',
         'TimeGAN training budget of epochs=5 is scientifically insufficient',
         'Phase 1 (autoencoder) gets 5//2=2 gradient steps; Phase 2 (supervisor) gets 2 steps. '
         'A GRU cannot converge in 2 steps. The embedding space H is meaningless → Phase 3 trains '
         'on noise. Yoon et al. (2019) use 5,000–10,000 iterations per phase. TimeGAN\'s poor '
         'ranking almost certainly reflects training budget, not architecture. Requires fix.'),
    ]

    y = 0.97
    row_h = 0.97 / len(items)
    for status, color, bg, title, desc in items:
        h = row_h - 0.004
        ax.add_patch(plt.Rectangle((0, y - h), 1.0, h,
                     transform=ax.transAxes, facecolor=bg,
                     edgecolor='#E0E0E0', linewidth=0.4, clip_on=True))
        ax.add_patch(plt.Rectangle((0, y - h), STATUS_W, h,
                     transform=ax.transAxes, facecolor=color,
                     edgecolor='none', clip_on=True))
        ax.text(STATUS_W / 2, y - h / 2, status,
                transform=ax.transAxes, fontsize=6, fontweight='bold',
                color='white', ha='center', va='center', rotation=90, clip_on=True)
        ax.text(STATUS_W + 0.008, y - 0.007, title, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=C['grey'], va='top', clip_on=True)
        wrapped = textwrap.fill(desc, width=148)
        for i, ln in enumerate(wrapped.split('\n')):
            ax.text(STATUS_W + 0.008, y - 0.007 - (i + 1) * 0.021, ln,
                    transform=ax.transAxes, fontsize=7.5,
                    color='#555555', va='top', clip_on=True)
        y -= row_h

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 12 — Grid Search
# ─────────────────────────────────────────────────────────────────────────────

def page_gridsearch(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Hyperparameter Grid Search: Why, How, and What It Documents',
               'Required for publication: systematic evidence that parameters were tested before selection')
    footer(fig, 12)

    gs = gridspec.GridSpec(1, 2, figure=fig,
                           left=0.03, right=0.97, top=0.89, bottom=0.04, wspace=0.06)
    axl = fig.add_subplot(gs[0]); axl.axis('off')
    axr = fig.add_subplot(gs[1]); axr.axis('off')

    left = [
        ('Why Grid Search is Required for Publication', C['navy'], [
            'A common criticism in ML papers: "how were hyperparameters chosen?"',
            'If the answer is "we tried a few and picked the best-looking result",',
            'reviewers rightly question whether the result is reproducible or cherry-picked.',
            '',
            'Grid search provides a paper trail:',
            '  • The search space is defined ex-ante (before seeing results).',
            '  • Selection criterion is objective (composite rank of stylized-fact metrics).',
            '  • Every candidate configuration is documented in a CSV (appendix).',
            '  • The winning configuration is then retrained at full epochs.',
            '',
            'This turns hyperparameter choice from a hidden decision into a',
            'documented, reproducible scientific step.',
        ]),
        ('The Search Protocol', C['blue'], [
            'For each (model, config) combination:',
            '  1. Set config[epochs] = 3  (reduced budget for speed)',
            '     Set config[seed]   = 42  (same seed for all candidates)',
            '  2. Train model on pooled BRICS training windows.',
            '  3. Generate synthetic series of same length as validation set.',
            '  4. Score with FinancialMetrics.compute_all_metrics().',
            '  5. Rank across all candidates on each metric (rank 1 = best).',
            '  6. composite_score = mean(all ranks).',
            '  7. Best config → retrain at full epochs (5) for final results.',
            '',
            'Why search_epochs=3 < final_epochs=5?',
            '  Reduced budget (coarse-to-fine) is standard practice (Bergstra 2012).',
            '  At 3 epochs each model produces a rough relative ranking;',
            '  the cardinal value of the score does not need to be precise.',
        ]),
        ('Composite Scoring Formula', C['teal'], [
            'For each metric m ∈ score_cols and each candidate config_i:',
            '',
            '  rank_m(config_i) = rank by |metric_m(config_i)|  (1 = best, 8 = worst)',
            '',
            '  composite_score_i = (1/|score_cols|) · sum_m rank_m(config_i)',
            '',
            'score_cols = {wasserstein, ks_statistic, kurtosis_diff, skewness_diff,',
            '              acf_returns_mae, acf_absolute_mae, tail_index_diff,',
            '              hurst_diff, energy_distance, arch_stat_diff}',
            '',
            'Excluded from scoring:',
            '  mse, mae — pointwise metrics (no alignment, meaningless)',
            '  ks_pvalue — correlated with ks_statistic, double-counting',
        ]),
    ]

    right = [
        ('Search Space (8 candidates = 2³ per model)', C['navy'], [
            'TimeGAN:  hidden_dim ∈ {24, 48}  ·  num_layers ∈ {2, 3}  ·  lr ∈ {1e-3, 5e-4}',
            'QuantGAN: lr ∈ {1e-4, 2e-4}  ·  n_critic ∈ {3, 5}  ·  lambda_gp ∈ {5, 10}',
            'FinGAN:   lr ∈ {1e-4, 2e-4}  ·  n_critic ∈ {3, 5}  ·  base_channels ∈ {32, 64}',
            '',
            'Fixed across all: noise_dim=100 (QuantGAN/FinGAN), noise_dim=24 (TimeGAN).',
            'Batch size: 128 (TimeGAN), 64 (QuantGAN/FinGAN).',
            'All candidates use seed=42 — ensures identical initialisation for fair comparison.',
        ]),
        ('Reproducibility', C['purple'], [
            'Each candidate calls torch.manual_seed(seed) + np.random.seed(seed).',
            'Same config + same seed → identical weights and training trajectory.',
            'Results saved to: grid_search_results/{Model}_grid_search.csv',
            '  Columns: all searched params + all metric values + composite_score.',
            '  This CSV is the appendix for hyperparameter documentation in the paper.',
        ]),
        ('Recommended Extensions for the Paper', C['amber'], [
            '• Extend to 20–30 configs × 10 epochs per candidate (currently 8 × 3).',
            '• Use Bayesian optimisation (Optuna) for more efficient exploration.',
            '• Ablation table: vary one parameter at a time, fix all others;',
            '  report the marginal effect on composite score.',
            '• 5-fold cross-validation across market splits for robust scoring.',
            '• Report the best config with 95% CI on composite score across seeds.',
        ]),
    ]

    for ax, blocks in [(axl, left), (axr, right)]:
        y = 0.98
        for title, color, lines in blocks:
            y = titled_col(ax, y, title, color, lines, dy=0.028, gap=0.018)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Page 13 — Ranked Next Steps
# ─────────────────────────────────────────────────────────────────────────────

def page_next_steps(pdf):
    fig = plt.figure(figsize=PAGE)
    header_bar(fig, 'Ranked Next Steps for Paper Publication',
               'Priority order based on impact on scientific rigour and reviewability')
    footer(fig, 13)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.86])
    ax.axis('off')

    steps = [
        ('CRITICAL', C['red'],   '#FFEBEE',
         '1 — Re-run the full pipeline and regenerate all results',
         'Evaluation metrics schema changed (ARCH-LM, Hurst, discriminative AUC, energy distance '
         'added; pointwise MSE/MAE removed from ranking). All files in thesis_results/ are stale. '
         'Must regenerate before any comparative analysis or figure creation for the paper.'),

        ('CRITICAL', C['red'],   '#FFEBEE',
         '2 — Fix TimeGAN training budget (epochs=5 is scientifically unsound)',
         'Phases 1 and 2 each get 2 gradient steps. The Yoon et al. (2019) paper uses 5,000–10,000 '
         'iterations per phase. TimeGAN\'s poor ranking almost certainly reflects this, not its '
         'architecture. Recommendation: set epochs ≥ 50 or add explicit per-phase iteration counts.'),

        ('HIGH',     C['amber'], '#FFF3E0',
         '3 — Robustness test on a different asset class (FX or Crypto)',
         'Supervisor request: train the best GAN (QuantGAN) on BRICS, then evaluate on EUR/USD or '
         'BTC/USD daily log returns. This validates out-of-distribution generalisation — a key '
         'contribution that differentiates the paper from single-asset GAN studies.'),

        ('HIGH',     C['amber'], '#FFF3E0',
         '4 — Add or discuss diffusion model baseline',
         'Takahashi & Mizuno (2025, Quant. Finance 25(10)) showed diffusion models outperform GANs '
         'on several stylized-fact metrics. Reviewers will ask why diffusion was not included. At '
         'minimum: dedicated Discussion section. Best: include a simple DDPM or score-matching '
         'baseline using an open-source implementation (e.g. denoising-diffusion-pytorch).'),

        ('HIGH',     C['amber'], '#FFF3E0',
         '5 — Document all LLM parameters for reproducibility',
         'DeepSeek experiments: report temperature, top_p, max_tokens, system prompt, user prompt. '
         'LoRA fine-tuning: r=8, alpha=16, target=[q_proj,v_proj], dropout=0.1, training epochs, '
         'batch size, gradient accumulation. Upload code to Papers with Code (paperswithcode.com).'),

        ('MEDIUM',   '#F9A825', '#FFFDE7',
         '6 — Expand grid search and publish full ablation table',
         'Current: 8 configs × 3 epochs. For publication: extend to 20–30 configs × 10 epochs; '
         'present ablation table showing marginal effect of each hyperparameter. This turns the '
         'grid search from a box-checking exercise into a publishable analysis.'),

        ('MEDIUM',   '#F9A825', '#FFFDE7',
         '7 — Update literature review with 2024/2025 papers',
         'Must cite: Meldrum et al. (2025) arXiv:2510.26076; SFAG (2026) arXiv:2601.12990; '
         'TSGBench Ang et al. (2023) PVLDB 17(3); Chronos Ansari et al. (2024) arXiv:2403.07815; '
         'LLMTime Gruver et al. NeurIPS 2023; Forging TS Hamdouche et al. (2025) arXiv:2505.17103.'),

        ('LOW',      C['green'], '#E8F5E9',
         '8 — LLM diversity analysis: correlation between independent generation runs',
         'Generate 10+ independent synthetic series from each LLM model. Compute pairwise '
         'Pearson correlation and Wasserstein distances. Mode collapse → high correlation. '
         'Diversity → low. This addresses the supervisor\'s question about LLM output diversity.'),

        ('LOW',      C['green'], '#E8F5E9',
         '9 — Fix cosmetic filename strip in validate_models()',
         'validate_models() calls replace("_val.parquet","") but files are named "_valid.parquet". '
         'Display names appear as "BOVESPA_valid" instead of "BOVESPA". One-line fix; no effect on results.'),
    ]

    n = len(steps)
    row_h = 0.96 / n
    y = 0.97
    for status, color, bg, title, desc in steps:
        h = row_h - 0.004
        ax.add_patch(plt.Rectangle((0, y - h), 1.0, h,
                     transform=ax.transAxes, facecolor=bg,
                     edgecolor='#E0E0E0', linewidth=0.4, clip_on=True))
        ax.add_patch(plt.Rectangle((0, y - h), 0.055, h,
                     transform=ax.transAxes, facecolor=color,
                     edgecolor='none', clip_on=True))
        ax.text(0.0275, y - h / 2, status,
                transform=ax.transAxes, fontsize=5.8, fontweight='bold',
                color='white', ha='center', va='center', rotation=90, clip_on=True)
        ax.text(0.063, y - 0.006, title,
                transform=ax.transAxes, fontsize=8.4, fontweight='bold',
                color=C['grey'], va='top', clip_on=True)
        # wrap description to fixed width
        wrapped = textwrap.fill(desc, width=148)
        for i, ln in enumerate(wrapped.split('\n')):
            ax.text(0.063, y - 0.006 - (i + 1) * 0.019, ln,
                    transform=ax.transAxes, fontsize=7.4,
                    color='#555555', va='top', clip_on=True)
        y -= row_h

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('reports', exist_ok=True)
    print(f'Generating: {OUTPUT_PDF}  (version {_DATE})')
    with PdfPages(OUTPUT_PDF) as pdf:
        page_title(pdf);          print('  p1   Title page')
        page_overview(pdf);       print('  p2   Project overview (fixed spacing)')
        page_problem(pdf);        print('  p3   The problem: financial time series from scratch')
        page_gan_intro(pdf);      print('  p4   What is a GAN? (from scratch)')
        page_wgan(pdf);           print('  p5   WGAN-GP: Wasserstein + gradient penalty')
        page_timegan(pdf);        print('  p6   TimeGAN deep dive')
        page_conv_models(pdf);    print('  p7   QuantGAN & FinGAN deep dive')
        page_stylized_facts(pdf); print('  p8   Stylized facts reference table')
        page_metrics_iid(pdf);    print('  p9   Why classic tests fail (i.i.d. problem)')
        page_metrics_new(pdf);    print('  p10  New time-series-aware metrics')
        page_audit(pdf);          print('  p11  Code audit')
        page_gridsearch(pdf);     print('  p12  Grid search methodology')
        page_next_steps(pdf);     print('  p13  Ranked next steps')

        d = pdf.infodict()
        d['Title']   = 'Synthetic Financial Time Series — GAN Pipeline Report'
        d['Author']  = 'UPC Research Collaboration'
        d['Subject'] = 'TimeGAN · QuantGAN · FinGAN on BRICS Emerging Market Indices'

    kb = os.path.getsize(OUTPUT_PDF) / 1024
    print(f'\nSaved: {OUTPUT_PDF}  ({kb:.0f} KB, 13 pages)')


if __name__ == '__main__':
    main()
