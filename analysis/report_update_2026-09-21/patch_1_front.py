"""Group 1: mapping hook, Highlights (headline, tiers, AUC), and the new 'Changes since the previous version' section."""
import sys
sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from patch_lib import *
t = load()

# ---- the mapping: every key of the update-evidence dict becomes a ${RPT_U_<KEY>} placeholder (the validator still fails on any unmapped one)
t = replace_once(t, '    mapping.update(guard_cells)\n',
                 '    mapping.update(guard_cells)\n'
                 '    # Findings added on 2026-09-21: read from artifacts by report_data.build_update_evidence; one placeholder per key.\n'
                 '    mapping.update({f"RPT_U_{k.upper()}": v for k, v in fmt["update"].items()})\n')

# ---- Highlights
t = replace_line(t, r'\item \textbf{A ${RPT_COMPUTE_BEST_PARAMS}-parameter model ranks first.}',
 r'''\item \textbf{First place is not resolved: a ${RPT_U_S4_FIRST_PARAMS}-parameter model is statistically indistinguishable from a ${RPT_U_S4_SECOND_PARAMS}-parameter one.} ${RPT_U_S4_FIRST}, fitted in ${RPT_U_S4_FIRST_SECONDS}\,s, has the best mean composite rank on the published draw (${RPT_BEST_COMPOSITE}, Section~\ref{sec:ranking}), but its margin over ${RPT_U_S4_SECOND} (${RPT_U_S4_SECOND_PARAMS} parameters, ${RPT_U_S4_SECOND_SECONDS}\,s per fit) is ${RPT_U_S4_MARGIN} composite points, with a 95\,\% bootstrap interval of ${RPT_U_S4_CI} over ${RPT_U_S4_N_UNITS} market--seed units (${RPT_U_S4_RESAMPLES} resamples): the interval ${RPT_U_S4_EXCLUDES_ZERO} zero. It spans about ${RPT_U_S4_WIDTH} composite points, so this benchmark cannot resolve a first--second difference smaller than about ${RPT_U_S4_HALFWIDTH}. Both models clear every fair baseline drawn from the training block alone (Section~\ref{sec:tiers}). The largest generator, ${RPT_COMPUTE_BIGGEST_MODEL}, has ${RPT_COMPUTE_BIGGEST_PARAMS} parameters and trains for ${RPT_COMPUTE_BIGGEST_SECONDS}\,s per market--seed, and ranks below the small model ${RPT_COMPUTE_BIGGEST_BELOW}.''')

# tiers: a new bullet directly after the 'margin is thin' bullet
t = replace_once(t, '\n\\item \\textbf{A shuffled copy of the real data beats most generators on volatility clustering.}',
 '\n' + r'''\item \textbf{Fair baselines split the field into three tiers.} Two baselines that read only the training block --- i.i.d.\ historical simulation and a stationary block bootstrap with blocks of about ${RPT_U_PW_ABS_RANGE} days --- separate the generators. ${RPT_U_TIER_TOP} clear every fair baseline; ${RPT_U_TIER_MIDDLE} are indistinguishable from the block bootstrap; ${RPT_U_TIER_BOTTOM} ranks ${RPT_U_BOTTOM_POSITION} of ${RPT_U_N_ENTRIES}, below i.i.d.\ historical simulation (Section~\ref{sec:tiers}). The shuffled control would win the composite if it were allowed to compete (${RPT_U_CTRL_COMP} against ${RPT_U_CTRL_BEST_COMP}), because it is the test block's own returns, permuted: the fidelity family accounts for ${RPT_U_CTRL_FID_SHARE_S0}\,\% to ${RPT_U_CTRL_FID_SHARE_S4}\,\% of its gap to i.i.d.\ historical simulation, depending on the scheme.
\item \textbf{A shuffled copy of the real data beats most generators on volatility clustering.}''')

# AUC bullet
t = replace_line(t, r"\item \textbf{Every generator's AUC lies above the real-versus-real null, but no result reaches the 5\,\% level.}",
 r'''\item \textbf{The discriminative AUC has no working reference here, and it no longer counts toward the rank.} The null is not $0.5$: on the real series its mean is ${RPT_U_NULL_MEAN_RNG_TA} across the ${RPT_U_NULL_N_MARKETS} markets at the fixed-split length, and a stationary GARCH-$t$ reference, the best case, still gives ${RPT_U_STAT_MEAN_RNG_WF} at walk-forward length. On fresh generation draws ${RPT_U_CELLSF_P975} of ${RPT_U_CELLS_N} market--model cells lie above the null's 97.5th percentile and ${RPT_U_CELLSF_196} above its mean plus 1.96 standard deviations; on the published draw ${RPT_U_CELLSP_P975} of ${RPT_U_CELLSP_N} lies above the 97.5th percentile (${RPT_U_CELLSP_MODEL}, ${RPT_U_CELLSP_MARKET}: ${RPT_U_CELLSP_AUC} against ${RPT_U_CELLSP_THR}) and ${RPT_U_CELLSP_196} above the mean plus 1.96 standard deviations (Section~\ref{sec:walkforward}). The walk-forward AUCs predate a scaler fix that could not be applied to them (Section~\ref{sec:issues}).''')

# downstream bullet: say which draw
t = replace_once(t, r'it has the lowest median QLIKE (${RPT_DS_BEST_QLIKE}).',
                 r'it has the lowest median QLIKE (${RPT_DS_BEST_QLIKE}). For the three GANs that backtest used a different generation draw from the one behind their metrics (Section~\ref{sec:issues}).')

# pointer bullet, then the new section directly after the Highlights
t = replace_once(t, '\\end{itemize}\n\n%% ─── Why this work matters',
 '\\item \\textbf{Read Section~\\ref{sec:issues} before citing any number.} It lists what is not resolved: a downstream-utility draw mismatch, walk-forward AUCs that predate the scaler fix, published AUCs that cannot be reproduced, no diffusion or language-model arms, and a univariate design.\n\\end{itemize}\n\n%% ─── Changes since the previous version ─────────────────────────────────────\n@@CHANGES@@\n%% ─── Why this work matters')
CHANGES = r"""\section{Changes since the previous version}
\label{sec:changes}
For readers who saw the earlier report. Each statement points to the section where it is developed. \emph{S0} is the earlier ranking scheme, in which \texttt{discriminative\_auc\_dist} and \texttt{discriminative\_auc\_absz} were two of ${RPT_U_W_TEMP_PLUS2} temporal metrics; \emph{S4} is the current one, with ${RPT_U_W_TEMP} temporal metrics and those two descriptive.
\begin{enumerate}[leftmargin=1.6em,itemsep=4pt]
\item \textbf{The headline changed.} ${RPT_U_S4_FIRST} does not beat ${RPT_U_S4_SECOND}; the two are statistically indistinguishable (Section~\ref{sec:ranking}).
\item \textbf{The discriminative AUC columns left the composite (S4).} They are reported but not ranked, because no reference for them works under any calibration tested (Sections~\ref{sec:s4} and~\ref{sec:walkforward}).
\item \textbf{The previously reported AUC null of ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD}$ came from a different experiment} and did not describe this pipeline's null, which sits well above $0.5$ (Section~\ref{sec:walkforward}).
\item \textbf{A metric audit now covers learned-representation metrics}: C-FID and MMD on TS2Vec embeddings (Section~\ref{sec:learned}).
\item \textbf{Fair trivial baselines were added and establish a tiered result} (Section~\ref{sec:tiers}).
\item \textbf{A scaler leak in the discriminative metric was fixed; no conclusion changed} (Section~\ref{sec:scaler}).
\end{enumerate}
Known issues that remain are listed in Section~\ref{sec:issues}.
"""
t = t.replace('@@CHANGES@@', CHANGES)
save(t)
print('group 1 (highlights bullets) applied')
