"""Group 4: the walk-forward discriminability subsection, rewritten around the corrected null (regenerated auc_null.json, stationary reference, purged folds)."""
import sys
sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from patch_lib import *
t = load()

NEW = r'''%% ── Walk-forward ──
\subsection{Discriminability and the corrected null}
\label{sec:walkforward}

\paragraph{The earlier null was a different experiment} The previous version of this report quoted a null of ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD} for the discriminative AUC and described it as measured over real-versus-real half-splits. It was not. The figure comes from an earlier experiment on independent stationary samples (${RPT_U_EARLIER_N} seeds, first recorded on ${RPT_U_EARLIER_FIRST}); its script ${RPT_U_EARLIER_SCRIPT}, it is not the output of the pipeline's own estimator, and no version of the pipeline ever ran ${RPT_U_EARLIER_N} draws of it. It does not describe this pipeline's null and is not used as one. The paired figure from the same experiment, the shift of ${RPT_U_EARLIER_MEAN} to ${RPT_U_EARLIER_SHUF} when the folds are shuffled, stays as a prior measurement of that experiment, correctly labelled; it reproduces in direction under this pipeline's estimator on a stationary reference (below).

\paragraph{The corrected null} The calibration is the fixed-length adjacent sliding null: for every start $s$ of a market's full ${RPT_U_NULL_YEARS}-year series, the AUC between the block $[s,s+L)$ and the block $[s+L,s+2L)$, at both evaluation lengths (the fixed-split test length $L$ of ${RPT_U_NULL_L_TA} observations, and the walk-forward test block, ${RPT_U_NULL_L_WF}), enumerated exhaustively, with the metric's own estimator. \emph{Limitation:} the population spans twenty years while the observed comparison is the test period, and the positions overlap almost completely, so the ${RPT_U_NULL_POSITIONS_TA} positions at the fixed-split length carry about ${RPT_U_NULL_PAIRS_TA} independent block pairs, and the ${RPT_U_NULL_POSITIONS_WF} at walk-forward length about ${RPT_U_NULL_PAIRS_WF}. Nothing was done to engineer around this, and a null built from test-period blocks alone cannot be formed: at the evaluation length the test block holds no room for two adjacent blocks.

\begin{table}[htbp]
\centering
\caption{The corrected AUC null at the fixed-split length, per market: block length $L$, independent block pairs, and, under contiguous (published) and purged cross-validation, the mean, standard deviation, 97.5th percentile and the share of draws below $0.5$. Features are standardised within each training fold.}
\label{tab:nulls}
{\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & & & \multicolumn{4}{c}{contiguous} & \multicolumn{3}{c}{purged} \\
\cmidrule(lr){4-7}\cmidrule(l){8-10}
Market & $L$ & Pairs & Mean & Sd & p97.5 & $<0.5$ & Mean & p97.5 & $<0.5$ \\
\midrule
${RPT_U_NULL_ROWS_TA}
\bottomrule
\end{tabular}}
\end{table}

\begin{table}[htbp]
\centering
\caption{The same at the walk-forward length ($L=\lfloor n/(K+1)\rfloor$ with $K=${RPT_N_FOLDS}$ folds).}
\label{tab:nullswf}
{\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}
\toprule
 & & & \multicolumn{4}{c}{contiguous} & \multicolumn{3}{c}{purged} \\
\cmidrule(lr){4-7}\cmidrule(l){8-10}
Market & $L$ & Pairs & Mean & Sd & p97.5 & $<0.5$ & Mean & p97.5 & $<0.5$ \\
\midrule
${RPT_U_NULL_ROWS_WF}
\bottomrule
\end{tabular}}
\end{table}

\paragraph{The per-market nulls sit well above $0.5$} Their means are ${RPT_U_NULL_MEAN_RNG_TA} at the fixed-split length and ${RPT_U_NULL_MEAN_RNG_WF} at walk-forward length (${RPT_U_NULL_MEAN_RNG_PURGED_TA} and ${RPT_U_NULL_MEAN_RNG_PURGED_WF} under purged folds), with standard deviations of ${RPT_U_NULL_SD_RNG_TA} and 97.5th percentiles of ${RPT_U_NULL_P975_RNG_TA} at the fixed-split length; the share of draws below $0.5$ is ${RPT_U_NULL_BELOW_RNG_TA}\,\%, depending on the market. An AUC of $0.6$ is therefore unremarkable for real-versus-real data. (The pipeline's earlier half-split null, ${RPT_U_HALFSPLIT_B} half-splits of the test block, gave means of ${RPT_U_HALFSPLIT_MEAN_RNG}; it is kept in the artifacts and is not used here.)

\paragraph{Even under ideal conditions the null is not $0.5$} Independent paths of one stationary GARCH-$t$ process, fitted per market and scored with the same estimator (${RPT_U_STAT_PAIRS} pairs per length; Table~\ref{tab:stat}), give null means of ${RPT_U_STAT_MEAN_RNG_WF} at walk-forward length and ${RPT_U_STAT_MEAN_RNG_TA} at the fixed-split length, standard deviations of ${RPT_U_STAT_SD_RNG_WF}, and 97.5th percentiles of ${RPT_U_STAT_P975_RNG_WF}. The market with the least persistent fit, ${RPT_U_STAT_LOW_MARKET} (persistence ${RPT_U_STAT_LOW_PERSIST}), gives mean ${RPT_U_STAT_LOW_MEAN}, standard deviation ${RPT_U_STAT_LOW_SD} and 97.5th percentile ${RPT_U_STAT_LOW_P975}. Shuffled folds raise the reference null's mean by ${RPT_U_STAT_SHIFT_RNG_WF} at walk-forward length (${RPT_U_STAT_SHIFT_RNG_TA} at the fixed-split length) --- the leak through overlapping windows --- while purged folds leave it at ${RPT_U_STAT_PURGED_RNG_WF}.

\begin{table}[htbp]
\centering
\caption{Stationary GARCH-$t$ reference null at walk-forward length, per market: persistence of the fit, contiguous mean, standard deviation and 97.5th percentile, and the purged and shuffled-fold means. Shuffled folds are a diagnostic only.}
\label{tab:stat}
{\footnotesize
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Market & Persistence & Mean & Sd & p97.5 & Purged & Shuffled \\
\midrule
${RPT_U_STAT_ROWS_WF}
\bottomrule
\end{tabular}}
\end{table}

\paragraph{Under contiguous folds, drift makes the classifier measure regime change} With the folds cut in time and the volatility level drifting, a classifier fitted on the training chunks meets a test chunk from a different regime, and its AUC can fall below $0.5$. Purged and embargoed folds (Section~\ref{sec:purged}) remove the leakage through overlapping windows without shuffling, and the result shows the drift is real rather than leakage. For NIFTY50, the draws of the fixed-split null whose contiguous AUC is below $0.5$ (${RPT_U_NIF_N} of ${RPT_U_NIF_OF}, ${RPT_U_NIF_SHARE}\,\%) average ${RPT_U_NIF_C} under contiguous folds and ${RPT_U_NIF_P} under purged folds, against ${RPT_U_NIF_S} under shuffled folds (before the scaler fix: ${RPT_U_NIF_C_PRE}, ${RPT_U_NIF_P_PRE} and ${RPT_U_NIF_S_PRE}). Only ${RPT_U_NIF_P_ABOVE}\,\% of them cross $0.5$ under purged folds, against ${RPT_U_NIF_S_ABOVE}\,\% under shuffled folds; the share of the whole null below $0.5$ goes from ${RPT_U_NIF_NULL_BELOW_C}\,\% to ${RPT_U_NIF_NULL_BELOW_P}\,\% when the folds are purged. The shuffled value is what leakage produces, not what a corrected estimator recovers. The same drift sets the shuffled control's AUC, which runs ${RPT_U_CTRL_AUC_C} across the ${RPT_U_CTRL_AUC_N} markets under contiguous folds and ${RPT_U_CTRL_AUC_P} under purged folds: a series with no dynamics, which a working discriminator would put far above the null, lands below it in ${RPT_U_CTRL_BELOW_N} of the ${RPT_U_CTRL_AUC_N} markets.

\paragraph{No cell clears a calibrated threshold on fresh draws; on the published draw one clears the 97.5th percentile} Against the corrected null at the fixed-split length, with each market--model cell the mean over three seeds and twenty fresh generation draws, ${RPT_U_CELLSF_P975} of ${RPT_U_CELLS_N} cells lie above the 97.5th percentile and ${RPT_U_CELLSF_196} above the mean plus 1.96 standard deviations (${RPT_U_CELLSFP_P975} and ${RPT_U_CELLSFP_196} under purged folds). On the published draw, with the cell the mean over the three seeds' single recorded draws, ${RPT_U_CELLSP_P975} of ${RPT_U_CELLSP_N} lies above the 97.5th percentile --- ${RPT_U_CELLSP_MODEL} in ${RPT_U_CELLSP_MARKET}, ${RPT_U_CELLSP_AUC} against ${RPT_U_CELLSP_THR} --- and ${RPT_U_CELLSP_196} above the mean plus 1.96 standard deviations. The published-draw values are the ones computed before the scaler fix (Section~\ref{sec:scaler}), which moves aggregate AUCs by at most ${RPT_U_FIX_MEAN_SHIFT}.

\begin{table}[htbp]
\centering
\caption{Discriminative AUC over every walk-forward fold of every evaluation, against the corrected walk-forward null of each market. $z$ is the mean over markets of the distance of the model's market-mean AUC from that market's null mean, in null standard deviations: a distance, not a test statistic. The last columns count market--model cells (of ${RPT_U_NULL_N_MARKETS} markets) above the null's 97.5th percentile and above its mean plus 1.96 standard deviations. \textbf{These AUCs were computed before the scaler fix and cannot be refreshed: the ${RPT_U_WF_FOLD_MODELS} fold models were never saved.} The raw AUC is shown although no metric ranks it; its optimum is $0.5$, not $0$.}
\label{tab:auc}
\begin{tabular}{@{}lcccccr@{}}
\toprule
Model & AUC (mean $\pm$ sd) & $|\mathrm{AUC}-0.5|$ & $z$ vs null & $>$ p97.5 & $>$ mean+1.96 sd & Folds \\
\midrule
${RPT_U_WFAUC_ROWS}
\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Mean discriminative AUC by walk-forward fold, over all models and markets (values computed before the scaler fix). Every fold starts at the same point, so the training window and the calendar period it covers grow together: fold ${RPT_FOLD_FIRST_FOLD} covers ${RPT_CAL_START_YEAR} to ${RPT_CAL_FIRST_END} and fold ${RPT_FOLD_LAST_FOLD} covers ${RPT_CAL_START_YEAR} to ${RPT_CAL_LAST_END}, so length and period are confounded.}
\label{tab:folds}
\begin{tabular}{@{}lccc@{}}
\toprule
Fold & Training length & Mean AUC & sd \\
\midrule
${RPT_FOLD_EFFECT_ROWS}
\bottomrule
\end{tabular}
\end{table}

\paragraph{What they show} Tables~\ref{tab:auc} and~\ref{tab:folds} show whether a classifier can tell each model's output from real data out of sample, and how that changes as the training window grows.

\paragraph{How they are computed} In each of the ${RPT_N_FOLDS} walk-forward folds (Section~\ref{sec:pipeline}) a freshly trained model generates a series the length of the fold's test block. A logistic regression then tries to separate real from synthetic ${RPT_U_AUC_WINDOW}-day windows described by their mean, standard deviation, mean absolute return and mean squared return, scored by ${RPT_U_AUC_N_SPLITS}-fold cross-validation without shuffling. An AUC of $0.5$ means the two are indistinguishable, but the null is not $0.5$ (above).

\paragraph{What this result tells us} \emph{Significance.} ${RPT_U_WF_N_MODELS_ABOVE_NULL_MEAN} of ${RPT_U_WF_N_MODELS} models have a pooled AUC above their markets' null means, as they would if any generator were slightly distinguishable, but ${RPT_U_WF_CELLS_P975} of ${RPT_U_WF_CELLS_N} market--model cells lies above the null's 97.5th percentile (${RPT_U_WF_HIT_MODEL} in ${RPT_U_WF_HIT_MARKET}, ${RPT_U_WF_HIT_AUC} against ${RPT_U_WF_HIT_THR}) and ${RPT_U_WF_CELLS_196} above the mean plus 1.96 standard deviations. No $p$-value is tabulated: the null has ${RPT_U_NULL_POSITIONS_WF} positions but about ${RPT_U_NULL_PAIRS_WF} independent block pairs, so the resolution the data support is set by the latter, and an interval or threshold is a fair summary while a small $p$-value would not be. That the AUC of a generator lies above the real-versus-real null is unremarkable, because the null is itself high.

\emph{Trend and its confound.} AUC falls as the training window grows, from ${RPT_FOLD_FIRST_AUC} at fold ${RPT_FOLD_FIRST_FOLD} (training on ${RPT_FOLD_FIRST_LEN} observations) to ${RPT_FOLD_LAST_AUC} at fold ${RPT_FOLD_LAST_FOLD} (${RPT_FOLD_LAST_LEN}), but that is an aggregate over models, markets and seeds, and the decline is ${RPT_FOLD_MONOTONIC}: fold(s) ${RPT_FOLD_RISE_FOLDS} rise above the fold before. Below the aggregate it is no more monotone: of the ${RPT_FOLD_N_SERIES} individual (model, market, seed) series, ${RPT_FOLD_N_SERIES_MONO} fall at every step, and of the ${RPT_FOLD_N_MODEL_CURVES} models' fold-mean curves, ${RPT_FOLD_N_MODELS_MONO} do. Every fold starts at the same point, so training length and historical period grow together: fold ${RPT_FOLD_FIRST_FOLD} covers ${RPT_CAL_START_YEAR} to ${RPT_CAL_FIRST_END} and fold ${RPT_FOLD_LAST_FOLD} covers ${RPT_CAL_START_YEAR} to ${RPT_CAL_LAST_END} (the ends differ by market), so this design cannot separate ``more training data helps'' from ``the later period is easier to imitate''.

'''
t = replace_between(t, '%% ── Walk-forward ──', '%% ── Tail index ──', NEW)
save(t)
print('group 4 applied')
