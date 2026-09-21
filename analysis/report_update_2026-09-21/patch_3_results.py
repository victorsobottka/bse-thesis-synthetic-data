"""Group 3: pipeline diagram and prose under S4; the ranking text; and three new results subsections (tiers, temporal metrics on series with no dynamics, learned metrics)."""
import sys
sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from patch_lib import *
t = load()

# ---------------- pipeline diagram (TikZ): temporal 5, descriptive 7
t = replace_once(t, r'\textbf{Temporal}\;(7, ranked)', r'\textbf{Temporal}\;(${RPT_U_N_TEMP}, ranked)')
t = replace_once(t, '''      residual kurtosis: tails beyond GARCH\\\\
      discriminative AUC: $|\\mathrm{AUC}-0.5|$, and $z$ against the null}};''', '''      residual kurtosis: tails beyond GARCH}};''')
t = replace_once(t, r'\textbf{Descriptive}\;(5)', r'\textbf{Descriptive}\;(${RPT_U_N_DESC_PM})')
t = replace_once(t, r'''      raw AUC: optimum $0.5$, not $0$}};''', r'''      discriminative AUC (raw, distance, $z$): no working reference}};''')
t = replace_once(t, r'scores it on the ${RPT_PERM_N_METRICS} per-market metrics ($7+7+5$) and ranks on the fourteen fidelity and temporal ones.',
                    r'scores it on the ${RPT_PERM_N_METRICS} per-market metrics (\(${RPT_U_FAMILY_SUM}\)) and ranks on the ${RPT_U_W_RANKED} fidelity and temporal ones.')
t = replace_once(t, r'seven ranked fidelity metrics, seven ranked temporal metrics and five descriptive ones that are reported but not ranked.',
                    r'${RPT_U_W_FID} ranked fidelity metrics, ${RPT_U_W_TEMP} ranked temporal metrics and ${RPT_U_W_DESC_PM} descriptive ones that are reported but not ranked.')
t = replace_once(t, r'the five generators are ranked on each of the fourteen ranked metrics of Table~\ref{tab:metrics}, rank~1 being closest to the real series. The fidelity rank is the mean over the seven fidelity metrics, the temporal rank the mean over the seven temporal metrics,',
                    r'the five generators are ranked on each of the ${RPT_U_W_RANKED} ranked metrics of Table~\ref{tab:metrics}, rank~1 being closest to the real series. The fidelity rank is the mean over the ${RPT_U_W_FID} fidelity metrics, the temporal rank the mean over the ${RPT_U_W_TEMP} temporal metrics,')
t = replace_once(t, r'${RPT_COMPUTE_BEST_MODEL} has the best mean composite rank (${RPT_BEST_COMPOSITE}); ${RPT_COMPUTE_RUNNERUP_MODEL} trails by ${RPT_COMPUTE_GAP_RUNNERUP} (${RPT_COMPUTE_RUNNERUP_COMPOSITE}).',
                    r'${RPT_COMPUTE_BEST_MODEL} has the best mean composite rank (${RPT_BEST_COMPOSITE}); ${RPT_COMPUTE_RUNNERUP_MODEL} trails by ${RPT_COMPUTE_GAP_RUNNERUP} (${RPT_COMPUTE_RUNNERUP_COMPOSITE}). That margin is not resolved by this benchmark (below).')
t = replace_once(t, r'Table~\ref{tab:ranking} gives, for each generator, its mean rank on the two metric families and on their combination, and how often it wins an evaluation outright.',
                    r'Table~\ref{tab:ranking} gives, for each generator, its mean rank on the two metric families and on their combination, under S4 (${RPT_U_W_TEMP} temporal metrics), and how often it wins an evaluation outright.')

# ---------------- ranking: two new paragraphs after 'What this result tells us'
RANKING = r'''
\paragraph{First place is not resolved} Under S4 on the published draw, ${RPT_U_S4_FIRST} has the lowest mean composite (${RPT_BEST_COMPOSITE}) and ${RPT_U_S4_SECOND} is ${RPT_U_S4_MARGIN_ABS} behind. The 95\,\% bootstrap interval of that margin, resampling the ${RPT_U_S4_N_UNITS} market--seed units (${RPT_U_S4_RESAMPLES} resamples), is ${RPT_U_S4_CI}: it ${RPT_U_S4_EXCLUDES_ZERO} zero, and ${RPT_U_S4_FIRST} is ahead in ${RPT_U_S4_AHEAD_UNITS} of ${RPT_U_S4_N_UNITS} units. The interval spans about ${RPT_U_S4_WIDTH} composite points, so the benchmark cannot resolve a first--second difference smaller than about ${RPT_U_S4_HALFWIDTH}. By seed the margins are ${RPT_U_S4_BY_SEED} (positive: ${RPT_U_S4_FIRST} ahead), ${RPT_U_S4_SEEDS_POSITIVE} of ${RPT_U_S4_N_SEEDS} seeds, and that agreement does not change the interval. No ranking scheme separates the two: across the ${RPT_U_N_SCENARIOS} scenario-and-frame combinations of the scenario matrix, ${RPT_U_N_SCENARIOS_EXCL_ZERO} bootstrap intervals exclude zero. Where the data speak, they say that the two are statistically indistinguishable, which is a result, not a failure to find a winner.

\paragraph{The earlier lead came from one seed, and generation noise dominates method} Under the earlier scheme S0 on the published draw the ${RPT_U_S0_FIRST} lead was ${RPT_U_S0_MARGIN} (interval ${RPT_U_S0_CI}); by seed the margins were ${RPT_U_S0_BY_SEED}, so it came from ${RPT_U_S0_SEEDS_AHEAD} seed of ${RPT_U_S0_N_SEEDS}. Replacing the legacy null by a length-matched one reversed the order (${RPT_U_S1_FIRST} first, by ${RPT_U_S1_MARGIN}). A fresh generation draw moves ${RPT_U_NOISE_DRAW_TO_DRAW} of the ${RPT_U_NOISE_UNITS_TOTAL} within-unit positions of the ranking; any change of AUC calibration or cross-validation scheme moves ${RPT_U_NOISE_SCHEME_LO} to ${RPT_U_NOISE_SCHEME_HI}, and dropping the two AUC columns (S4) moves ${RPT_U_NOISE_S4}. Generation noise dominates methodological choice. These comparisons were run on the values as they stood before the scaler fix (Section~\ref{sec:scaler}), which moves aggregate AUCs by at most ${RPT_U_FIX_MEAN_SHIFT}.
'''
t = insert_before(t, '\n%% ── Compute ──', RANKING)

# ---------------- tiers subsection (new), directly after the compute subsection
TIERS = r'''
%% ── Tiers ──
\subsection{A tiered result against fair baselines}
\label{sec:tiers}
The composite ranks generators against each other. It does not say whether a generator does better than a method that needs no fitting. Two baselines that read only the training block answer that; their construction and their confinement to the training block are in Section~\ref{sec:baseline-methods}. \emph{I.i.d.\ historical simulation} resamples returns with replacement. The \emph{block-bootstrap baseline} is a stationary block bootstrap with the mean block length tuned by the Politis--White rule on $|r|$, about ${RPT_U_PW_ABS_RANGE} observations. The specified bootstrap, tuned on raw returns, degenerated to near-i.i.d.\ because raw returns lack linear autocorrelation (mean block length ${RPT_U_PW_R_USED_RANGE} observations); tuning on $|r|$ is the correct specification, and the returns-tuned version is kept as a comparison entry only.

The baselines are ranked with the five generators and the shuffled control in one pool of ${RPT_U_N_ENTRIES} entries per unit and generation draw, on the same metrics; a unit's value is the mean over its ${RPT_U_SPREAD_DRAWS} fresh draws and the overall value the mean over the ${RPT_U_S4_N_UNITS} units. The generators in the pool are these fresh Phase~B draws, not the published recorded ones (which cannot be re-scored: Section~\ref{sec:issues}). S0 uses the AUC columns as published; S4 does not use them.

\begin{table}[htbp]
\centering
\caption{The ${RPT_U_N_ENTRIES}-entry ranking: fidelity rank, temporal rank (S0 / S4), composite (S0, S4) and position under S0 / S4. The two bootstraps and the shuffled control are comparison entries; none enters the published composite. The pool of a unit contains the five generators, the control and the two bootstraps specified as baselines (i.i.d.\ and the returns-tuned block bootstrap); the $|r|$-tuned baseline is compared with the generators in Table~\ref{tab:mid}.}
\label{tab:tiers}
\begin{tabular}{@{}lccccc@{}}
\toprule
Entry & Fidelity & Temporal & Composite S0 & Composite S4 & Position \\
\midrule
${RPT_U_TIER_ROWS}
\bottomrule
\end{tabular}
\end{table}

\paragraph{Top tier: ${RPT_U_TIER_TOP}} They clear every fair baseline. The best generator, ${RPT_U_BL_BEST_GEN}, is ahead of i.i.d.\ historical simulation by ${RPT_U_IID_MARGIN_S0} composite points under S0 and ${RPT_U_IID_MARGIN_S4} under S4; the 95\,\% bootstrap intervals over units, ${RPT_U_IID_CI_S0} and ${RPT_U_IID_CI_S4}, are ${RPT_U_IID_ALL_EXCL_ZERO} (they also exclude zero when the draws inside each unit are resampled). Both also stay ahead of the block-bootstrap baseline (Table~\ref{tab:mid}).

\paragraph{Middle tier: ${RPT_U_TIER_MIDDLE}} Both are ahead of i.i.d.\ historical simulation but are indistinguishable from the block-bootstrap baseline: ${RPT_U_MID_ALL_CONTAIN_ZERO} under both schemes.

\begin{table}[htbp]
\centering
\caption{The middle tier against the block-bootstrap baseline (tuned on $|r|$): composite of the generator minus composite of the baseline (negative: generator ahead) with its 95\,\% bootstrap interval over the ${RPT_U_S4_N_UNITS} units, under S0 and S4, and against i.i.d.\ historical simulation under S4.}
\label{tab:mid}
\begin{tabular}{@{}llll@{}}
\toprule
Generator & vs block bootstrap, S0 & vs block bootstrap, S4 & vs i.i.d., S4 \\
\midrule
${RPT_U_MID_ROWS}
\bottomrule
\end{tabular}
\end{table}

\paragraph{Bottom tier: ${RPT_U_TIER_BOTTOM}} ${RPT_U_BOTTOM_MODEL} is ${RPT_U_BOTTOM_POSITION} of ${RPT_U_BOTTOM_N}, below i.i.d.\ historical simulation (${RPT_U_IID_POSITION}): against it the margin is ${RPT_U_BOTTOM_VS_IID_S0} under S0 (${RPT_U_BOTTOM_VS_IID_S0_EXCL}) and ${RPT_U_BOTTOM_VS_IID_S4} under S4 (${RPT_U_BOTTOM_VS_IID_S4_EXCL}).

\paragraph{Both baselines are stochastic} Across the ${RPT_U_SPREAD_DRAWS} fresh draws i.i.d.\ historical simulation occupies positions ${RPT_U_SPREAD_IID_POSITIONS} of ${RPT_U_N_ENTRIES} and the returns-tuned block bootstrap ${RPT_U_SPREAD_SB_R_POSITIONS}; in ${RPT_U_SPREAD_IID_AHEAD} and ${RPT_U_SPREAD_SB_R_AHEAD} of the draws respectively is either ahead of every generator.

\paragraph{The shuffled control} If it were allowed to compete it would win: on the published draw under S0 its composite is ${RPT_U_CTRL_COMP} against ${RPT_U_CTRL_BEST_COMP} for ${RPT_U_CTRL_BEST_NAME}, and it is ahead of the best generator in ${RPT_U_CTRL_UNITS_BEATS} of ${RPT_U_S4_N_UNITS} units, with a fidelity rank of ${RPT_U_CTRL_FID_RANK}. The reason is its marginal. The control is the test block's own returns, permuted, so it matches the test marginal exactly: the fidelity family accounts for ${RPT_U_CTRL_FID_SHARE_S0}\,\% (S0) to ${RPT_U_CTRL_FID_SHARE_S4}\,\% (S4) of its composite gap to i.i.d.\ historical simulation, which has no access to the test block. Excluding the control from the ranking is therefore legitimate: its win is the oracle marginal, and a fair no-dynamics baseline does not win.

\paragraph{What this does not show} It does not show that the temporal family penalises missing dynamics: the control's temporal rank, ${RPT_U_CTRL_TEMP_S0} (S0) and ${RPT_U_CTRL_TEMP_S4} (S4), is level with the generators' mean of ${RPT_U_GEN_TEMP_S0} and ${RPT_U_GEN_TEMP_S4}. Which temporal metrics do and do not, and why, is the next subsection.

'''
t = insert_before(t, '\n%% ── Figures ──', TIERS)

# ---------------- temporal decomposition and learned metrics: after the shuffled-control audit, before walk-forward
DECOMP = r'''
\subsection{The temporal metrics on a series with no dynamics}
\label{sec:temporal-decomp}
Table~\ref{tab:decomp} ranks the shuffled control and the two bootstraps on each temporal metric among the ${RPT_U_N_ENTRIES} entries, beside the generators' mean rank and the raw values. The question is which metrics fail to penalise a series with no dynamics.

\begin{table}[htbp]
\centering
\caption{Mean rank among the ${RPT_U_N_ENTRIES} entries (1: closest to real) of the shuffled control, i.i.d.\ historical simulation and the returns-tuned block bootstrap on each temporal metric, the mean rank of the five generators, and the raw metric values (mean over ${RPT_U_S4_N_UNITS} units and ${RPT_U_SPREAD_DRAWS} draws): control, i.i.d., and the range over the generators.}
\label{tab:decomp}
{\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
Metric & Ctrl & i.i.d. & Block & Gens & Ctrl raw & i.i.d. raw & Gens raw \\
\midrule
${RPT_U_TD_ROWS}
\bottomrule
\end{tabular}}
\end{table}

\paragraph{\texttt{hurst\_diff} penalises missing dynamics strongly.} The control ranks ${RPT_U_TD_HURST_CTRL} and i.i.d.\ historical simulation ${RPT_U_TD_HURST_IID}, against ${RPT_U_TD_HURST_GENS} for the four generators other than the one that fails it.

\paragraph{\texttt{acf\_absolute\_mae} penalises it weakly.} A flat autocorrelation function costs ${RPT_U_TD_ABS_CTRL} (control) and ${RPT_U_TD_ABS_IID} (i.i.d.); the best generator, ${RPT_U_TD_ABS_BEST_NAME}, scores ${RPT_U_TD_ABS_BEST}. The gap of ${RPT_U_MECH_GAP} is smaller than that generator's own draw-to-draw standard deviation within a unit (${RPT_U_TD_ABS_DRAW_SD_BEST}; GARCH ${RPT_U_TD_ABS_DRAW_SD_GARCH}), so one draw of about five hundred observations cannot resolve it.

\paragraph{\texttt{acf\_returns\_mae} sits at the noise floor for every entry.} Real and shuffled autocorrelations of returns are both near zero, so their mean absolute difference is sampling noise: ${RPT_U_TD_RET_CTRL} for the control, with every entry between ${RPT_U_TD_RET_ALL}. The metric has no resolution, and it ranks i.i.d.\ historical simulation (${RPT_U_TD_RET_RANK_IID}) ahead of the control (${RPT_U_TD_RET_RANK_CTRL}).

\paragraph{\texttt{acf\_squared\_mae} is won by the control and by i.i.d.\ historical simulation.} They score ${RPT_U_TD_SQ_CTRL} and ${RPT_U_TD_SQ_IID}, better than every generator (${RPT_U_TD_SQ_GENS}). A flat autocorrelation function costs the mean magnitude of the real one, and the generators' volatility clustering is wrong by more than that, in different ways. Measured on the autocorrelation of $|r|$ over lags 1--50 (real: lag 1 ${RPT_U_MECH_REAL_LAG1}, lag 50 ${RPT_U_MECH_REAL_LAG50}, mean ${RPT_U_MECH_REAL_MEAN}), QuantGAN over-produces clustering and never lets it decay (lag 50 ${RPT_U_MECH_Q_LAG50}, mean ${RPT_U_MECH_Q_MEAN}); GARCH and GJR-GARCH are too weak at lag 1 (${RPT_U_MECH_GARCH_LAG1} and ${RPT_U_MECH_GJR_LAG1}); TimeGAN is right on average but erratic from draw to draw, worse than the control in ${RPT_U_MECH_T_WORSE_CTRL}\,\% of draws, while QuantGAN beats it in ${RPT_U_MECH_Q_BEATS_CTRL}\,\%. That is a generator flaw the metric correctly detects, not a broken metric; the profile was measured for $|r|$, and the squared-return metric responds to the same errors but was not profiled separately.

\paragraph{\texttt{resid\_kurtosis\_diff} penalises the wrong entry.} It penalises i.i.d.\ historical simulation (rank ${RPT_U_TD_RESID_IID}) but not the control (${RPT_U_TD_RESID_CTRL}, better than every generator, ${RPT_U_TD_RESID_GENS}), because the control has the test block's exact marginal: the temporal family is not free of the oracle either.

'''
LEARNED = r'''
\subsection{Learned-representation metrics: C-FID and MMD on TS2Vec}
\label{sec:learned}
The audit was extended to metrics computed on a learned embedding, the setting of the TSGBench benchmark: a TS2Vec encoder trained per market on the training windows, real and synthetic test windows embedded, and a distance between the two clouds.

\paragraph{C-FID is not estimable here.} Stride-1 windows share ${RPT_U_LM_WINDOW_SHARED} of ${RPT_U_LM_WINDOW_LEN} observations, so the ${RPT_U_LM_WINDOWS} test windows carry an effective sample of only ${RPT_U_LM_NEFF_RANGE} independent windows (from the autocorrelation of the leading embedding component), and the embedding covariance has a participation ratio of ${RPT_U_LM_PR_RANGE}. A Fr\'echet distance needs the full covariance of the embedding; it is not estimable from that. This is a property of the windowing, not of the encoder, and no C-FID value is reported.

\paragraph{MMD ranks the shuffled control lowest.} An unbiased MMD$^2$ with a Gaussian kernel at the median-heuristic bandwidth, against an exhaustive adjacent-sliding null, ranks the shuffled control as the closest of the six series to the real block under all ${RPT_U_LM_N_CFG} encoder configurations (mean MMD$^2$ ${RPT_U_LM_CTRL_MMD}; the next-lowest series ${RPT_U_LM_NEXT_MMD}): the three are the main configuration, a wider embedding, and TSGBench's own literal training recipe. It is the lowest series in ${RPT_U_LM_CTRL_LOWEST_UNITS} units respectively, and it is detected in ${RPT_U_LM_CTRL_DETECT_UNITS} of ${RPT_U_LM_UNITS} units (${RPT_U_LM_CTRL_DETECT_DRAWS}\,\% of its draws reach $p\le0.05$).

\paragraph{Properly calibrated, the test has almost no power.} Under the scale-matched null (the kernel bandwidth fixed at the observed one for null and observed alike) the detection rate of the generator that fails most, TimeGAN, falls from ${RPT_U_LM_TG_RATE_MAIN}\,\% to ${RPT_U_LM_TG_RATE_MATCHED}\,\%.

\paragraph{The embedding carries weak ordering information; the failure belongs to the statistic.} A logistic-regression probe fitted on validation-block windows separates real from permuted test-block windows at a mean AUC of ${RPT_U_LM_PROBE_EMB} (${RPT_U_LM_PROBE_EMB_RNG} across the ${RPT_U_LM_PROBE_MARKETS} markets), comparable to ${RPT_U_LM_PROBE_ACF} from a probe on three hand-crafted autocorrelation features (${RPT_U_LM_PROBE_ACF_RNG}); ${RPT_U_LM_PROBE_MARKETS} markets cannot support a significance claim in either direction. The hypothesis that the embedding is permutation-invariant by construction was tested and is refuted by that probe. MMD$^2$ between the real block and its permutations, ${RPT_U_LM_PERM_MMD_RNG} on average, lies below the nulls' medians (${RPT_U_LM_NULL_MEDIAN_RNG}), with $p$-values of ${RPT_U_LM_MMD_PERM_P_RNG} and ${RPT_U_LM_MMD_PERM_DRAW_SHARE}\,\% of permuted draws at $p\le0.05$. The statistic, at this effective sample size, does not use the ordering information that is there.

'''
t = insert_before(t, '%% ── Walk-forward ──', DECOMP + LEARNED)
save(t)
print('group 3 applied')
