"""Group 2: families and metrics table under S4, the two design-decision paragraphs that changed, and a new subsection of methods added in this version."""
import sys
sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from patch_lib import *
t = load()

# ---------------- families
t = replace_once(t, r'Seven fidelity and seven temporal metrics are ranked.', r'${RPT_U_W_FID_CAP} fidelity and ${RPT_U_W_TEMP} temporal metrics are ranked.')
t = replace_once(t, r'autocorrelation, long memory, conditional tails and a classifier that sees windows of consecutive days.',
                 r'autocorrelation, long memory and conditional tails. A classifier that sees windows of consecutive days is ordering-sensitive too, but since S4 it is descriptive (Section~\ref{sec:s4}).')
t = replace_between(t, r'Two counts recur and are not the same.', r"Every ``$k$ of $N$'' below says which of these it counts.",
 r'''Two counts recur and are not the same. The \emph{per-market metrics} are the ${RPT_PERM_N_METRICS} columns of each \texttt{<market>\_metrics.csv}: ${RPT_U_W_FID} fidelity, ${RPT_U_W_TEMP} temporal and ${RPT_U_W_DESC_PM} descriptive. Of these, ${RPT_PERM_N_INVARIANT} are permutation-invariant --- the ${RPT_U_W_FID} fidelity metrics plus two descriptive ones, \texttt{skewness\_diff} and \texttt{kurtosis\_diff}. ${RPT_U_W_FID_CAP} of those ${RPT_PERM_N_INVARIANT} are ranked; the two descriptive ones are not. Separately, the pipeline's \texttt{DESCRIPTIVE\_COLS} has ${RPT_U_W_DESC_TOTAL} members: the ${RPT_U_W_DESC_PM} in the per-market files (those two, \texttt{arch\_pvalue\_diff}, \texttt{arch\_stat\_diff}, \texttt{discriminative\_auc\_raw} and, since S4, \texttt{discriminative\_auc\_dist} and \texttt{discriminative\_auc\_absz}) and \texttt{var\_coverage\_error} and \texttt{garch\_persistence\_diff}, which come from the pooled downstream backtest and appear in no per-market file. ''')
t = insert_before(t, r'\subsection{The metrics}' + '\n' + r'\label{sec:metrics}',
 r'''\paragraph{Why the discriminative AUC is not ranked} Until S4 two discriminative-AUC columns were ranked, as the sixth and seventh temporal metrics. Neither has a working reference under any calibration tested, so both are now descriptive; the evidence is in Section~\ref{sec:s4} and the corrected null in Section~\ref{sec:walkforward}.

''')

# ---------------- metrics table: caption, rows moved from Temporal to Descriptive
t = replace_once(t, r'The first ${RPT_PERM_N_METRICS} rows (fidelity, temporal, and five descriptive)', r'The first ${RPT_PERM_N_METRICS} rows (${RPT_U_W_FID} fidelity, ${RPT_U_W_TEMP} temporal and ${RPT_U_W_DESC_PM} descriptive)')
i = t.index(r'discriminative\_auc\_dist & Indistinguish\-ability &'); j = t.index('\n', t.index(r'discriminative\_auc\_absz & Indistinguish\-ability against a null'))
t = t[:i] + t[j + 1:]
t = replace_once(t, r'''& Bollerslev\tcite{36} & ranked \\
''', r'''& Bollerslev\tcite{36} & ranked \\[4pt]
''')
t = replace_once(t, r'''discriminative\_auc\_raw & Classifier AUC & the AUC above, unsigned & Yoon et al.\tcite{4} & descriptive \\[4pt]''',
 r'''discriminative\_auc\_raw & Classifier AUC & AUC of a logistic regression separating real from synthetic ${RPT_U_AUC_WINDOW}-day windows described by mean, standard deviation, mean $|r_t|$ and mean $r_t^2$; features standardised within each training fold; ${RPT_U_AUC_N_SPLITS}-fold unshuffled cross-validation (purged and embargoed as an option, Section~\ref{sec:purged}) & discriminative score of Yoon et al.\tcite{4}; CTBench\tcite{16} & descriptive \\
discriminative\_auc\_dist & Distance of the AUC from chance & $|\mathrm{AUC}-0.5|$ & this work & descriptive (ranked until S4) \\
discriminative\_auc\_absz & Distance from a real-versus-real null & $|\mathrm{AUC}-\mu_0|/\sigma_0$, with $\mu_0$ and $\sigma_0$ from ${RPT_U_LEGACY_NREP} real-versus-real half-splits of the same test series & this work & descriptive (ranked until S4) \\[4pt]''')

# ---------------- design decisions
t = replace_between(t, r'\paragraph{Why walk-forward, not random CV}', r'\paragraph{Why per-market training, not cross-market pooling}',
 r'''\paragraph{Why walk-forward, not random CV} Shuffled cross-validation leaks. Our ${RPT_U_AUC_WINDOW}-day evaluation windows overlap by ${RPT_U_AUC_WINDOW_OVERLAP} observations, so \texttt{shuffle=True} places near-duplicate windows in both train and test (Tashman 2000\tcite{13}; Bergmeir \& Ben\'{\i}tez 2012\tcite{14}). An earlier experiment on independent stationary samples (${RPT_U_EARLIER_N} seeds; its script ${RPT_U_EARLIER_SCRIPT}) reported that shuffled folds move the discriminative-AUC null from ${RPT_U_EARLIER_MEAN} to ${RPT_U_EARLIER_SHUF}; that is a prior measurement of a different experiment and is labelled as one (Section~\ref{sec:walkforward}). The effect reproduces under this pipeline's own estimator on a stationary GARCH-$t$ reference: shuffled folds raise its null mean by ${RPT_U_STAT_SHIFT_RNG_WF} at walk-forward length and ${RPT_U_STAT_SHIFT_RNG_TA} at fixed-split length, across the ${RPT_U_STAT_N_MARKETS} markets. Purged and embargoed folds remove the overlap without shuffling (Section~\ref{sec:purged}).

''')
t = replace_between(t, r'\paragraph{Why discriminative AUC is ranked on $|\text{AUC}-0.5|$}', r'\paragraph{Why downstream utility uses conditional VaR}',
 r'''\paragraph{Why the discriminative AUC was ranked on $|\text{AUC}-0.5|$, and why it no longer is} \textit{Evidence for the original choice:} under an ascending rank on the raw value, an anti-predictive AUC of $0.30$ outranked an indistinguishable $0.50$, because the optimum is $0.5$, not $0$. The rank on $|\text{AUC}-0.5|$ assumed a real-versus-real null of $0.5$, which the corrected null contradicts (Section~\ref{sec:walkforward}); a second ranked column, a $z$-score against a null, depended on how that null was built. Both columns left the ranked set at S4 (Section~\ref{sec:s4}).

''')

# ---------------- new subsection of methods added in this version, after the design decisions
NEW = r'''\subsection{Methods added in this version}
\label{sec:newmethods}
Four methods enter the pipeline or the audit in this version. Each is stated with the measurement that motivated it, and each measurement is an artifact of this repository.

\paragraph{S4: the discriminative AUC leaves the ranked set}\label{sec:s4} The decision was taken on the following grounds, before its effect on the ranking was known. \textit{(i) Neither column has a working reference.} $|\mathrm{AUC}-0.5|$ assumes a real-versus-real null of $0.5$; the null on these series has mean ${RPT_U_NULL_MEAN_RNG_TA} at the fixed-split length (Table~\ref{tab:nulls}). $|\mathrm{AUC}-\mu_0|/\sigma_0$ reads a null that rests on about ${RPT_U_NULL_PAIRS_TA} independent block pairs drawn from ${RPT_U_NULL_YEARS} years of history, and its construction moved first and second place when it was changed. \textit{(ii) The correct estimator does not repair them.} Purged and embargoed cross-validation (below) leaves the anti-predictive draws where they were, so they are regime drift, not leakage (Section~\ref{sec:walkforward}). \textit{(iii) The value is set by the direction of drift, not by generator quality:} the shuffled control's AUC runs ${RPT_U_CTRL_AUC_P} across the ${RPT_U_CTRL_AUC_N} markets under purged folds, and is above the null in one and below it in the others. \textit{(iv) Nothing clears a calibrated threshold} on fresh draws (${RPT_U_CELLSF_P975} of ${RPT_U_CELLS_N} cells, Section~\ref{sec:walkforward}). \textit{(v) Method choice is small next to generation noise:} a fresh generation draw moves ${RPT_U_NOISE_DRAW_TO_DRAW} of ${RPT_U_NOISE_UNITS_TOTAL} within-unit positions, a change of AUC calibration or cross-validation scheme moves ${RPT_U_NOISE_SCHEME_LO} to ${RPT_U_NOISE_SCHEME_HI}, and dropping the two columns (S4) moves ${RPT_U_NOISE_S4} (all measured before the scaler fix; Section~\ref{sec:scaler}). \texttt{TEMPORAL\_COLS} is now ${RPT_U_W_TEMP} metrics, the two AUC columns are in \texttt{DESCRIPTIVE\_COLS}, and the comment beside them in the notebook records these grounds.

\paragraph{Purged and embargoed cross-validation, as an option}\label{sec:purged} \texttt{compute\_discriminative\_score(cv='purged')} keeps the folds contiguous and drops, on both sides of each test chunk, the training windows within one window length (${RPT_U_EMB_WINDOW}) of it, so no observation is shared between a training row and a test row. At ${RPT_U_EMB_FOLDS} folds it drops ${RPT_U_EMB_DROPPED} training rows per fold (${RPT_U_EMB_PCT_TA}\,\% of the training rows at the fixed-split length, ${RPT_U_EMB_PCT_WF}\,\% at walk-forward length); the smallest training set is ${RPT_U_EMB_MIN_TRAIN} rows, so the folds remain large enough to fit. The default is unchanged (contiguous, as published); shuffled folds are refused. Both schemes are computed for every null (Table~\ref{tab:nulls}).

\paragraph{The scaler fix}\label{sec:scaler} The discriminative metric standardised its features on all rows before the folds were cut, so the test fold's mean and scale entered the training features under every scheme, purged included. The scaler is now a step of a scikit-learn \texttt{Pipeline} and is fitted within each training fold. Measured on the same pairs before and after: the null means and standard deviations move by at most ${RPT_U_FIX_MEAN_SHIFT}, individual AUC values by up to ${RPT_U_FIX_MAX_SINGLE} in a few series in strongly drifting stretches (${RPT_U_FIX_PUB_MAX} on the published draw), and no count or range in Section~\ref{sec:walkforward} changes. The cached AUC nulls now carry the scaler convention in their key, so pre-fix caches are recomputed rather than reused.

\paragraph{Fair baselines, and their confinement to the training block}\label{sec:baseline-methods} Two trivial generators with no access to the test block were added as comparison entries; they never enter the composite. \emph{I.i.d.\ historical simulation} resamples returns with replacement from the training block; it is the standard industry baseline for value-at-risk. The \emph{stationary block bootstrap} (Politis \& Romano) resamples geometric-length blocks, circularly, inside the training block, with the mean block length chosen by the Politis--White rule (\texttt{arch} ${RPT_U_BL_ARCH}). Applied to raw returns the rule returns ${RPT_U_PW_R_RANGE} (${RPT_U_PW_FLOORED} market below one observation, floored at one), because raw returns lack linear autocorrelation, so that bootstrap is nearly i.i.d.; applied to $|r|$ it returns ${RPT_U_PW_ABS_RANGE}, the scale of volatility persistence, which is the specification that matters for the temporal metrics and is the one called the block-bootstrap baseline in the text. Table~\ref{tab:pw} gives both. Each baseline is drawn ${RPT_U_BL_DRAWS_PER_UNIT} times per market--seed and scored through the same \texttt{FinancialMetrics} code path as the generators; the generators' and control's rows reproduced from that path equal the stored Phase~B rows exactly. \emph{Confinement:} the generating functions receive the training array, a length and a random generator and nothing else; the validation file is not opened, and from the test file only the row count is taken, from the parquet footer. Of ${RPT_U_BL_N_DRAWS} stored draws, ${RPT_U_BL_OUTSIDE} contain a value that is not in the training block, every index used lies below the training length, and regenerating from the recorded seeds reproduces every draw.

\begin{table}[htbp]
\centering
\caption{The two bootstraps, per market: training length, target length, and the Politis--White mean block length on the returns and on $|r|$, then the lengths used (the returns-based length floored at one observation).}
\label{tab:pw}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Market & Train & Target & PW, $r$ & PW, $|r|$ & Used, $r$ & Used, $|r|$ \\
\midrule
${RPT_U_PW_ROWS}
\bottomrule
\end{tabular}
\end{table}

'''
t = insert_before(t, '%% ─── Architecture comparison', NEW)
save(t)
print('group 2 applied')
