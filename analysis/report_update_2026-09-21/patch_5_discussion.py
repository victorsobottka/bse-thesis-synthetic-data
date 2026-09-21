"""Group 5: circularity count, Discussion, the Known-issues section, Limitations, Next steps, glossary and appendices."""
import sys
sys.path.insert(0, __import__('pathlib').Path(__file__).parent.as_posix())
from patch_lib import *
t = load()

# ---------------- circularity: the temporal family now has ${RPT_U_W_TEMP} members
t = replace_once(t, r'\texttt{resid\_kurtosis\_diff} is one of seven temporal metrics, and the temporal result does not rest on it.',
                    r'\texttt{resid\_kurtosis\_diff} is one of ${RPT_U_W_TEMP} temporal metrics, and the temporal result does not rest on it.')

# ---------------- Discussion
t = replace_once(t, r'\subsection{A small model ranks first, and why that is the interesting part}',
                    r'\subsection{A small model is at the top, and this benchmark cannot tell it from a large one}')
t = replace_once(t, r'''${RPT_COMPUTE_BEST_MODEL} finishes first overall on the equal-weighted composite. The per-cell composite count goes the other way (gradient ${RPT_WIN_COMPOSITE_GRAD} of ${RPT_WIN_N_CELLS}), and ${RPT_COMPUTE_RUNNERUP_MODEL} trails by only ${RPT_COMPUTE_GAP_RUNNERUP}: the econometric edge is in the mean rank and in the dynamics, not a sweep.''',
                    r'''${RPT_COMPUTE_BEST_MODEL} has the lowest mean composite. The per-cell composite count goes the other way (gradient ${RPT_WIN_COMPOSITE_GRAD} of ${RPT_WIN_N_CELLS}), and ${RPT_COMPUTE_RUNNERUP_MODEL} trails by only ${RPT_COMPUTE_GAP_RUNNERUP}, a margin whose 95\,\% bootstrap interval is ${RPT_U_S4_CI} (Section~\ref{sec:ranking}): the two are statistically indistinguishable, so the claim this report can support is that a ${RPT_U_S4_FIRST_PARAMS}-parameter model fitted in about ${RPT_U_S4_FIRST_SECONDS}\,s is as good as a ${RPT_U_S4_SECOND_PARAMS}-parameter one, and that both clear every fair baseline (Section~\ref{sec:tiers}); not that the small one wins.''')
t = replace_once(t, r'''The general form of the caveat is the part we would ask others to adopt.''',
 r'''The audit also has limits, and the fair baselines show where. A control that is the test block's own returns wins the composite because of its marginal, not because the temporal family fails (Section~\ref{sec:tiers}); but the temporal family on its own ranks a series with no dynamics level with the generators' mean, because only one of its ${RPT_U_W_TEMP} metrics penalises the absence of dynamics strongly (Section~\ref{sec:temporal-decomp}). Metrics computed on a learned embedding did not help: the statistic at this sample size does not use the ordering information the embedding carries (Section~\ref{sec:learned}).

The general form of the caveat is the part we would ask others to adopt.''')
t = replace_once(t, r"\paragraph{Whether the six-parameter win holds across asset classes} GJR-GARCH's leverage term",
                    r"\paragraph{Whether the econometric result holds across asset classes} The leverage term of GJR-GARCH")
t = replace_once(t, r'''a stationary block bootstrap (keeps short-range dependence, destroys long memory),''',
                    r'''a stationary block bootstrap (keeps short-range dependence, destroys long memory; a first version, used as a fair baseline rather than a control, is in Section~\ref{sec:tiers}),''')

# ---------------- Known issues: a marked section before the Limitations
ISSUES = r'''%% ─── Known issues ────────────────────────────────────────────────────────────
\section{Known issues and what is not resolved}
\label{sec:issues}
\begin{tcolorbox}[sharp corners,boxrule=0.5pt,colframe=Rule,colback=white,left=8pt,right=8pt,top=6pt,bottom=6pt]
This section lists what is not resolved, so that co-authors know before citing a number from this report.
\end{tcolorbox}
\begin{enumerate}[leftmargin=1.6em,itemsep=6pt]
\item \textbf{Downstream utility and the metrics use different generation draws for the GANs.} For ${RPT_U_DSM_N_GANS} of the generators (${RPT_U_DSM_GANS}) the series saved for the downstream backtest is a second call of \texttt{generate()}, not the series their metrics were computed on: ${RPT_U_DSM_GAN_AGREE} of ${RPT_U_DSM_N} market--seed units agree for each of them, against ${RPT_U_DSM_EXACT_N} of ${RPT_U_DSM_N} for ${RPT_U_DSM_EXACT}, which re-seed on every call. Fidelity and VaR/QLIKE therefore describe different samples for the GANs. The published values are unchanged. The correction for the next production run is to feed the metrics draw to the downstream backtest; that change is not yet in the code.
\item \textbf{The walk-forward discriminative AUC predates the scaler fix and cannot be refreshed} without retraining the ${RPT_U_WF_FOLD_MODELS} fold models (${RPT_U_WF_FILES} model--market--seed files of ${RPT_U_WF_FOLDS} folds each), whose weights were never saved. The fix moved aggregate AUCs (null means and standard deviations) by at most ${RPT_U_FIX_MEAN_SHIFT}, so the conclusions of Section~\ref{sec:walkforward} are expected to hold, but the values in Tables~\ref{tab:auc} and~\ref{tab:folds} are pre-fix.
\item \textbf{The published discriminative AUC of QuantGAN and CNN-WGAN-GP cannot be reproduced under the fix,} because their CUDA random-number state was not saved. The published series of ${RPT_U_REPRO_MODELS} and of the control can be recomputed (to within ${RPT_U_REPRO_MAX} for the pre-fix value), so the published AUC columns cannot be updated consistently across all five models; they are flagged as pre-fix in every metrics file. Under S4 this does not touch the composite. For the same reason the recorded GAN series behind the published metrics were never stored, so the scenario matrix and the baseline comparison score fresh generation draws of the generators, not the published ones.
\item \textbf{Diffusion and language-model generators have not been run.} The roster is three GANs and two econometric baselines.
\item \textbf{The benchmark is univariate.} The ${RPT_U_N_MARKETS} markets are evaluated independently; cross-market dependence is not tested, and per-market training rules it out by design (Section~\ref{sec:decisions}).
\item \textbf{Some paragraphs written before this version still carry figures typed from earlier measurements} (the ARCH-LM accuracy, the QLIKE small-sample check, the budget-parity timings, the min-max comparison). They are marked as prior measurements where used and are not read from this run's artifacts.
\end{enumerate}

'''
t = insert_before(t, '%% ─── Limitations ──', ISSUES)

# ---------------- Limitations
t = replace_once(t, r'''\textbf{No diffusion baseline.} Takahashi \& Mizuno (2025)\tcite{29} report diffusion-based generators outperforming GANs on several stylized-fact metrics. This benchmark does not include one. Reviewers will ask; the answer is not yet.''',
                    r'''\textbf{No diffusion or language-model generator.} Takahashi \& Mizuno (2025)\tcite{29} report diffusion-based generators outperforming GANs on several stylized-fact metrics. This benchmark does not include one, nor a language-model arm. Reviewers will ask; the answer is not yet.''')
t = replace_once(t, r'''The empirical AUC null, the repeated-run nondeterminism figures, the smoke-budget ranking and the Monte Carlo accuracy of the ARCH-LM variants were measured before this run and are recorded in the repository's documentation, not in this run's artifacts. They are marked as prior where they are used.''',
                    r'''The repeated-run nondeterminism figures, the smoke-budget ranking and the Monte Carlo accuracy of the ARCH-LM variants were measured before this run and are recorded in the repository's documentation, not in this run's artifacts. They are marked as prior where they are used. The AUC null is no longer among them: it is computed from this run's artifacts (Section~\ref{sec:walkforward}), and the earlier ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD} figure appears only as a labelled quotation of a different experiment.''')

# ---------------- Next steps
t = replace_once(t, r'''\item \textbf{Adversarial-control battery.} Add the block-bootstrap, phase-randomised and residual-shuffle controls beside the permutation control and report each metric's separation from each.''',
                    r'''\item \textbf{Adversarial-control battery.} Add the phase-randomised and residual-shuffle controls beside the permutation control and the two fair bootstraps (Section~\ref{sec:tiers}), and report each metric's separation from each.
\item \textbf{Close the known issues} (Section~\ref{sec:issues}): store the metrics draw and feed it to the downstream backtest, and retrain the walk-forward fold models with the fixed scaler and with their weights saved.''')
t = replace_once(t, r'\item \textbf{A diffusion baseline}\tcite{29}, scored by the same pipeline, since diffusion generators are reported to outperform GANs on several stylised-fact metrics and this benchmark has none.',
                    r'\item \textbf{A diffusion baseline}\tcite{29} and a language-model arm, scored by the same pipeline, since diffusion generators are reported to outperform GANs on several stylised-fact metrics and this benchmark has none.')

# ---------------- glossary and appendices
t = replace_once(t, r'The ranked quantity is the distance from 0.5.', r'It is reported but not ranked: the null is not $0.5$ (Section~\ref{sec:walkforward}), and the two columns that were ranked until S4 have no working reference (Section~\ref{sec:s4}).')
t = replace_once(t, r'''Discriminative AUC direction &
Ranking ascending on raw AUC put an anti-predictive classifier (0.30) above an indistinguishable one (0.50). &
Ranked on $|\mathrm{AUC}-0.5|$, with a $z$-score against the empirical null. &
Prior: repository documentation. \\''',
r'''Discriminative AUC direction and calibration &
Ranking ascending on raw AUC put an anti-predictive classifier (0.30) above an indistinguishable one (0.50); the replacement ranks assumed a null of $0.5$ or a null of ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD} that came from a different experiment (next rows). &
Ranked on $|\mathrm{AUC}-0.5|$ and a $z$-score, then, at S4, not ranked (Section~\ref{sec:s4}). &
Prior: repository documentation. Computed: corrected null (Section~\ref{sec:walkforward}). \\
\addlinespace
AUC null quoted from another experiment &
The null ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD} was described as ${RPT_U_EARLIER_N} real-versus-real half-splits; it came from an experiment on independent stationary samples whose script ${RPT_U_EARLIER_SCRIPT}. The pipeline's own null is ${RPT_U_NULL_MEAN_RNG_TA} at the fixed-split length. &
Relabelled; the null regenerated as the fixed-length adjacent sliding null at both lengths, with a stationary GARCH-$t$ reference (Section~\ref{sec:walkforward}). &
Computed: \texttt{auc\_null.json}, stationary reference. \\
\addlinespace
Scaler fitted before the folds were cut (\emph{after the run}) &
The discriminative metric standardised its features on all rows before cross-validation, so test-fold statistics entered training under every scheme. Null means and standard deviations move by at most ${RPT_U_FIX_MEAN_SHIFT}; single AUC values by up to ${RPT_U_FIX_MAX_SINGLE}. &
Scaler inside a \texttt{Pipeline}, fitted within each training fold; AUC-null caches keyed on the convention; published AUCs flagged pre-fix, not overwritten (Section~\ref{sec:scaler}). &
Computed: pre- and post-fix nulls and cells. \\''')
t = replace_once(t, r'''{Ascending rank on raw AUC put $0.30$ above $0.50$; the optimum is $0.5$, not $0$. The null is not exactly $0.5$ either --- measured at $0.506\pm0.084$ over 15 real-vs-real half-splits, so an observed $0.62$ is consistent with a perfect generator. \textbf{Fix:} rank on $|\text{AUC}-0.5|$, report a $z$-score against the empirical null, and do not shuffle the CV folds (shuffling moves the null to $0.584$, because 20-day windows overlap by 19 observations).}''',
r'''{Ascending rank on raw AUC put $0.30$ above $0.50$; the optimum is $0.5$, not $0$. The null is not $0.5$ either: it is ${RPT_U_NULL_MEAN_RNG_TA} on the real series at the fixed-split length, so an observed $0.62$ is consistent with a perfect generator; the earlier figure of ${RPT_U_EARLIER_MEAN}\,$\pm$\,${RPT_U_EARLIER_SD} came from a different experiment. \textbf{Fix:} the column no longer counts toward the rank (S4), the null is the corrected sliding null, the folds are not shuffled (${RPT_U_AUC_WINDOW}-day windows overlap by ${RPT_U_AUC_WINDOW_OVERLAP} observations), and the scaler is fitted within each training fold.}''')
t = replace_once(t, r'discriminative\_auc\_dist, discriminative\_auc\_absz $\cdot$ \textsc{Temporal} \\', r'discriminative\_auc\_dist, discriminative\_auc\_absz $\cdot$ \textsc{Descriptive} (ranked until S4) \\')
t = replace_once(t, r"Every change to data handling, training or evaluation below was made before the run reported here; the report's numbers are from the corrected pipeline.",
                    r'Every change to data handling, training or evaluation below was made before the run reported here, except the one marked \emph{after the run}, which affects only the descriptive AUC values (Section~\ref{sec:scaler}).')
save(t)
print('group 5 applied')
