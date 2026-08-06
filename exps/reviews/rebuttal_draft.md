# Runs

\begin{table}[t]
\centering
\caption{$\mathrm{CRPS}_{\mathrm{sum}}{\downarrow}$ results across 4 datasets. Best result per row in \textbf{bold}.}
\label{tab:crps}
\resizebox{0.8\textwidth}{!}{
\begin{tabular}{llrrrrrrrrrrrrrrr}
\toprule
& & \multicolumn{5}{c}{15 min (3-step)}
  & \multicolumn{5}{c}{30 min (6-step)}
  & \multicolumn{5}{c}{60 min (12-step)} \\
\cmidrule(lr){3-7}\cmidrule(lr){8-12}\cmidrule(lr){13-17}
Dataset & Backbone & naïve & TCVAR & SDRF & LASER & Teger
                   & naïve & TCVAR & SDRF & LASER & Teger
                   & naïve & TCVAR & SDRF & LASER & Teger \\
\midrule

\multirow{4}{*}{PeMS03} & LSTM & 0.0475 & 0.0405 & 0.0396 & 0.0387 & \textbf{0.0383} & 0.0476 & 0.0419 & 0.0408 & \textbf{0.0397} & 0.0401 & 0.0503 & 0.0491 & 0.0463 & 0.0436 & \textbf{0.0401} \\
 & Transformer & 0.0457 & 0.0355 & 0.0334 & 0.0313 & \textbf{0.0292} & 0.0473 & 0.0371 & 0.0350 & 0.0329 & \textbf{0.0305} & 0.0490 & 0.0386 & 0.0363 & 0.0340 & \textbf{0.0316} \\
 & xLSTM & 0.0448 & 0.0390 & 0.0371 & 0.0351 & \textbf{0.0316} & 0.0464 & 0.0405 & 0.0382 & 0.0360 & \textbf{0.0328} & 0.0479 & 0.0420 & 0.0398 & 0.0377 & \textbf{0.0341} \\
 & MTGNN & 0.0539 & 0.0491 & 0.0470 & 0.0449 & \textbf{0.0428} & 0.0509 & 0.0422 & 0.0412 & \textbf{0.0403} & 0.0445 & 0.0583 & 0.0541 & 0.0518 & 0.0496 & \textbf{0.0483} \\
\midrule
\multirow{4}{*}{PeMS04} & LSTM & 0.0229 & 0.0197 & 0.0181 & 0.0165 & \textbf{0.0134} & 0.0242 & 0.0210 & 0.0191 & 0.0173 & \textbf{0.0144} & 0.0255 & 0.0221 & 0.0202 & 0.0184 & \textbf{0.0152} \\
 & Transformer & 0.0218 & 0.0156 & 0.0146 & 0.0136 & \textbf{0.0115} & 0.0231 & 0.0171 & 0.0160 & 0.0150 & \textbf{0.0125} & 0.0243 & 0.0178 & 0.0167 & 0.0156 & \textbf{0.0133} \\
 & xLSTM & 0.0168 & 0.0155 & 0.0145 & 0.0135 & \textbf{0.0112} & 0.0180 & 0.0166 & 0.0153 & 0.0140 & \textbf{0.0120} & 0.0191 & 0.0177 & 0.0163 & 0.0149 & \textbf{0.0128} \\
 & MTGNN & 0.0301 & 0.0255 & 0.0218 & 0.0182 & \textbf{0.0142} & 0.0335 & 0.0301 & 0.0260 & 0.0220 & \textbf{0.0190} & 0.0310 & 0.0275 & 0.0266 & 0.0258 & \textbf{0.0184} \\
\midrule
\multirow{4}{*}{PeMS07} & LSTM & 0.0968 & 0.0956 & 0.0926 & 0.0896 & \textbf{0.0871} & 0.0972 & 0.0949 & 0.0944 & 0.0940 & \textbf{0.0886} & 0.1001 & 0.0989 & 0.0973 & 0.0958 & \textbf{0.0922} \\
 & Transformer & 0.0957 & 0.0932 & 0.0923 & 0.0914 & \textbf{0.0871} & 0.0974 & 0.0949 & 0.0937 & 0.0926 & \textbf{0.0886} & 0.0991 & 0.0966 & 0.0956 & 0.0946 & \textbf{0.0901} \\
 & xLSTM & 0.0941 & 0.0907 & 0.0905 & 0.0903 & \textbf{0.0864} & 0.0958 & 0.0922 & 0.0919 & 0.0916 & \textbf{0.0878} & 0.0975 & 0.0938 & 0.0937 & 0.0936 & \textbf{0.0891} \\
 & MTGNN & 0.1052 & 0.1013 & 0.0967 & 0.0921 & \textbf{0.0874} & 0.1012 & 0.0955 & \textbf{0.0942} & 0.1013 & 0.0978 & 0.1098 & 0.1023 & 0.1015 & 0.1026 & \textbf{0.0964} \\
\midrule
\multirow{4}{*}{Brussels} & LSTM & 0.1455 & 0.0863 & 0.0850 & 0.0871 & \textbf{0.0809} & 0.1487 & 0.0894 & 0.0880 & 0.0895 & \textbf{0.0811} & 0.1512 & 0.0917 & 0.0905 & 0.0921 & \textbf{0.0842} \\
 & Transformer & 0.1668 & 0.0786 & 0.0782 & 0.0779 & \textbf{0.0668} & 0.1671 & 0.0793 & 0.0785 & 0.0794 & \textbf{0.0692} & 0.1717 & 0.0817 & 0.0805 & 0.0821 & \textbf{0.0695} \\
 & xLSTM & 0.1448 & 0.0748 & 0.0740 & 0.0752 & \textbf{0.0591} & 0.1473 & 0.0801 & 0.0788 & 0.0828 & \textbf{0.0611} & 0.1496 & 0.0819 & 0.0808 & 0.0821 & \textbf{0.0619} \\
 & MTGNN & 0.1471 & 0.0918 & 0.0905 & 0.0953 & \textbf{0.0853} & 0.1507 & 0.0987 & 0.0977 & 0.0967 & \textbf{0.0832} & 0.1535 & 0.0971 & 0.0970 & 0.0970 & \textbf{0.0935} \\
\bottomrule
\end{tabular}
}
\end{table}


\begin{table}[t]
\centering
\caption{\textbf{Ablation ladder.} CRPS$_{\mathrm{sum}}\downarrow$ at the 12-step (60\,min) horizon. Row~(3) is the capacity control: it adds the spatial covariance head with reweighting disabled, isolating \emph{added covariance capacity} from \emph{curvature guidance}. Rows~(4)--(6) match row~(7)'s total added edge mass \emph{and} multiplier distribution, so any remaining gap isolates where the mass is placed. $\Delta$ is the mean relative error reduction w.r.t.\ row~(3), averaged over the four datasets; positive is better. $\dagger$ marks cells where row~(7) improves on row~(3) at $p<0.05$; $\ddagger$ marks the same against row~(5) (two-sample $t$, $n=3$ seeds). Per-cell tests have low power at this $n$; the primary statistics are the across-configuration tests in Table~10. \textbf{Error bars:} $\pm$ is the standard deviation over $n=3$ seeds for rows~(3)--(7) and~(9). Rows~(1), (2) and~(8) reproduce the corresponding entries of Table~1; their $\pm$ is the spread over the 24 rolling forecast windows of Table~7 and is \emph{not} comparable to the seed spreads above. \textbf{Row~(9)} is run on Brussels only ($N=195$), where the un-projected precision matrix is tractable and the node-space resistance identity holds exactly; $\Delta$ is undefined for a single dataset. On Brussels the un-projected variant recovers 24.15\% of row~(3)'s error versus 24.51\% for row~(8) (xLSTM), and 2.40\% versus 2.60\% (MTGNN), indicating that the $R=10$ projection costs essentially no accuracy.}
\label{tab:ladder}
\small
\setlength{\tabcolsep}{4pt}
\resizebox{0.8\textwidth}{!}{
\begin{tabular}{clcccccl}
\toprule
\# & Variant & PeMS03 & PeMS04 & PeMS07 & Brussels & $\Delta$ vs.\ (3) & Isolates \\
\midrule
\multicolumn{8}{l}{\textit{xLSTM backbone}} \\
\midrule
(1) & Backbone + diagonal Gaussian & 0.0479$\pm$0.0049 & 0.0191$\pm$0.0023 & 0.0975$\pm$0.0096 & 0.1496$\pm$0.0091 & -35.12\% & baseline \\
(2) & + temporal covariance (= TCVAR, our impl.) & 0.0420$\pm$0.0052 & 0.0177$\pm$0.0036 & 0.0938$\pm$0.0104 & 0.0819$\pm$0.0142 & -7.32\% & temporal structure \\
(3) & + spatial covariance, $\lambda=0$ (no reweighting) & 0.0387$\pm$0.0036 & 0.0149$\pm$0.0017 & 0.0919$\pm$0.0064 & 0.0820$\pm$0.0073 & $0$ & added covariance capacity \\
(4) & + uniform reweighting (mass-matched) & 0.0380$\pm$0.0034 & 0.0160$\pm$0.0019 & 0.0933$\pm$0.0051 & 0.0831$\pm$0.0066 & -2.11\% & added edge mass \\
(5) & + permuted curvature (mass-matched null) & 0.0395$\pm$0.0042 & 0.0173$\pm$0.0010 & 0.0972$\pm$0.0046 & 0.0869$\pm$0.0069 & -7.48\% & topological alignment \\
(6) & + inverse curvature (mass-matched) & 0.0397$\pm$0.0045 & 0.0177$\pm$0.0014 & 0.0986$\pm$0.0059 & 0.0876$\pm$0.0074 & -8.87\% & sign of the signal \\
(7) & \textbf{+ curvature reweighting} & 0.0347$\pm$0.0030 & 0.0137$\pm$0.0016$^\ddagger$ & 0.0902$\pm$0.0077 & 0.0627$\pm$0.0087$^\dagger$$^\ddagger$ & +10.94\% & the claim \\
(8) & + volatility scaling (full) & 0.0341$\pm$0.0023 & 0.0128$\pm$0.0014 & 0.0891$\pm$0.0073 & 0.0619$\pm$0.0094 & +13.38\% & marginal scale \\
(9) & full, projection off (Brussels only) & -- & -- & -- & 0.0622$\pm$0.0098 & -- & projection fidelity \\
\midrule
\multicolumn{8}{l}{\textit{MTGNN backbone}} \\
\midrule
(1) & Backbone + diagonal Gaussian & 0.0583$\pm$0.0061 & 0.0310$\pm$0.0027 & 0.1098$\pm$0.0096 & 0.1535$\pm$0.0099 & -23.12\% & baseline \\
(2) & + temporal covariance (= TCVAR, our impl.) & 0.0541$\pm$0.0059 & 0.0275$\pm$0.0039 & 0.1023$\pm$0.0109 & 0.0971$\pm$0.0149 & -1.37\% & temporal structure \\
(3) & + spatial covariance, $\lambda=0$ (no reweighting) & 0.0533$\pm$0.0039 & 0.0270$\pm$0.0019 & 0.1013$\pm$0.0071 & 0.0960$\pm$0.0085 & $0$ & added covariance capacity \\
(4) & + uniform reweighting (mass-matched) & 0.0538$\pm$0.0027 & 0.0272$\pm$0.0021 & 0.1016$\pm$0.0068 & 0.0964$\pm$0.0080 & -0.60\% & added edge mass \\
(5) & + permuted curvature (mass-matched null) & 0.0540$\pm$0.0025 & 0.0274$\pm$0.0011 & 0.1020$\pm$0.0072 & 0.0973$\pm$0.0092 & -1.21\% & topological alignment \\
(6) & + inverse curvature (mass-matched) & 0.0547$\pm$0.0017 & 0.0283$\pm$0.0017 & 0.1029$\pm$0.0086 & 0.0979$\pm$0.0086 & -2.75\% & sign of the signal \\
(7) & \textbf{+ curvature reweighting} & 0.0499$\pm$0.0024 & 0.0190$\pm$0.0018$^\dagger$$^\ddagger$ & 0.0978$\pm$0.0064 & 0.0939$\pm$0.0079 & +10.41\% & the claim \\
(8) & + volatility scaling (full) & 0.0483$\pm$0.0023 & 0.0184$\pm$0.0015 & 0.0964$\pm$0.0070 & 0.0935$\pm$0.0097 & +12.17\% & marginal scale \\
(9) & full, projection off (Brussels only) & -- & -- & -- & 0.0937$\pm$0.0075 & -- & projection fidelity \\
\bottomrule
\end{tabular}
}
\end{table}

\begin{table}[t]
\centering
\caption{\textbf{Across-configuration tests over the 8 (dataset, backbone) cells of Table~9.} Because per-cell $t$-tests have low power at $n=3$ seeds, these aggregate tests are the primary statistics. The sign test and the paired Wilcoxon signed-rank test are distribution-free and do not depend on the seed variance estimates. One-sided $p$-values, taken in the direction stated in the Direction column. Rows~(6) and~(4) are predicted to be \emph{non}-improvements: inverse curvature should underperform the null if the sign of the curvature carries information, and uniform mass addition should not help if the gain is not simply added edge mass.}
\label{tab:ladder-stats}
\small
\begin{tabular}{llccc}
\toprule
Comparison & Direction & Mean & Range & Sign / Wilcoxon $p$ \\
\midrule
(7) vs (5) \emph{targeting} & (7) better in 8/8 & $+14.23\%$ & $+3.49$ to $+30.66\%$ & $0.0039$ / $0.0039$ \\
(7) vs (3) \emph{capacity} & (7) better in 8/8 & $+10.68\%$ & $+1.85$ to $+29.63\%$ & $0.0039$ / $0.0039$ \\
(6) vs (5) \emph{signed signal} & (6) worse in 8/8 & $-1.39\%$ & $-3.28$ to $-0.51\%$ & $0.0039$ / $0.0039$ \\
(8) vs (7) \emph{volatility} & (8) better in 8/8 & $+2.38\%$ & $+0.43$ to $+6.57\%$ & $0.0039$ / $0.0039$ \\
(4) vs (3) \emph{mass alone} & (4) worse in 7/8 & $-1.35\%$ & $-7.38$ to $+1.81\%$ & $0.0352$ / $0.0391$ \\
\bottomrule
\end{tabular}
\end{table}

% Companions

\begin{table}[ht]
\centering
\caption{Companion Table A}
\begin{tabular}{l l l l l l l l l l}
\hline
Dataset & Teger $\Delta\mathcal{K}$ & Null $\Delta\mathcal{K}$ & $p$ & Teger $\Delta\lambda_2$ & Null $\Delta\lambda_2$ & $p$ & Teger $\Delta\phi(S^\star)$ & Null $\Delta\phi(S^\star)$ & $p$ \\
\hline
Brussels & 34.1\% & 1.2$\pm$0.4\% & <0.001 & 37.1\% & 1.5$\pm$0.6\% & <0.001 & 33.9\% & 0.8$\pm$0.5\% & <0.001 \\
PeMS03 & 20.6\% & 1.1$\pm$0.3\% & <0.001 & 21.2\% & 1.3$\pm$0.4\% & <0.001 & 21.9\% & 0.9$\pm$0.3\% & <0.001 \\
PeMS04 & 22.6\% & 1.4$\pm$0.5\% & <0.001 & 23.6\% & 1.6$\pm$0.5\% & <0.001 & 22.9\% & 1.0$\pm$0.4\% & <0.001 \\
PeMS07 & 9.1\% & 0.8$\pm$0.2\% & 0.002 & 9.3\% & 0.9$\pm$0.3\% & 0.003 & 8.9\% & 0.6$\pm$0.2\% & 0.004 \\
\hline
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{Companion Table B}
\begin{tabular}{l l l l l l}
\hline
Dataset & $\rho(-\kappa, (v_i-v_j)^2)$ & Null (5) & Teger (7) & Oracle & \% of oracle gain \\
\hline
Brussels & 0.82 & 0.0869 & 0.0627 & 0.0600 & 89.9\% \\
PeMS03 & 0.76 & 0.0395 & 0.0347 & 0.0335 & 80.0\% \\
PeMS04 & 0.79 & 0.0173 & 0.0137 & 0.0130 & 83.7\% \\
PeMS07 & 0.71 & 0.0972 & 0.0902 & 0.0890 & 85.3\% \\
\hline
\end{tabular}
\end{table}

\begin{table}[t]
\centering
\caption{Sensitivity to the reweighting functional form. All rows use the same
curvature ranking and the same total added edge mass; only the shape of
$b_{ij}(\kappa_{ij})$ changes. CRPS$_{\mathrm{sum}}\downarrow$, xLSTM, mean$\pm$std
over $\geq\!3$ seeds.}
\label{tab:form}
\small
\begin{tabular}{llcccc}
\toprule
Form & $b_{ij}$ & PeMS03 & PeMS04 & PeMS07 & Brussels \\
\midrule
softplus (ours) & $\mathrm{softplus}(\tau(\kappa_0-\kappa_{ij}))$ & $0.0316 \pm 0.0023$ & $0.0112 \pm 0.0011$ & $0.0864 \pm 0.0035$ & $0.0591 \pm 0.0042$ \\
exponential     & $\exp(\tau(\kappa_0-\kappa_{ij}))$              & $0.0324 \pm 0.0025$ & $0.0118 \pm 0.0013$ & $0.0871 \pm 0.0038$ & $0.0612 \pm 0.0045$ \\
linear (hinge)  & $\max(0,\tau(\kappa_0-\kappa_{ij}))$            & $0.0331 \pm 0.0022$ & $0.0125 \pm 0.0015$ & $0.0885 \pm 0.0041$ & $0.0635 \pm 0.0051$ \\
hard top-$k$    & $\mathbf{1}[\kappa_{ij}\le\kappa_{(k)}]$        & $0.0345 \pm 0.0028$ & $0.0132 \pm 0.0018$ & $0.0898 \pm 0.0045$ & $0.0681 \pm 0.0058$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[ht]
\centering
\caption{Companion Table D}
\begin{tabular}{l l l l}
\hline
Graph measure & Pearson $r$ (pooled) & Spearman $\rho$ (pooled) & Mean within-dataset $\rho$ \\
\hline
$\Delta\mathcal{K}$ (Kirchhoff) & 0.91 & 0.88 & 0.84 \\
$\Delta\lambda_2$ (spectral gap) & 0.93 & 0.89 & 0.86 \\
$\Delta\phi(S^\star)$ (conductance) & 0.89 & 0.85 & 0.82 \\
\hline
\end{tabular}
\end{table}



## W1 (kgxJ): time-resolved bottleneck ↔ residual-correlation test — findings

**Bottom line: the test does not support the motivating claim, on either dataset.**
This document reports that result and what to do about it.

Artifacts:
- `src/curvature_bottleneck_residual_corr_timeresolved.py` (pre-specification in the module docstring)
- `metrics_stat/xLSTM/{brussels,pems03}_xlstm_bottleneck_residual_corr_timeresolved.json`
- `visualizations/{brussels,pems03}_xlstm_bottleneck_residual_corr_timeresolved.pdf`
- `NeurIPS/timeresolved_bottleneck_table.tex`

---

## 1. Why the original W1 test could not have answered the question

This codebase reserves `prediction_horizon + num_pred_rolling` steps after the validation
cutoff for test. For **every** 5-min traffic dataset (Brussels, PeMS03/04/07: H=12, R=24)
that is 35 steps = **2.9 hours of calendar time on a single weekday**. Brussels' held-out
decoder span is 2024-01-10 09:20–12:10; all 24 rolling origins lie between 09:20 and 11:15.

So the original null result was obtained from one 2.9-hour midday window. It could not
speak to a claim about *time-varying*, latent-state-dependent propagation. Sub-windowing
those 24 origins is not a remedy either: at T<12 the independence noise floor
`sqrt(2/(pi(T-1)))` exceeds 0.24, which is larger than the entire Q0–Q3 spread being
interpreted (0.195–0.210).

## 2. What was run instead

Same checkpoints, 21-day in-sample evaluation span (6037 rolling origins each), residuals
resolved by traffic regime. Hypothesis, windows, primary test, control test and span
selection were all fixed **before** any window-resolved number was computed (see the
docstring pre-specification block).

- **H1**: a bottleneck only binds under congestion, so the association between edge
  bottleneck score `b_ij` and `|rho_ij|` should be stronger at peak than at night, on
  weekdays but not weekends.
- **Primary statistic** `beta_b`: coefficient on `z(b)` in
  `|rho_ij| ~ 1 + z(b) + z(distance) + z(degree)` — distance-adjusted, because Balanced
  Forman curvature on a proximity graph is close to a common-neighbour count.
- **Null**: node permutation (valid under temporal autocorrelation; permutes the node axis
  only). 10,000 permutations, shared stream across windows so contrasts are paired.
- **Multiplicity**: Holm across the 12 pre-specified window tests. The 24 hour-of-day bins
  are descriptive only.
- Estimator validated against a planted effect (recovered at z=+5.99) and against pure
  noise (fabricated nothing).

## 3. Results

### Primary and control contrasts

| dataset | contrast | Δ | z | p | H1 requires |
|---|---|---|---|---|---|
| Brussels | **PRIMARY** peak−night, weekday | **−0.0066** | −1.06 | 0.855 (1-sided) | Δ > 0 ✗ |
| Brussels | CONTROL peak−night, weekend | −0.0122 | −2.40 | 0.016 (2-sided) | smaller than weekday |
| Brussels | weekday − weekend | +0.0056 | | | > 0 ✓ |
| PeMS03 | **PRIMARY** peak−night, weekday | **+0.0050** | +0.51 | 0.301 (1-sided) | Δ > 0 — right sign, n.s. |
| PeMS03 | CONTROL peak−night, weekend | +0.0142 | +1.85 | 0.032 (1-sided) | smaller than weekday ✗ |
| PeMS03 | weekday − weekend | **−0.0092** | | | > 0 ✗ **falsified** |

On Brussels the primary contrast has the **wrong sign**: peak `beta_b` (+0.0062) is *lower*
than night (+0.0128). On PeMS03 the sign is right but not significant, and the
pre-specified falsifier fails outright — the weekend contrast is *larger* than the weekday
contrast, which is the opposite of what a commute-congestion mechanism predicts.

### Per-window (weekday), Holm-corrected

No window is significant on either dataset. Brussels smallest `p_Holm` = 0.68 (midday);
PeMS03 all `p_Holm` = 1.00. The quartile profile stays non-monotone in every regime.

### The cherry-picking demonstration (important)

Descriptive hour-of-day sweep, uncorrected:

- **Brussels: 3 of 24 hourly windows reach nominal p<0.05** (expected by chance: 1.2).
  They are **03:00 (p=0.040), 14:00 (p=0.0033), 19:00 (p=0.040)**. None is a commute peak;
  03:00 is the deepest free-flow hour of the night, i.e. the *opposite* of the proposed
  mechanism. The best of them does not survive Holm (0.079).
- **PeMS03: 0 of 24** reach nominal p<0.05.

This is what searching time windows for a monotone profile would have produced: a
"significant" 3 a.m. window that contradicts the mechanism it would have been used to
support. A confidence-4 reviewer asking for the selection rule would end the discussion.

### How large an effect can be excluded

Contrast null SD is 0.0063 (Brussels) and 0.0096 (PeMS03). At 2σ this excludes
peak-vs-night differences larger than **≈6% (Brussels) / ≈5% (PeMS03) of mean |rho| per
standard deviation of bottleneck score**. Absence of evidence is not proof of absence, but
the effect, if it exists, is bounded to be small.

## 4. Recommended manuscript change

The sentence at `submitted_manuscript.tex:144` is an empirical assertion the data does not
support:

> "...the same bottlenecks that distort message passing also distort the spatial covariance
> of residuals---concentrating dependence along thin chains and underestimating cross-region
> correlation."

Suggested replacement:

> Capturing this structure is, at heart, a covariance-modeling problem on a dynamic graph.
> Bottlenecks that constrain message passing are a natural place to look for structure in
> the spatial covariance of residuals, and we use curvature to decide where the covariance
> model is given additional flexibility. We stress that this is a design motivation rather
> than an empirically established mechanism: in Appendix~\ref{app:residual-corr} we test it
> directly, with a pre-specified hypothesis and a node-permutation null, and find no
> evidence that residual correlation is systematically elevated on bottleneck edges---at a
> power sufficient to exclude effects larger than roughly 6\% of mean $|\rho|$ per standard
> deviation of bottleneck score. The justification for curvature-guided reweighting is
> therefore the spectral analysis of Section~\ref{sec:theory} together with the empirical
> gains of Section~\ref{sec:results}, not a verified claim about residual covariance.

Nothing else in the paper depends on the claim: the spectral-gap / effective-resistance
theory is about the graph, not about residuals, and Table 1's gains stand on their own.

## 5. Recommended posture toward kgxJ

Reviewer strength S4 already credits the paper for being "transparent about the scope of its
claims" and calls that "refreshing." Leaning into that is the strongest available move:

> You asked us to compare residual correlation at bottleneck vs. non-bottleneck edges. We
> did, and we report that it does not support the motivating sentence in our introduction.
> We additionally found that our original test was underpowered by construction (the
> held-out split is 2.9 hours), so we re-ran it over 21 days with a pre-specified
> congestion-regime hypothesis, distance and degree controls, and a node-permutation null.
> The pre-specified contrast is null on both datasets and our pre-specified falsifier fails
> on PeMS03. We have accordingly rewritten the motivation to present curvature as an
> inductive bias justified by the spectral analysis and the measured gains, and removed the
> unsupported mechanistic claim.

This converts W1 from "unsupported claim" into "claim corrected," and it also speaks to W2,
since we now say plainly that we do not demonstrate the over-squashing mechanism as the
source of the gains.

## 6. Caveats

- **Residuals are in-sample.** The held-out tail is 2.9 h, so the 21-day span lies inside
  the training period. The analysed quantity is the *spatial correlation* of residuals
  rather than their magnitude, and the primary comparison is between windows of the same
  fitted model, so common in-sample optimism differences out — but this is a diagnostic,
  not out-of-sample confirmation. A clean out-of-sample version needs a retrain with the
  split shifted back ~14 days (~50 min/run).
- One backbone (xLSTM), h=1 residuals, two datasets.
- `b_ij` comes from the checkpoint's learned curvature module; a different parameterisation
  could behave differently.
- A U-shaped `|rho|`-vs-`b` profile — which is what the original figure shows — is the
  signature of a **signed** common factor (mixed-sign loadings give high `|rho|` at both
  extremes). This was confirmed on synthetic data and is a plausible alternative reading of
  the original plot.

## 7. Two defects found along the way (independent of the above)

1. **The committed `brussels_xlstm_bottleneck_residual_corr.json` is not reproducible.**
   `model.predict(n_samples=100)` ran with an unseeded torch RNG. Re-running the identical
   command moved Q0 from 0.2074 to 0.1923 and **flipped the sign** of the median-split
   statistic (−0.0020 → +0.0015). Both scripts now seed explicitly; that JSON should be
   regenerated before it is shown to anyone.
2. **The PeMS03 checkpoint cannot be reproduced from the current repo state.** Its
   `tod`/`dow` embeddings are (288, 38) and (7, 5), but `pems03_flow.csv`'s integer
   `datetime` column yields 26,138 `tod` categories and 1 `dow`. It was trained with the
   branch now commented out at `train_batch.py:551-552`, which overwrites `datetime` with a
   synthetic 5-min range **over rows**. Consequence worth noting: the PeMS03 model has no
   valid time-of-day feature at all, so it could not have read any time-of-day structure off
   an input covariate.



Recheck done against the full text. The attribution is confirmed, the core of the advice holds, but there are **four corrections** — one of them in my own draft text, and one that turns the W5 experiment from "nice to have" into the single most valuable thing you can run.

## 1. My draft had the notation backwards. Fix this before anything else.

The quote is Footnote 1 to Section 3.2 of Black, Wan, Nayyeri & Wang (ICML 2023) — confirmed verbatim, and confirmed that your [22] (Topping et al.) is a different paper. But Black et al.'s convention is:

- $\sigma_1 \le \sigma_2 \le \dots \le \sigma_n$ — eigenvalues of the **unnormalised** $L$; $\sigma_2$ is what they call the spectral gap.
- $\lambda_i$ — eigenvalues of the **normalised** $\hat L$, range $[0,2]$.

So $d_{\min}\lambda_k \le \sigma_k \le d_{\max}\lambda_k$ reads *normalised on the outside, unnormalised in the middle*. My previous draft used $\sigma$ for the normalised gap and $\lambda$ for the unnormalised — exactly inverted, and also inverted relative to **your own** paper, which uses $\lambda_k(L'_t) \ge \lambda_k(L_t)$ for the unnormalised spectrum. Three conventions in play. In a rebuttal answering a reviewer who is counting your typos, importing a swapped sandwich inequality would be fatal. Restate the relation in *your* notation and say so explicitly.

## 2. The upgrade: the W5 experiment also rescues Table 2

This is the thing I missed last time and it changes the calculus.

$\hat L = D^{-1/2}LD^{-1/2}$ is **exactly invariant under uniform rescaling** $W \mapsto cW$. So the normalised spectrum cannot move for trivial reasons — any change in $\lambda_2(\hat L)$ after Algorithm 1 is attributable *entirely* to the non-uniform, curvature-targeted component of your reweighting.

That directly answers the objection I raised earlier about Table 2 being arithmetic rather than mechanism. Let me sharpen that objection with your own Appendix I numbers, because it's worse than I said: 10% of edges get $3.5$–$11.3\times$, the remaining 90% get $\approx 1\times$, so the mean multiplier is roughly $1.4$. Uniform scaling by $1.4$ reduces the Kirchhoff index by $\approx 29\%$ — which is essentially the 20–34% you report. **Table 2's first two columns are consistent with "we multiplied every weight by 1.4."** Please check this number yourself; the learned $\tau, \lambda, \kappa_0$ differ from the initialisation in Table 6, so my estimate is indicative, not exact.

The refinement to my earlier blanket claim: your **third** column is fine. $\phi_W(S) = \mathrm{cut}(S)/\min\{\mathrm{vol}(S), \mathrm{vol}(V\setminus S)\}$ is scale-invariant, since numerator and denominator both carry the factor $c$. So conductance already survives the confound. It's columns 1 and 2 that don't — and $\lambda_2(\hat L)$ would.

Net: one eigendecomposition per graph, no retraining, and it simultaneously (a) closes W5 empirically, (b) de-confounds Table 2, and (c) gives you the scale-free evidence that curvature targeting does something a uniform multiplier doesn't. Run it before you write the rebuttal, not after.

## 3. A proofreading catch the reviewer hasn't made yet

Line 144 fixes your ordering as $\lambda_n \ge \lambda_{n-1} \ge \dots \ge \lambda_1$, so $\lambda_1 = 0$ and the spectral gap is $\lambda_2$. Table 2's second column reports $\lambda'_{n-1}/\lambda_{n-1}$ — under your own convention, the **second-largest** eigenvalue, which is not the spectral gap and has no role in your Section 3.4.1 argument. Either the column header is wrong or the quantity is. Given the tenor of this review, fix it in the revision and mention it in the changes table rather than waiting to be caught.

## 4. The full paper is more double-edged than the footnote suggested

Three passages you need to have an answer for, because citing Black et al. puts them on the table:

**(a) They explicitly caution against reweighting-based resistance claims.** On Arnaiz-Rodríguez et al. (2022): its effectiveness cannot be attributed to reduced total resistance, because the reweighted graph has approximately the same pairwise resistances as the original. That is a reweighting method, like yours, and it is the closest prior work to your intervention. Your Corollary 5 is still *true* — $L' \succeq L$ gives it — but a reader coming from Black et al. will ask whether it is true for reasons that matter. See point 2: the scale-invariant numbers are your answer.

**(b) They rank effective resistance above spectral gap, and above curvature.** Their Corollary 3.8 (the $\sigma_2$ bound) is described as looser than the $R_{\text{tot}}$ bound, and they note there was previously no theoretical evidence tying the spectral gap to information passing. Their Figure 2 makes the point that Balanced Forman curvature measures *local* connectivity while effective resistance measures *global*, and their experiments have curvature-based SDRF underperforming the global methods.

The tactical consequence: **lead your W5 response with Direction 2 (effective resistance / Kirchhoff index), not Direction 1 (spectral gap).** Effective resistance is defined through $L^+$ — the unnormalised pseudoinverse — so your Section 3.4.2 is already stated in the object Black et al. treat as canonical, with no normalisation question arising at all. That is a much cleaner alignment than anything you can claim about the gap. Their Lemma 3.1 additionally shows $R_{u,v}$ is computable from $\hat L^+$, i.e. the quantity is indifferent to the choice you're being criticised for.

**(c) Pre-empt the curvature-is-local point in one sentence** rather than letting a reviewer or the AC raise it: your setting fixes the topology, so a global edge-addition criterion is unavailable to you by construction, and the local criterion is being used to *reweight* an existing physical graph — which is also your Figure 2 argument against LASER. Turn the limitation into the design rationale you already have.

## Corrected Remark 4

> **Remark 4 (Choice of Laplacian).** We work throughout with the unnormalised Laplacian $L = D - W$, which is the matrix constructed in Algorithm 1 and the one underlying the effective resistance and Kirchhoff index of Section 3.4.2. This follows Black et al. (2023), whose analysis of over-squashing is built on effective resistance, defined via $L^{+}$, and on the unnormalised spectral gap; they note that Cheeger-type inequalities exist for both the normalised and unnormalised gap (Chung, 1997), so both quantify connectivity and bottleneck structure, and that the two spectra are related by $d_{\min}\,\lambda_k(\tilde L) \le \lambda_k(L) \le d_{\max}\,\lambda_k(\tilde L)$, where $\tilde L = D^{-1/2}LD^{-1/2}$.
>
> This relation does not transfer Löwner monotonicity. Reweighting modifies $D$ as well as $W$, so the constants in the sandwich change with the spectrum; combining it with Proposition 3 would require $\lambda_k(L')/\lambda_k(L) \ge d'_{\max}/d_{\min}$, which we do not assume and do not expect to hold in general. We therefore make no claim that $\lambda_k(\tilde L') \ge \lambda_k(\tilde L)$.
>
> We instead report the normalised spectral gap $\lambda_2(\tilde L)$ before and after Algorithm 1 for all four graphs (Table 2). Because $\tilde L$ is invariant under uniform rescaling $W \mapsto cW$, this quantity is unaffected by the overall increase in edge weight and isolates the effect of curvature-targeted, non-uniform reweighting; we report the degree ratio $d_{\max}/d_{\min}$ alongside it.

## Rebuttal paragraph

> **W5.** We agree that our monotonicity argument does not cover the normalised Laplacian, and Remark 4 now states the reason precisely rather than deferring it: reweighting changes $D$, so the Löwner order does not transfer, and we make no claim about $\lambda_k(\tilde L)$. We note that the unnormalised spectral gap and, more centrally, effective resistance — defined through $L^{+}$ — are the objects used in the over-squashing analysis of Black et al. (2023, ICML), that Cheeger's inequality holds in both normalised and unnormalised form (Chung, 1997), and that the two spectra are two-sidedly related. Our Section 3.4.2 is stated directly in the effective-resistance formulation. We have additionally measured $\lambda_2(\tilde L)$ before and after Algorithm 1 on all four graphs (Table 2, new column). Since $\tilde L$ is invariant to uniform rescaling of the weights, this also isolates the contribution of curvature-targeted reweighting from the overall weight increase, addressing a possible confound in our original Table 2. [Report the measured numbers.]

## Answer to your actual question

Yes — W5 is answerable without the covariance-vs-distance plot. That plot is for W2 and remains the thing that decides the paper; W5 needs only the honest scope statement, the correct citation, and one eigendecomposition per graph.

But the eigendecomposition is not optional now. Without it you are conceding W5 and leaving the Table 2 confound live for the AC to find. With it you convert a concession into a strengthened result, and the marginal cost is an afternoon.

I can write the corrected Prop 3(ii) with proof, or draft the new Table 2 caption and column definitions, whenever you want.

Short answer: **no.** Table 2 is the one thing that cannot answer W2, because Table 2 *is* what the reviewer objected to. Reread the sentence: the theoretical results "merely show that edge reweighting improves standard graph proxies such as spectral gap and effective resistance, without ever connecting these quantities to the Jacobian decay." Table 2 reports the spectral gap ratio, the Kirchhoff index ratio, and a conductance ratio. Answering with it restates the premise of the objection and treats the conclusion as unstated.

## The proposed strategy has a broken link — and citing it makes W2 worse

The chain "bottleneck → spectral/resistance quantities → Jacobian decay → rewiring helps" is sound, and every citation in it is real. The problem is where the third arrow lives. Black et al.'s Lemma 3.2 bounds $\|\partial h_u^{(r)}/\partial x_v\|$ by $(2\alpha\beta)^r \sum_{l=0}^{r}(\hat A^l)_{uv}$; Topping's influence score is $J_{r+1}(i,s)$; Di Giovanni's obstruction is layer- and width-indexed. All three are theorems *about* the recursion $h^{(l+1)}_v = \phi_l\big(h^{(l)}_v, \sum_{u \in \mathcal N(v)} \hat A_{uv}\psi_l(h^{(l)}_u)\big)$. Take that recursion away and the bounds have no left-hand side.

So if you cite four papers whose Jacobian results all presuppose message passing, and your model has none, you convert "borrowed terminology" from an accusation into a demonstration. The reviewer's strict reading survives the citations intact.

## But you do have a Jacobian, and it is graph-dependent

This is better than what I told you last time, and it's the thing that actually opens W2.

Section 3.5 does not just report uncertainty. It *refines the forecast*: at time $t+1$ you condition the next-step residual on the observed window, so the refined prediction at node $i$ is a linear function of observed residuals at other nodes,

$$\hat\eta_{t+1} = M_t\,\eta_{t-D+1:t}, \qquad M_t = \Sigma_{21}\Sigma_{11}^{-1},$$

and $M_t$ depends on $G_t = Q_t^{-1}$, hence on $L'_t$, hence on Algorithm 1. Define

$$J_{ij}^{(k)} \;=\; \left|\frac{\partial \hat x_{i,t+1}}{\partial \eta_{j,t-k}}\right| \;=\; \big|(M_t)_{i,(k,j)}\big|.$$

That is the sensitivity of an output at one node to an input at another, mediated by the graph. It is precisely the *object* Topping, Di Giovanni, and Black use to define over-squashing. Message passing is required to prove *their bounds on it* — it is not required to *define* it, and it is not required for the phenomenon (distant nodes failing to influence each other) to occur.

That reframes your whole response to W2. You are not borrowing the term; you are adopting their definition and proving the corresponding bound for your architecture instead of theirs. And $M_t$ is explicitly constructed at inference, so $J^{(k)}_{ij}$ is a matrix read-off, not even an autograd call.

The provable part: $Q_t = (\alpha+\sigma_{\min})I_R + \beta L_t^{(R)}$, so

$$G_t = \int_0^\infty e^{-(\alpha+\sigma_{\min})s}\, e^{-\beta s L_t^{(R)}}\, ds,$$

a Laplace-weighted heat kernel. Diffusion-suppressing bottlenecks in $L'_t$ therefore suppress entries of $G_t$, hence entries of $M_t$, hence $J_{ij}$ — decay in the same spectral quantities Table 2 measures. Same phenomenon, correct object, no borrowed vocabulary.

The honest gap remains: $L^{(R)}_t = \hat P^\top L'_t \hat P$ lives in the $R$-dimensional projected space, so the decay is in the $L^{(R)}$ geometry, and the route back to node pairs runs through the learned, unconstrained $\hat P$ and $L^{bat}_t$. State that explicitly — the AC will find it otherwise.

## What Table 2 is for

It becomes the middle link, not the answer:

1. Curvature-targeted reweighting improves expansion (Table 2 — with the $\lambda_2(\tilde L)$ column, which is scale-invariant and therefore not confounded).
2. Improved expansion tightens the heat-kernel decay in $G_t$ (new proposition).
3. Which measurably thickens the tail of $J_{ij}$ against graph distance (new figure, no retraining).

Presented as "Table 2 answers W2," it fails. Presented as step 2 of 3, it does real work. Say this explicitly in the rebuttal: *the reviewer is right that graph proxies alone do not constitute an over-squashing result; we now supply the missing link rather than asserting it.*


```python
import os
import pandas as pd

# Check files in current working directory or /mnt/data or search for CRPS_results(1).xlsx
print(os.listdir('.'))
for root, dirs, files in os.walk('.'):
    for f in files:
        if 'CRPS' in f or 'xlsx' in f:
            print(os.path.join(root, f))


```

```text
['CRPS_results(1).xlsx']
./CRPS_results(1).xlsx


```

```python
excel_path = './CRPS_results(1).xlsx'
xls = pd.ExcelFile(excel_path)
print("Sheet names:", xls.sheet_names)

# Read crps_mean sheet
df = pd.read_excel(excel_path, sheet_name='crps_mean')
print(df.to_markdown(index=False))


```

```text
Sheet names: ['CRPS_results', 'CRPS_mean']


```

```python
excel_path = './CRPS_results(1).xlsx'
xls = pd.ExcelFile(excel_path)
print("Sheet names:", xls.sheet_names)


```

```text
Sheet names: ['CRPS_results', 'CRPS_mean']


```

```python
df = pd.read_excel(excel_path, sheet_name='CRPS_mean')
print(df.to_markdown(index=False))


```

```text
| Unnamed: 0   | Unnamed: 1   | 60 min (12-step)   | Unnamed: 3   | Unnamed: 4          | Unnamed: 5   | Unnamed: 6   |
|:-------------|:-------------|:-------------------|:-------------|:--------------------|:-------------|:-------------|
| Dataset      | Backbone     | naïve              | TCVAR        | SDRF                | LASER        | Teger        |
| PeMS03       | LSTM         | 0.0481             | 0.0403       | 0.0364154325126359  | 0.0361       | 0.0342       |
| PeMS03       | Transformer  | 0.049              | 0.0386       | 0.0375080028035388  | 0.034        | 0.0311       |
| PeMS03       | xLSTM        | 0.0479             | 0.042        | 0.04314518171972427 | 0.0377       | 0.0270       |
| PeMS03       | MTGNN        | 0.0512             | 0.0505       | 0.0458917174239944  | 0.0433       | 0.0399       |
| PeMS07       | LSTM         | 0.0831             | 0.0789       | 0.0590306001335018  | 0.0551       | 0.0491       |
| PeMS07       | Transformer  | 0.0812             | 0.0761       | 0.0563434070966889  | 0.0526       | 0.0471       |
| PeMS07       | xLSTM        | 0.0715             | 0.0688       | 0.0472472025718656  | 0.0437       | 0.0334       |
| PeMS07       | MTGNN        | 0.0998             | 0.0843       | 0.064883542066028   | 0.0627       | 0.0621       |
| Brussels     | LSTM         | 0.0712             | 0.0617       | 0.0446775506708618  | 0.0431       | 0.0342       |
| Brussels     | Transformer  | 0.0583             | 0.0517       | 0.0332312997749124  | 0.0342       | 0.0275       |
| Brussels     | xLSTM        | 0.0696             | 0.0529       | 0.0352709150015111  | 0.0331       | 0.0239       |
| Brussels     | MTGNN        | 0.0765             | 0.0541       | 0.0381439460145133  | 0.0372       | 0.0294       |


```

```python
df_raw = pd.read_excel(excel_path, sheet_name='CRPS_mean', header=None)
print(df_raw.to_string())


```

```text
           0            1                 2       3         4       5       6
0        NaN          NaN  60 min (12-step)     NaN       NaN     NaN     NaN
1    Dataset     Backbone             naïve   TCVAR      SDRF   LASER   Teger
2     PeMS03         LSTM            0.0481  0.0403  0.036415  0.0361  0.0342
3     PeMS03  Transformer             0.049  0.0386  0.037508   0.034  0.0311
4     PeMS03        xLSTM            0.0479   0.042  0.043145  0.0377  0.0270
5     PeMS03        MTGNN            0.0512  0.0505  0.045892  0.0433  0.0399
6     PeMS07         LSTM            0.0831  0.0789  0.059031  0.0551  0.0491
7     PeMS07  Transformer            0.0812  0.0761  0.056343  0.0526  0.0471
8     PeMS07        xLSTM            0.0715  0.0688  0.047247  0.0437  0.0334
9     PeMS07        MTGNN            0.0998  0.0843  0.064884  0.0627  0.0621
10  Brussels         LSTM            0.0712  0.0617  0.044678  0.0431  0.0342
11  Brussels  Transformer            0.0583  0.0517  0.033231  0.0342  0.0275
12  Brussels        xLSTM            0.0696  0.0529  0.035271  0.0331  0.0239
13  Brussels        MTGNN            0.0765  0.0541  0.038144  0.0372  0.0294


```

| Dataset | Backbone | naïve | TCVAR | SDRF | LASER | Teger |
| --- | --- | --- | --- | --- | --- | --- |
| PeMS03 | LSTM | 0.0481 | 0.0403 | 0.036415 | 0.0361 | **0.0342** |
| PeMS03 | Transformer | 0.0490 | 0.0386 | 0.037508 | 0.0340 | **0.0311** |
| PeMS03 | xLSTM | 0.0479 | 0.0420 | 0.043145 | 0.0377 | **0.0270** |
| PeMS03 | MTGNN | 0.0512 | 0.0505 | 0.045892 | 0.0433 | **0.0399** |
| PeMS07 | LSTM | 0.0831 | 0.0789 | 0.059031 | 0.0551 | **0.0491** |
| PeMS07 | Transformer | 0.0812 | 0.0761 | 0.056343 | 0.0526 | **0.0471** |
| PeMS07 | xLSTM | 0.0715 | 0.0688 | 0.047247 | 0.0437 | **0.0334** |
| PeMS07 | MTGNN | 0.0998 | 0.0843 | 0.064884 | 0.0627 | **0.0621** |
| Brussels | LSTM | 0.0712 | 0.0617 | 0.044678 | 0.0431 | **0.0342** |
| Brussels | Transformer | 0.0583 | 0.0517 | 0.033231 | 0.0342 | **0.0275** |
| Brussels | xLSTM | 0.0696 | 0.0529 | 0.035271 | 0.0331 | **0.0239** |
| Brussels | MTGNN | 0.0765 | 0.0541 | 0.038144 | 0.0372 | **0.0294** |