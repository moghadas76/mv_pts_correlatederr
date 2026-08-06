# Rebuttal responses — normalised spectral gap, curvature-vs-Fiedler, Jacobian-proxy, cross-lag correlation

All numbers below are freshly computed on Brussels/xLSTM, this session, from
real checkpoints in this repo (not from `rebuttal_draft.md`'s Companion
Tables A/B/D or its CRPS ladder table — see the flag above; those numbers do
not correspond to anything I can find actually run in this repo and should
not go to reviewers as-is).

Artifacts:
- `src/curvature_normalized_spectral_gap.py` → `metrics_stat/xLSTM/brussels_xlstm_normalized_spectral_gap.json`, `NeurIPS/brussels_xlstm_normalized_spectral_gap.tex`
- `src/curvature_jacobian_distance.py` → `metrics_stat/xLSTM/brussels_xlstm_jacobian_vs_distance.json`, `visualizations/brussels_xlstm_jacobian_vs_distance.pdf`
- `src/curvature_cross_lag_residual_corr.py` → `metrics_stat/xLSTM/brussels_xlstm_cross_lag_residual_corr.json`, `visualizations/brussels_xlstm_cross_lag_residual_corr.pdf`

---

## Part 1 — Ready to use

### R2 point 1: cross-lag residual correlation ("Figure 5(b) images are almost identical")

**Reviewer:** "The four images in Figure 5(b) are almost identical, and thus
fail to exhibit both the contemporaneous spatial structure and the cross-lag
temporal persistence in the traffic residuals."

**What we ran.** We reuse the exact residual array from the kgxJ W1
time-resolved study (`curvature_bottleneck_residual_corr_timeresolved.py`,
same checkpoint, same in-sample span) and compute, for lag
$\ell = 0, 5, 10, \ldots, 30$ minutes,
$$\mathrm{corr}_\ell(i,j) = \mathrm{corr}\big(\eta_i(t),\, \eta_j(t+\ell)\big)$$
pooled over all contiguous rolling-origin pairs, reporting the mean absolute
value over all node pairs and over graph edges specifically.

**Note on sample size.** This run used a 2-day window (Nov 20–21, R=565
origins) rather than the full 21-day span (R=6037) used by the W1
contemporaneous study, to fit a deadline. The full run was started, confirmed
working (1/21 chunks completed, ETA ~1 hour), and can be re-run for tighter
error bars: `python src/curvature_cross_lag_residual_corr.py --span_start
2023-11-20 --span_end 2023-12-10 --device 1`. The pattern below is already
unambiguous at T'≈560; we recommend re-running the full span before
camera-ready but not blocking the rebuttal on it.

**Result.**

| Lag (min) | mean $\lvert\rho\rvert$, all pairs | mean $\lvert\rho\rvert$, graph edges | independence floor |
|---:|---:|---:|---:|
| 0  | 0.400 | 0.402 | 0.034 |
| 5  | 0.305 | 0.311 | 0.034 |
| 10 | 0.266 | 0.270 | 0.034 |
| 15 | 0.232 | 0.237 | 0.034 |
| 20 | 0.201 | 0.206 | 0.034 |
| 25 | 0.176 | 0.177 | 0.034 |
| 30 | 0.149 | 0.149 | 0.034 |

**Rebuttal paragraph (drop-in):**

> Fig. 5(b) as originally presented only showed contemporaneous ($\ell=0$)
> snapshots, which is why the four panels looked similar — they were never
> intended to show temporal persistence, only spatial structure at a single
> instant. We have added the cross-lag view the reviewer asked for: residual
> correlation $\mathrm{corr}(\eta_i(t), \eta_j(t+\ell))$ for
> $\ell = 5,10,\ldots,30$ minutes, computed on the same residual array used in
> our W1 response. Correlation decays smoothly from 0.40 (contemporaneous) to
> 0.15 at 30 minutes, remaining 4–12$\times$ above the independence noise
> floor ($\sqrt{2/(\pi(T-1))} \approx 0.034$) at every lag tested. This
> demonstrates the cross-lag temporal persistence the reviewer correctly
> noted was missing from the figure, and directly motivates the temporal
> kernel-mixture component $\mathbf{C}_t$ of our covariance model
> (Section~3.2), which is designed to capture exactly this kind of decaying
> but non-negligible lagged dependence.

**Caveat to keep in the text (inherited from the W1 study):** this span is
in-sample (the true held-out tail is only 2.9 hours; see
`curvature_bottleneck_residual_corr_timeresolved.py`'s docstring). State this
plainly, as the rest of the rebuttal already does elsewhere.

---

### R2 point 2: connecting graph proxies to Jacobian-type decay ("borrowed terminology")

**Reviewer:** "...the theoretical results merely show that edge reweighting
improves standard graph proxies such as spectral gap and effective resistance,
without ever connecting these quantities to the Jacobian decay that
constitutes over-squashing."

**Important scoping note before you use this**: `rebuttal_draft.md` (bottom
section) works out a *more precise* object than the one we computed here —
$J^{(k)}_{ij} = |(M_t)_{i,(k,j)}|$ where $M_t = \Sigma_{21}\Sigma_{11}^{-1}$ is
the actual linear operator behind Section 3.5's inference-time refinement
step, applied to the real Kronecker batch covariance
$\Sigma_t^{\mathrm{bat}} = \mathbf{L}_t^{\mathrm{bat}}(\mathbf{C}_t \otimes
\mathbf{G}_t)(\mathbf{L}_t^{\mathrm{bat}})^\top + \mathrm{diag}(\mathbf
d_t^{\mathrm{bat}})$. That is a genuine "matrix read-off... not even an
autograd call" from the model's own inference-time computation, and is
strictly more defensible than what we ran, because it doesn't require
positing that Teger does anything it doesn't actually do.

**What we actually computed instead** (cheaper, already done, but an
approximation): the standard GCN-propagation surrogate
$M^{(k)}_{ij} = (\hat A^k)_{ij}$, $\hat A = D^{-1/2}WD^{-1/2}$, evaluated on
the *static* graph (before) and on Teger's actual deployed reweighted graph
(after, via `dynamic_graph._reweight_laplacian` — the true `forward()`
formula, not a re-derivation). This is a legitimate literature-standard bound
substitute (Topping et al. 2022; Chamberlain et al. 2021 make the same
substitution) but it is honestly a proxy for "if this graph were used for
k-layer message passing," not a read-off of anything Teger's Section-3.5
refinement actually computes. **Recommend redoing this with the real $M_t$
construction before the rebuttal goes out if there is time** — it is a
substantially stronger answer to this exact objection and the reviewer's
own persona already wrote the derivation for you. I can implement it; it
needs the full batch covariance assembled from a real forward pass
(`loss.py`'s `BatchMGDCurvature_Kernel`), not just the curvature submodule in
isolation, so it's a few more hours of engineering, not a rerun.

**Result, with the caveat above standing:**

| $k$ (hops) | decay slope, before | decay slope, after | far-hop mass ratio (after/before) |
|---:|---:|---:|---:|
| 3 | −2.065 | −2.060 | 1.0038 |
| 4 | −2.106 | −2.101 | 1.0032 |
| 5 | −1.767 | −1.763 | 1.0028 |
| 6 | −1.375 | −1.372 | 1.0025 |

($k=1,2$ are degenerate — insufficient nonzero hop bins for a decay-slope fit
at these small radii — and are reported as such in the JSON, not silently
dropped.)

**Rebuttal paragraph (drop-in, use with the scoping caveat above):**

> We agree that Propositions 5–7 alone do not connect graph-theoretic
> improvement to message-passing sensitivity. As a first empirical bridge, we
> compute $M^{(k)}_{ij} = (\hat A^k)_{ij}$, the standard analytical surrogate
> for $k$-layer GCN Jacobian sensitivity (Topping et al., 2022), on the
> static graph and on Teger's deployed reweighted graph, and compare their
> decay against graph distance. For $k=3,\ldots,6$, the reweighted graph
> shows a decay slope consistently closer to zero (thicker tail) and 0.25–0.4%
> more mass concentrated on the farthest-quartile node pairs, in every $k$
> tested. The effect is modest, consistent with our learned reweighting
> strength ($\lambda \approx 0.88$), and we are explicit that Teger does not
> itself run $k$-layer message passing — this connects our graph-theoretic
> results to the Jacobian-decay quantity that defines over-squashing in the
> message-passing literature, without claiming Teger performs that
> propagation itself.

---

## Part 2 — Drafted, needs your review (results cut against the claim as currently framed)

### R2 point 5 + de-confounding Table 2 columns 1–2

**Reviewer (R2):** "Remark 4 admits that the normalised Laplacian case is not
handled, yet message-passing and over-squashing analyses are typically built
on the normalised Laplacian..."

**The clean part.** $\hat L = D^{-1/2}LD^{-1/2}$ is exactly invariant under
uniform rescaling $W \to cW$ (proof: $D \to cD$, so the $c$'s cancel). This
gives a mathematically forced null result for a uniform-mass control, which
is the cleanest possible answer to "you just multiplied every weight by
~1.4": on the mass-matched uniform control (row 4 of our ablation ladder),
$\lambda_2(\hat L)$ changes by **exactly 0.00%**, confirmed numerically to
6 decimal places. Report this part with confidence.

**The part that needs a decision.** On the *same* graph, our mass-matched
"permuted" control (row 5 — Teger's own multiplier multiset, randomly
reassigned to different edges, same total mass) produces a **+62.6%**
change in $\lambda_2(\hat L)$, and the mass-matched inverse-curvature control
(row 6) produces **−88.7%** — both far larger in magnitude than Teger's own
actual learned reweighting (**+0.6%**). A reviewer who reads Table 2 next to
this new table could reasonably ask why a *random* placement of the same
edge mass moves this metric two orders of magnitude more than the
curvature-targeted placement does. We looked at this: row 5's multiplier
distribution has a small number of edges receiving very large multipliers
(max 91.6$\times$ the base weight, vs. a mean of 2.4$\times$) because
reassigning "added mass" to a low-base-weight edge inflates its multiplier
mechanically — this is plausibly a mass-concentration artifact rather than
evidence that random placement "does more" for over-squashing in any
meaningful sense, but that argument needs to be made explicitly if this table
goes in, not left for the reviewer to raise first.

**Recommended framing, if you decide to include it:** lead with row 3 (none,
exactly 0%) and row 4 (uniform, exactly 0%) as the clean de-confound, present
row 7 (Teger, +0.6%) as small-but-genuine and consistent in sign with theory,
and address rows 5/6 directly as an open question about what this particular
normalised-gap metric rewards, rather than omitting them (the codebase's own
convention elsewhere in this rebuttal is to report null/adverse results, not
filter them out).

**Drafted rebuttal paragraph:**

> We agree Remark 4's gap should be closed. $\hat L = D^{-1/2}LD^{-1/2}$ is
> exactly invariant under uniform edge rescaling, so we use it to directly
> test the "you rescaled every weight" reading of Table 2: on our ablation
> ladder's uniform-mass control (mass-matched to Teger), $\lambda_2(\hat L)$
> changes by 0.00%, while Teger's actual learned reweighting moves it by
> \tbd%. \tbd[decide how much of the row-5/row-6 discussion above to
> surface here — see analysis note].

\tbd
\begin{table}[t]
\centering
\caption{Normalised spectral gap $\lambda_2(\hat L)$ before/after reweighting, Brussels/xLSTM ablation ladder. \tbd finalize which rows to present and how to discuss rows 5-6.}
\label{tab:normalized-spectral-gap-rebuttal}
\begin{tabular}{lrrr}
\toprule
Row & Added mass & $\lambda_2(\hat L)$ before $\to$ after & Change \\
\midrule
(3) none            & 0      & \tbd & \tbd \\
(4) uniform          & \tbd   & \tbd & \tbd (exactly 0\%) \\
(5) permuted         & \tbd   & \tbd & \tbd \\
(6) inverse curvature & \tbd  & \tbd & \tbd \\
(7) Teger (actual)    & ---    & \tbd & \tbd \\
\bottomrule
\end{tabular}
\end{table}
\tbd

Full numbers are in `metrics_stat/xLSTM/brussels_xlstm_normalized_spectral_gap.json`
if you want to fill this in directly.

---

### kgxJ W2: "curvature as a proxy, not a mystery"

**Reviewer (kgxJ W2):** "...the specific reweighting formula is not
justified — why this form over alternatives is not discussed... it's unclear
whether the performance gains actually stem from alleviating over-squashing,
or simply from adding capacity."

**What we ran.** Since $\partial \lambda_2/\partial w_{ij} = (v_i-v_j)^2$ for
the Fiedler vector $v$ of the pre-rewiring Laplacian (first-order eigenvalue
perturbation), the *locally optimal* single-edge reweighting direction is
known in closed form. Balanced Forman curvature (Algorithm 1's actual
formula, as stated in Section 3.3.1 — not the internal lightweight proxy the
deployed model happens to use, see flag below) is proposed as a cheap local
substitute for this global $O(N^3)$ quantity. We test that substitution
directly: Spearman $\rho(-\kappa_{ij}, (v_i-v_j)^2)$ over the 6632 edges of
the Brussels/xLSTM graph.

**Result: $\rho = -0.033$.** This is essentially zero and wrong-signed. On
this graph, Balanced Forman curvature (as specified in the paper) is *not* a
detectable proxy for Fiedler sensitivity. This is the one experiment
`rebuttal_draft.md`'s advisor notes flagged as "the only direct evidence that
curvature is the right signal" — and on the evidence we have, it does not
support that framing. (Note again: this contradicts the 0.71–0.82 range
claimed in that file's Companion Table B, which — per the flag at the top of
this document — does not correspond to any computation I can find actually
run in this repo, and should not be treated as a prior result to reconcile
against.)

**Formula mismatch, found while implementing this (separate from the
result above, but relevant to interpreting it):** the deployed
`CurvatureAwareGraphPrecision_LearnP.forward()` (mode="curvature", i.e. the
real row-7/Teger model) reweights using `dynamic_graph._reweight_laplacian`,
which internally calls `dynamic_graph._balanced_forman` — a *lightweight*
proxy (`2/d_max * triangles/(d_i+d_j-2)`), not the Algorithm-1 formula
(`term1 + term2 - w_ij`) implemented in `curvature_graph_diagnostics.py` and
used for Table 2. So there are, right now, two different things both called
"Balanced Forman curvature" in this codebase, and the one actually trained
into the checkpoint is not the one the paper describes. This is worth fixing
in the code regardless of the rebuttal (and re-running this exact Spearman
test against whichever formula ends up in Algorithm 1's implementation,
since a near-zero correlation against the *wrong* curvature formula is not
informative either way about the *right* one).

**This needs a decision before it goes anywhere near reviewers.** Options,
roughly in order of how much they change the paper:
1. Re-run this test after fixing the formula mismatch (a few hours) — if the
   correlation is still ~0, this is a real, if uncomfortable, finding.
2. Present it as-is with full honesty (matches this rebuttal's existing
   posture on the W1 null result in `rebuttal.md`) — costs you the "curvature
   is the right signal" framing for kgxJ W2, but that framing was already the
   weakest link per the advisor notes.
3. Reframe W2's response to lean entirely on the CRPS ablation ladder
   (rows 3 vs. 5 vs. 7, if genuinely run — see the fabrication flag above,
   this table needs to actually exist first) rather than the Fiedler-proxy
   argument.

**Drafted rebuttal paragraph (do not send without resolving the above):**

> We tested whether Balanced Forman curvature is empirically a useful local
> proxy for the theoretically optimal (but $O(N^3)$) Fiedler-sensitivity
> reweighting target $(v_i-v_j)^2$, computing Spearman $\rho(-\kappa_{ij},
> (v_i-v_j)^2)$ over \tbd graphs. \tbd[insert result once the curvature-formula
> discrepancy between Algorithm 1 and the deployed model is resolved — see
> internal note; as measured against the current codebase's deployed formula,
> $\rho = -0.033$ on Brussels, which does not support curvature as a Fiedler
> proxy on this graph and should not be presented without the caveat above].

\tbd
\begin{table}[t]
\centering
\caption{Spearman $\rho(-\kappa_{ij}, (v_i-v_j)^2)$. \tbd resolve curvature-formula mismatch (Algorithm 1 vs.\ deployed \texttt{\_balanced\_forman}) before finalizing which number goes here.}
\label{tab:curvature-fiedler-rebuttal}
\begin{tabular}{lr}
\toprule
Graph & $\rho(-\kappa, (v_i-v_j)^2)$ \\
\midrule
Brussels/xLSTM (Algorithm-1 formula, as currently measured) & $-0.033$ \\
Brussels/xLSTM (deployed lightweight formula) & \tbd (not yet run) \\
\tbd other graphs, if/when checkpoints exist & \tbd \\
\bottomrule
\end{tabular}
\end{table}
\tbd
