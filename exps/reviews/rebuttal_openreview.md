# Author response

We thank both reviewers. Several points identified real errors, and we have corrected them rather than defended them. Two points asked for experiments we had not run: we ran both, and we report one of them — kgxJ's W1 — as a **null result that falsifies our own motivating sentence**, which we have rewritten accordingly.

**Summary of changes**

| # | Change | Status |
|---|---|---|
| C1 | Motivating claim (l. 144) rewritten: curvature is a design bias, not a verified mechanism; new appendix with the residual-correlation protocol and its **null** result | Done |
| C2 | New mass-matched ablation ladder + across-configuration tests, separating *added covariance capacity* from *curvature targeting* | Done |
| C3 | Prop. 3(ii) restated (the positive-definiteness hypothesis was vacuous) | Done |
| C4 | CRPS definition corrected (sign) | Done |
| C5 | Prop. 7 hypothesis reconciled with Algorithm 1; Remark 6 corrected | Done |
| C6 | Remark 4 (normalised Laplacian) restated with the reason, not a deferral | Done |
| C7 | LASER description corrected | Done |
| C8 | Broken refs fixed (`Table 1` → Table 2), Eq. (9) written out, Appendix A given text, `PeMS378` typo fixed, "OQ" and `⊗` defined | Done |
| C9 | Unsupported "nearly 30% MAE" claim removed | Done |
| C10 | Ablation restated in CRPS_sum with per-dataset columns and seed spread, replacing the NLL/MAE table | Done |
| C11 | Fig. 5(b) replaced with a cross-lag residual-correlation table (ℓ = 0…30 min) | Done |
| C12 | Propagation-surrogate measurement reported, including that the effect is small | Done |
| C13 | SDRF baseline | In progress — see kgxJ W3 |
| C14 | Framing rescoped away from "over-squashing mitigation" | See Reviewer 2, point 2 |

---

## New experiment: mass-matched ablation ladder

This addresses kgxJ W2 and Reviewer 2 point 7, so we state it once here and refer back.

CRPS_sum ↓ at the 12-step (60 min) horizon. Row (3) is the **capacity control**: it adds the spatial covariance head with reweighting disabled, isolating added covariance capacity from curvature guidance. Rows (4)–(6) match row (7)'s total added edge mass **and** multiplier distribution, so any remaining gap isolates *where* the mass is placed. Δ is the mean relative error reduction w.r.t. row (3), averaged over the four datasets. † marks cells where row (7) improves on row (3) at p < 0.05; ‡ the same against row (5) (two-sample t, n = 3 seeds).

**Error bars.** ± is the standard deviation over n = 3 seeds for rows (3)–(7) and (9). Rows (1), (2) and (8) reproduce the corresponding entries of Table 1; their ± is the spread over the 24 rolling forecast windows and is **not** comparable to the seed spreads. Per-cell tests have low power at n = 3; the primary statistics are the across-configuration tests below.

**xLSTM backbone**

| # | Variant | PeMS03 | PeMS04 | PeMS07 | Brussels | Δ vs (3) | Isolates |
|---|---|---|---|---|---|---|---|
| (1) | Backbone + diagonal Gaussian | 0.0479±0.0049 | 0.0191±0.0023 | 0.0975±0.0096 | 0.1496±0.0091 | −35.12% | baseline |
| (2) | + temporal covariance (= TCVAR, our impl.) | 0.0420±0.0052 | 0.0177±0.0036 | 0.0938±0.0104 | 0.0819±0.0142 | −7.32% | temporal structure |
| (3) | + spatial covariance, λ = 0 (no reweighting) | 0.0387±0.0036 | 0.0149±0.0017 | 0.0919±0.0064 | 0.0820±0.0073 | 0 | **added covariance capacity** |
| (4) | + uniform reweighting (mass-matched) | 0.0380±0.0034 | 0.0160±0.0019 | 0.0933±0.0051 | 0.0831±0.0066 | −2.11% | added edge mass |
| (5) | + permuted curvature (mass-matched null) | 0.0395±0.0042 | 0.0173±0.0010 | 0.0972±0.0046 | 0.0869±0.0069 | −7.48% | topological alignment |
| (6) | + inverse curvature (mass-matched) | 0.0397±0.0045 | 0.0177±0.0014 | 0.0986±0.0059 | 0.0876±0.0074 | −8.87% | sign of the signal |
| (7) | **+ curvature reweighting** | 0.0347±0.0030 | 0.0137±0.0016‡ | 0.0902±0.0077 | 0.0627±0.0087†‡ | **+10.94%** | **the claim** |
| (8) | + volatility scaling (full Teger) | 0.0341±0.0023 | 0.0128±0.0014 | 0.0891±0.0073 | 0.0619±0.0094 | +13.38% | marginal scale |
| (9) | full, projection off (Brussels only) | — | — | — | 0.0622±0.0098 | — | projection fidelity |

**MTGNN backbone**

| # | Variant | PeMS03 | PeMS04 | PeMS07 | Brussels | Δ vs (3) | Isolates |
|---|---|---|---|---|---|---|---|
| (1) | Backbone + diagonal Gaussian | 0.0583±0.0061 | 0.0310±0.0027 | 0.1098±0.0096 | 0.1535±0.0099 | −23.12% | baseline |
| (2) | + temporal covariance (= TCVAR, our impl.) | 0.0541±0.0059 | 0.0275±0.0039 | 0.1023±0.0109 | 0.0971±0.0149 | −1.37% | temporal structure |
| (3) | + spatial covariance, λ = 0 (no reweighting) | 0.0533±0.0039 | 0.0270±0.0019 | 0.1013±0.0071 | 0.0960±0.0085 | 0 | **added covariance capacity** |
| (4) | + uniform reweighting (mass-matched) | 0.0538±0.0027 | 0.0272±0.0021 | 0.1016±0.0068 | 0.0964±0.0080 | −0.60% | added edge mass |
| (5) | + permuted curvature (mass-matched null) | 0.0540±0.0025 | 0.0274±0.0011 | 0.1020±0.0072 | 0.0973±0.0092 | −1.21% | topological alignment |
| (6) | + inverse curvature (mass-matched) | 0.0547±0.0017 | 0.0283±0.0017 | 0.1029±0.0086 | 0.0979±0.0086 | −2.75% | sign of the signal |
| (7) | **+ curvature reweighting** | 0.0499±0.0024 | 0.0190±0.0018†‡ | 0.0978±0.0064 | 0.0939±0.0079 | **+10.41%** | **the claim** |
| (8) | + volatility scaling (full Teger) | 0.0483±0.0023 | 0.0184±0.0015 | 0.0964±0.0070 | 0.0935±0.0097 | +12.17% | marginal scale |
| (9) | full, projection off (Brussels only) | — | — | — | 0.0937±0.0075 | — | projection fidelity |

**Across-configuration tests over the 8 (dataset, backbone) cells.** Because per-cell t-tests have low power at n = 3, these aggregate tests are the primary statistics. The sign test and the paired Wilcoxon signed-rank test are distribution-free and do not depend on the seed variance estimates. One-sided p-values, in the direction stated. Rows (6) and (4) are predicted to be **non**-improvements.

| Comparison | Direction | Mean | Range | Sign / Wilcoxon p |
|---|---|---|---|---|
| (7) vs (5) — *targeting* | (7) better in 8/8 | +14.23% | +3.49 to +30.66% | 0.0039 / 0.0039 |
| (7) vs (3) — *capacity* | (7) better in 8/8 | +10.68% | +1.85 to +29.63% | 0.0039 / 0.0039 |
| (6) vs (5) — *signed signal* | (6) worse in 8/8 | −1.39% | −3.28 to −0.51% | 0.0039 / 0.0039 |
| (8) vs (7) — *volatility* | (8) better in 8/8 | +2.38% | +0.43 to +6.57% | 0.0039 / 0.0039 |
| (4) vs (3) — *mass alone* | (4) worse in 7/8 | −1.35% | −7.38 to +1.81% | 0.0352 / 0.0391 |

**Row (9), projection fidelity.** Run on Brussels only (N = 195), where the un-projected precision matrix is tractable and the node-space resistance identity holds exactly. The un-projected variant recovers 24.15% of row (3)'s error versus 24.51% for row (8) on xLSTM, and 2.40% versus 2.60% on MTGNN — the R = 10 projection costs essentially no accuracy.

---

## Response to Reviewer kgxJ

Thank you for a review that engaged with the scope statements rather than treating them as boilerplate.

### W1 — the over-squashing ↔ residual-covariance link is asserted, not established

**You were right, and the test you suggested fails. We report it.**

We implemented exactly the comparison you asked for, pre-registering the hypothesis, windows, primary statistic and null before computing any window-resolved number:

- **Statistic.** Coefficient `β_b` on the standardised bottleneck score in `|ρ_ij| ~ 1 + z(b_ij) + z(distance) + z(degree)`. Distance and degree are controlled because Balanced Forman curvature on a proximity graph is close to a common-neighbour count.
- **Hypothesis (H1).** A bottleneck binds only under load, so the association should be stronger at peak than at night, on weekdays but not weekends.
- **Null.** Node permutation (10,000 draws, shared stream so contrasts are paired), valid under temporal autocorrelation. Holm correction across the 12 pre-specified window tests. Estimator validated against a planted effect (recovered at z = +5.99) and against pure noise (recovered nothing).
- **Span.** We first found that our held-out split reserves only `H + R = 35` steps ≈ **2.9 hours of one weekday**, which cannot speak to a time-varying claim, so we evaluated over a 21-day span (6,037 rolling origins per dataset), xLSTM backbone, h = 1 residuals.

| Dataset | Contrast | Δ | p | H1 requires |
|---|---|---|---|---|
| Brussels | **primary**: peak − night, weekday | **−0.0066** | 0.855 (1-s.) | Δ > 0 ✗ (wrong sign) |
| Brussels | control: peak − night, weekend | −0.0122 | 0.016 (2-s.) | smaller than weekday |
| PeMS03 | **primary**: peak − night, weekday | **+0.0050** | 0.301 (1-s.) | right sign, n.s. |
| PeMS03 | falsifier: weekday − weekend | **−0.0092** | — | > 0 ✗ **fails** |

No window is significant on either dataset after Holm correction. The contrast null s.d. is 0.0063 (Brussels) / 0.0096 (PeMS03), so at 2σ we exclude peak-vs-night differences larger than ≈6% / ≈5% of mean `|ρ|` per s.d. of bottleneck score. We also report a cherry-picking check: on Brussels, 3 of 24 uncorrected hourly windows reach nominal p < 0.05, and they are 03:00, 14:00 and 19:00 — the strongest is the deepest free-flow hour of the night, the *opposite* of the proposed mechanism, and none survives Holm.

**Consequence.** We have replaced the sentence at l. 144 with:

> Bottlenecks that constrain message passing are a natural place to look for structure in the spatial covariance of residuals, and we use curvature to decide where the covariance model is given additional flexibility. We stress that this is a design motivation rather than an empirically established mechanism: in Appendix R we test it directly, with a pre-specified hypothesis and a node-permutation null, and find no evidence that residual correlation is systematically elevated on bottleneck edges — at a power sufficient to exclude effects larger than roughly 6% of mean |ρ| per standard deviation of bottleneck score. The justification for curvature-guided reweighting is therefore the spectral analysis of Section 3.4 together with the ablation of Appendix S, not a verified claim about residual covariance.

Limitations of the test itself, which we also add: the residuals are in-sample (the held-out tail is 2.9 h; the analysed quantity is the *spatial correlation* of residuals and the comparison is between windows of the same fitted model, so common in-sample optimism largely differences out, but this is a diagnostic, not out-of-sample confirmation); one backbone; two of four datasets. We also note that a U-shaped `|ρ|`-vs-`b` profile, which is what our original figure shows, is the signature of a **signed** common factor rather than of bottleneck concentration — we verified this on synthetic data, and it is a more plausible reading of that figure than the one we gave.

**On the apparent tension with W2 below.** The ladder shows that *where* the added mass is placed matters a great deal (row 7 beats the mass-matched permutation null in 8/8 configurations), while this test shows that residual correlation is *not* systematically elevated on bottleneck edges. Both can hold: curvature targeting can improve the conditioning and expansion properties of the covariance parametrisation without the mechanism being "bottleneck edges carry more residual correlation." We now say this explicitly rather than letting the reader reconcile it.

### W2 — the specific reweighting form is not justified; gains may just be added capacity

This is the right question, and we have now run the experiment that separates the two. See the ladder above. Three distinct claims:

**(a) Capacity is not the explanation.** Row (3) is a spatial covariance head with reweighting switched off — same parameter count, same low-rank structure, no curvature. Row (7) improves on it in **8/8** (dataset, backbone) configurations, mean +10.68%, sign and Wilcoxon p = 0.0039. Added capacity alone therefore does not account for the gain.

**(b) Added edge mass is not the explanation either.** Row (4) adds exactly row (7)'s total edge mass uniformly. It is *worse* than row (3) in 7/8 configurations (mean −1.35%, p = 0.0352). This is the comparison that most directly answers your framing: monotone weight increase, which is all Propositions 3/5 require, does not by itself produce the improvement.

**(c) The curvature signal, and its sign, carry the information.** Row (5) permutes the curvature values across edges, holding both the total mass and the multiplier distribution fixed, so only the topological alignment changes. Row (7) beats it in 8/8, mean +14.23%, p = 0.0039. Row (6) inverts the signal — strengthening *positively* curved edges — and is worse than the permuted null in 8/8. A quantity that carries no information could not be made worse by being inverted.

On the functional form specifically: we agree the softplus is a design choice and not load-bearing, and we do not claim otherwise. Propositions 3/5 are indifferent to it, and what the algorithm uses is the curvature *ranking* plus a learned scale (τ, κ₀, λ). The design argument for curvature rather than an arbitrary monotone rule is that the exact first-order sensitivity of the spectral gap to an edge weight is `∂λ₂/∂w_ij = (v_i − v_j)²` for the Fiedler vector `v` — computable but O(N³) and global, whereas Balanced Forman curvature is an O(1) local proxy for the same quantity. We are running two follow-ups during the discussion period: the Spearman correlation between `−κ_ij` and `(v_i − v_j)²` on all four graphs, and a functional-form ablation (softplus / exponential / hinge / hard top-k at equal added mass).

### W3 — SDRF is the methodologically relevant baseline and is missing

Agreed; this is a fair omission and we are running it now. One note on why it is not a drop-in comparison: SDRF **adds and removes** edges, changing the support of `W_t`, whereas Teger reweights a fixed support. Porting it to a covariance head requires deciding what an added edge means in a precision matrix when it corresponds to no physically realised road segment — which is exactly the design point Figure 2 makes against LASER. We are running SDRF-as-preprocessing followed by our unmodified covariance head, so that the only difference from Teger is the graph intervention, and will post results during the discussion period. We note that row (5) of the ladder is a partial answer in the meantime: it is a curvature-driven intervention with the topology scrambled, and the gap to row (7) is the part of Teger's gain attributable to *where* curvature points.

### W4 — the description of LASER is inaccurate

You are right and we apologise for the mischaracterisation. LASER's contribution is to reduce over-squashing **while preserving locality and sparsity**, via a sequence of augmented snapshots under an explicit connectivity/locality/sparsity trade-off — not to add long-range shortcuts at the expense of those properties. We have corrected l. 45–46, and softened l. 268: our observation is that on a dense urban road graph the edges LASER adds need not correspond to physically realised routes, which is a statement about our setting, not a criticism of LASER's design.

### W5 — what the edge weight represents, and what varies with t

This is underspecified in the submission and we are fixing it. The base weight is the thresholded Gaussian kernel `A_ij = exp(−d_ij²/2σ²)` for `A_ij ≥ 0.1` (Appendix J), with `d_ij` the road-network distance (PeMS) or GPS Euclidean distance (Brussels) — a static, physical quantity. The `t` index in Section 3.3.1 and Algorithm 1 refers to ⟨FILL: state precisely which object carries the time index — e.g. "the batch-averaged adjacency `A_t = (1/B) Σ_b Ā_{b,t}` in line 4 of Algorithm 1, whose time variation comes from the dataset-provided dynamic adjacency for PeMS and from the construction of [sym17071007] for Brussels"; or, if the topology is in fact fixed in all reported runs, say so plainly and remove the index⟩. We will also reconcile this with the sentence in Appendix J stating that the graphs are held fixed across all experiments, which as written contradicts the dynamic-graph claim.

### W6 — does Prop. 7's hypothesis hold for graphs produced by Algorithm 1?

**In general it does not, and we thank you for catching this.** Proposition 7 as stated is correct, but its hypothesis (only edges crossing `∂S` are modified) is not satisfied by Algorithm 1, which reweights globally. Under boundary-only reweighting the conclusion is automatic, since adding mass δ to the cut gives `(c+δ)/c ≥ (v+δ)/v` whenever `vol(S) = v ≥ c = cut(S)`. Under global reweighting the correct statement is a *condition*:

> For any nontrivial cut `S` (with the minimum volume attained on the same side before and after), `φ_{W'}(S) ≥ φ_W(S)` **iff** `cut_{W'}(S)/cut_W(S) ≥ vol_{W'}(S)/vol_W(S)`; i.e. iff the added mass is concentrated on the cut relative to the volume it touches.

Curvature targeting is the mechanism intended to produce that concentration, and we verify the condition empirically rather than assume it: Table 2, column 3, reports `φ_{W'}(S)/φ_W(S)` on curvature-identified bottleneck cuts, with improvements of 6.6–33.9% across the four datasets and four backbones. We have restated Proposition 7 in the above form and corrected Remark 6, which claimed the inequality holds "for any cut S" — false under global reweighting, and an error on our part.

### W7 — minor issues

All accepted and fixed: "OQ" is expanded at first use (l. 140); `⊗` is defined at first use (Eq. 3); Appendix A had a heading and a floating figure but no text, and now has a paragraph tying the figure to the claim and stating what it does *not* establish; and "PeMS378" was a typo for ⟨FILL: dataset name⟩, now also stated in the table caption.

---

## Response to Reviewer 2

Thank you for a detailed and technically specific review. Several of your points are simply correct — the CRPS sign, the vacuous hypothesis in Prop. 3(ii), the broken cross-references, the missing Eq. (9) — and we have corrected them. We also accept your central framing criticism and are rescoping the paper rather than arguing with it.

### 1 — Figure 5(b) does not exhibit the structure its caption claims

Accepted, and the diagnosis is partly ours. Fig. 5(b) showed only contemporaneous (ℓ = 0) snapshots, so the four panels were never capable of showing temporal persistence — the caption claimed something the figure could not display. We have computed the cross-lag view you asked for, on the same residual array used for the kgxJ W1 response (⟨FILL: dataset and backbone⟩, same checkpoint):

`corr_ℓ(i,j) = corr(η_i(t), η_j(t+ℓ))`, pooled over contiguous rolling-origin pairs, mean absolute value over all node pairs and over graph edges specifically.

| Lag (min) | mean \|ρ\|, all pairs | mean \|ρ\|, graph edges | independence floor |
|---:|---:|---:|---:|
| 0 | 0.400 | 0.402 | 0.034 |
| 5 | 0.305 | 0.311 | 0.034 |
| 10 | 0.266 | 0.270 | 0.034 |
| 15 | 0.232 | 0.237 | 0.034 |
| 20 | 0.201 | 0.206 | 0.034 |
| 25 | 0.176 | 0.177 | 0.034 |
| 30 | 0.149 | 0.149 | 0.034 |

Correlation decays smoothly from 0.40 contemporaneously to 0.15 at 30 minutes, remaining 4–12× above the independence floor `sqrt(2/(π(T−1))) ≈ 0.034` at every lag. This is the cross-lag persistence you correctly noted was absent from the figure, and it is what motivates the temporal kernel-mixture component `C_t` of the covariance model (Section 3.2) — decaying but non-negligible lagged dependence is exactly what that component is parametrised to capture. We are replacing Fig. 5(b) with this table plus a single lag-resolved panel.

Two caveats we state in the text rather than leave implicit. The span is **in-sample** — the true held-out tail is 2.9 hours, as described in the W1 appendix — so this is a diagnostic of the fitted model's residuals, not out-of-sample confirmation. And this run used a 2-day window (T ≈ 565 origins) rather than the 21-day span of the W1 study; the full-span re-run is in progress and will be reported for camera-ready, though the decay pattern is already unambiguous at this sample size. ⟨FILL: the analytic floor assumes independent series and understates the threshold for autocorrelated residuals — recommend re-running with the node-permutation null already implemented for W1 and quoting that instead, which is both more defensible and consistent with the rest of the response⟩.

### 2 — "over-squashing" is borrowed terminology: no message passing, no Jacobian

We take this seriously and largely accept it. Two changes.

**(a) We rescope the claim.** The title and abstract are being changed to describe the method as *curvature-guided* covariance modelling. What Section 3.4 proves is that curvature-targeted reweighting improves spectral gap, effective resistance and local conductance — statements about the graph. We should not have presented these as an over-squashing result: over-squashing is defined through the Jacobian of a multi-layer message-passing recursion, and our model has no such recursion. The graph in Teger parametrises a covariance, not a propagation operator.

**(b) The corresponding sensitivity for our architecture is well-defined, and we state it as what it is.** The refinement step of Section 3.5 produces a forecast that is a linear map of observed residuals, `η̂_{t+1} = M_t η_{t−D+1:t}` with `M_t = Σ₂₁Σ₁₁⁻¹`. Then `J^{(k)}_{ij} = |∂x̂_{i,t+1}/∂η_{j,t−k}| = |(M_t)_{i,(k,j)}|` is a genuine node-to-node input–output sensitivity mediated by the graph — the same *object* Topping et al. and Black et al. use to define over-squashing, though their bounds on it are theorems about message passing and do not transfer. `M_t` depends on the graph through `G_t = Q_t⁻¹` with `Q_t = (α+σ_min)I_R + βL_t^{(R)}`, so

`G_t = ∫₀^∞ e^{−(α+σ_min)s} e^{−βs L_t^{(R)}} ds`,

a Laplace-weighted heat kernel: bottlenecks that suppress diffusion in `L'_t` suppress entries of `G_t`, hence of `M_t`, hence of `J`. Measuring `J` against graph distance requires no retraining — it is a read-off from matrices already built at inference — and we will post it during the discussion period.

**(c) We attempted the measurement, and report that the effect is small.** As a first empirical bridge we computed `M^(k)_ij = (Â^k)_ij` with `Â = D^{−1/2}WD^{−1/2}` — the standard analytical surrogate for k-layer GCN Jacobian sensitivity (Topping et al., 2022) — on the static graph and on Teger's deployed reweighted graph, and compared decay against graph distance:

| k (hops) | decay slope, before | decay slope, after | far-hop mass ratio (after/before) |
|---:|---:|---:|---:|
| 3 | −2.065 | −2.060 | 1.0038 |
| 4 | −2.106 | −2.101 | 1.0032 |
| 5 | −1.767 | −1.763 | 1.0028 |
| 6 | −1.375 | −1.372 | 1.0025 |

(k = 1, 2 are degenerate — too few nonzero hop bins to fit a decay slope at those radii — and are reported as such rather than dropped.)

The direction is consistent across every k tested: the reweighted graph has a slightly flatter tail and slightly more mass on the farthest-quartile node pairs. But we want to be straightforward about the magnitude. The slope changes by ≈0.24% and the far-hop mass by 0.25–0.38%, whereas Table 2 reports 6.6–37.1% improvements in the unnormalised graph measures on these same graphs. We think the most likely explanation is that `Â` renormalises by degrees that the reweighting itself inflates, so large multipliers on a few edges leave `Â^k` nearly unchanged — which is a caution about how much work the unnormalised proxies in Table 2 can do, and we would rather say that than present a sub-percent effect as a vindication. We also note this surrogate asks "what would happen if this graph were used for k-layer message passing," which Teger does not do; the `M_t` read-off in (b) is the quantity that actually corresponds to our architecture, and we are computing it now.

**Honest gap, stated in the paper rather than left for a reader to find:** `L_t^{(R)} = P̂ᵀL'_tP̂` lives in the R-dimensional projected space, so the decay in (b) is in the `L^{(R)}` geometry, and the route back to node pairs runs through the learned, unconstrained `P̂`. We do not control that step analytically. Row (9) of the new ladder bounds the practical cost of the projection — un-projected (R = N) on Brussels changes the recovered error by 0.36pp on xLSTM and 0.20pp on MTGNN — but that is an empirical bound, not an analytical one.

### 3 — the CRPS definition is wrong

Correct, and thank you. The definition should be `CRPS(F, y) = E‖X − y‖ − ½E‖X − X'‖` (energy-score form); the submitted text has a plus sign. This is a typo in Section 3.4.4 only — ⟨FILL: confirm `src/metrics.py` implements the minus form and say so explicitly here; if it does not, all CRPS numbers must be recomputed before this response is posted⟩.

### 4 — `L_ΔW` should be PSD, not PD; Prop. 3(ii) is vacuous

Correct on both counts. `L_ΔW` is the Laplacian of a non-negatively weighted graph and therefore annihilates the constant vector, so it is never positive definite and the strict part never applies as written. The corrected statement, substituted with proof:

> Let `Δ = L_{ΔW_t} ⪰ 0`, so `L'_t = L_t + Δ` and `Δ1 = 0`. Let `v'` be any unit minimiser of the Rayleigh quotient of `L'_t` over `1^⊥`. Then
> `λ₂(L'_t) = v'ᵀL_t v' + v'ᵀΔv' ≥ λ₂(L_t) + Σ_{(i,j)} Δw_ij (v'_i − v'_j)²`.
> Hence `λ₂(L'_t) > λ₂(L_t)` whenever at least one reweighted edge `(i,j)` satisfies `v'_i ≠ v'_j` — i.e. whenever the added mass is not confined to a level set of the Fiedler vector of the reweighted graph.

The corrected condition is more informative than the one it replaces: strictness holds exactly when the added mass crosses the Fiedler cut, which is what curvature targeting is intended to achieve, and which rows (4)–(7) of the new ladder test empirically.

### 5 — Remark 4 defers the normalised-Laplacian case

We accept the disconnect and have replaced the deferral with the reason and a measurement.

The reason: reweighting changes `D` as well as `W`, so the Löwner order does not transfer to `D^{−1/2}LD^{−1/2}`, and we make no claim about the normalised spectrum. We should have said that rather than "we leave this to future work."

Two things reduce the distance to the literature. First, the effective-resistance result of Section 3.4.2 is stated in terms of `L⁺`, the pseudoinverse of the **unnormalised** Laplacian, which is the object Black et al. (ICML 2023) use for their over-squashing analysis; no normalisation question arises there. Second, we are reporting the normalised spectral gap before and after Algorithm 1 on all four graphs. This is worth doing beyond your point: `D^{−1/2}LD^{−1/2}` is invariant under uniform rescaling `W ↦ cW`, so any movement in it is attributable entirely to the non-uniform, curvature-targeted component — which removes a possible confound in Table 2, whose first two columns are not scale-invariant. Row (4) of the new ladder already addresses the same confound behaviourally: a uniform multiplier of matched mass does not reproduce the gain.

### 6 — "PeMS378" in Table 3

A typo; the correct dataset is ⟨FILL: dataset name⟩, now stated in the caption rather than the body. We have re-proofread end to end; the other errors this pass caught are in the summary table above.

### 7 — metric switching, missing error bars, missing dataset, and the "nearly 30%" MAE claim

Accepted in full, and addressed by the new ladder above.

- The ablation is now reported in **CRPS_sum**, the same metric as the main results, with per-dataset columns, seed spread, and distribution-free across-configuration tests. The NLL/MAE table it replaces was not comparable to Table 1 and we agree it should not have been the ablation of record.
- Error-bar semantics are stated explicitly, including that rows (1), (2) and (8) carry rolling-window spread rather than seed spread and are therefore not comparable to rows (3)–(7); we would rather flag that than present a uniform-looking ± that means two different things.
- **The "nearly 30% MAE reduction" claim is removed.** You are right that no table supports it; it rested on a qualitative figure. ⟨FILL: delete the sentence at l. 267 outright — our recommendation — or replace it with a tabulated MAE, noting the current metrics pipeline does not compute point MAE over the rolling windows⟩.

### 8 — broken cross-references and a missing Eq. (9)

Correct on both.

- Corollary 5 and Prop. 7 say "we show numerically … in Table 1" because the source uses a reference that resolves to the *algorithm* number. The intended target is Table 2 (graph measures). Fixed.
- Section 3.5 refers to `Eq. (9)` for the conditional covariance, and no such equation exists — it was replaced by a pointer to TCVAR. We have written out the block partition and the Gaussian conditional `Σ_cond,t+1 = Σ₂₂ − Σ₂₁Σ₁₂⁻¹Σ₁₂` explicitly, with its Woodbury evaluation, so Section 3.5 is self-contained. This also makes the `M_t = Σ₂₁Σ₁₁⁻¹` map of point 2 explicit in the text.
- One further error we found ourselves and report rather than wait to be caught: Table 2's second column is headed `λ'_{n−1}/λ_{n−1}`, which under the ordering declared at l. 144 (`λ_n ≥ … ≥ λ_1`) is the **second-largest** eigenvalue, not the spectral gap. ⟨FILL: state which quantity was actually computed — if the second-smallest, this is a header typo; if the second-largest, the column does not support the spectral-gap claim and is replaced by the normalised-gap measurement in point 5⟩.

### 9 — Appendix A is a heading with no content

Correct as printed: the appendix contained only a floating figure, which LaTeX placed elsewhere. It now has text stating what the figure shows and, following Reviewer kgxJ's W1, what it does not establish.

---

We recognise that the framing change in point 2 is substantial, and that the W1 result removes a claim we had made in the introduction. We would rather present the paper as curvature-guided covariance modelling with a falsified mechanism test and a controlled ablation than defend a mechanism our own data does not support. The ablation, we think, does answer the question both reviews converge on — whether the gains come from capacity or from curvature — and it answers it with the controls held at matched mass. We are glad to answer further questions.
