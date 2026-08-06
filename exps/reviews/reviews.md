Official Review of Submission22681 by Reviewer kgxJ
Official Reviewby Reviewer kgxJ07 Jul 2026, 17:24 (modified: 23 Jul 2026, 17:50)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer kgxJRevisions
Summary:
This paper proposes Teger, a backbone-agnostic probabilistic framework for spatio-temporal traffic forecasting that models the spatial correlation structure of forecast residuals. The core idea is to identify bottleneck edges via Balanced Forman curvature and reweight them (without adding or removing edges), then integrate this into a low-rank-plus-diagonal covariance head that remains tractable through the Woodbury identity. The paper provides theoretical support for over-squashing alleviation in terms of spectral gap, effective resistance, and local bottleneck conductance, and shows consistent improvements in CRPS over TCVAR and LASER across four backbones (LSTM, Transformer, xLSTM, MTGNN) and four traffic datasets (PeMS03/04/07, Brussels).

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
S1. Approaching spatial residual covariance modeling in traffic forecasting through the lens of over-squashing and graph curvature is a novel angle not explored by prior work.

S2. Consistent performance improvements across four real-world datasets and four backbones.

S3. The framework is computationally tractable and backbone-agnostic — the covariance head integrates with any autoregressive encoder without modifying its architecture, and the Woodbury identity keeps inference cost low.

S4. The paper is transparent about the scope of its claims — it explicitly acknowledges that edge reweighting is a strictly weaker intervention than LASER/SDRF's structural rewiring, and that Section 4.3's interpretability analysis is qualitative rather than a quantitative validation. This kind of self-aware scoping is refreshing and helps calibrate the paper's claims appropriately.

W1. The motivating claim that over-squashing is intrinsically linked to spatial residual covariance (lines 37, 51-57) is intuitively plausible but not established. Propositions 3 and Corollary 5 hold for any monotonic edge-weight increase, not specifically curvature-guided selection, so they don't justify why curvature is the right lens. Comparing residual correlation at bottleneck vs. non-bottleneck edges would help support this claim.

W2. While using SDRF's curvature to identify bottleneck edges is reasonable, the specific reweighting formula is not justified— why this form over alternatives is not discussed. Since Propositions 3/Corollary 5 hold for any monotonic reweighting regardless of form, they don't ground this specific heuristic either — so it's unclear whether the performance gains actually stem from alleviating over-squashing, or simply from adding capacity to the covariance model.

W3. The paper frames SDRF and LASER as the two representative rewiring paradigms and builds its method directly on SDRF's curvature signal, yet only compares against LASER experimentally (Table 1), omitting SDRF — the more methodologically relevant baseline.

W4. The description of LASER (lines 45-46) seems inaccurate. LASER's actual contribution is preserving locality and sparsity while reducing over-squashing, not adding long-range shortcuts at their expense. This misrepresents the closest related work.

W5. It's unclear what the edge weight actually represents. Appendix J defines it via a static quantity (road-network/GPS distance), which doesn't vary with t. Yet Section 3.3.1 and Algorithm 1 attach a time index to the adjacency matrix without defining what varies over time or why. This leaves the paper's "dynamic graph" claim somewhat underspecified.

W6. Proposition 7 requires the rewiring to touch only edges crossing the boundary of S, but Algorithm 1 reweights edges globally by curvature. Does the proposition's hypothesis actually hold for graphs produced by Algorithm 1?

W7. Several minor issues affect readability: the acronym "OQ" (presumably "over-squashing") is never defined; the Kronecker product symbol is used without definition; Appendix A ("Motivation") has a title but no content; and it's unclear whether "PeMS378" (line 293) is a typo, and if so, which dataset it refers to.

Quality: 3: good
Clarity: 3: good
Significance: 3: good
Originality: 3: good

Summary:
This paper proposes Teger, a backbone-agnostic probabilistic module for spatio-temporal forecasting that models the spatial and temporal correlation of forecast residuals, motivated by the observation that prediction errors in traffic networks are correlated across sensors rather than independent. Attached to the latent state of any autoregressive encoder (LSTM, Transformer, or xLSTM), Teger places a curvature-aware spatial covariance inside a low-rank-plus-diagonal noise head, factorizing the residual covariance into a temporal kernel-mixture matrix and a graph-based spatial factor through a Kronecker structure that keeps likelihood evaluation tractable via the Woodbury identity; bottleneck edges identified by Balanced Forman curvature are reweighted, projected into a low-dimensional factor space, and turned into a valid covariance, with an inference-time refinement step that adds node-wise volatility scaling. The paper supports this design with a theoretical analysis linking the reweighting to spectral-gap improvement, reduced effective resistance, and local bottleneck alleviation, and with experiments on four real-world datasets across four backbones showing consistent CRPS improvements, complemented by interpretability and ablation studies.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Weaknesses:

The motivation is questionable. The four images in Figure 5(b) are almost identical, and thus fail to exhibit both the contemporaneous spatial structure and the cross-lag temporal persistence in the traffic residuals.
The title of the paper promises mitigation of over-squashing, yet the method never performs the multi-layer message passing in which over-squashing is actually defined; the graph is used only to parametrize a covariance matrix. The theoretical results merely show that edge reweighting improves standard graph proxies such as spectral gap and effective resistance, without ever connecting these quantities to the Jacobian decay that constitutes over-squashing. In this sense, the over-squashing framing serves as borrowed terminology rather than a phenomenon that the model genuinely addresses.
The definition of CRPS is wrong. There should be a minus sign instead of a plus sign in the correct definition of CRPS.
 should be positive semi-definite instead of positive definite. In Proposition 3(ii), the condition is therefore vacuous, and the strict part of the proposition is never applicable.
Remark 4 admits that the normalised Laplacian case is not handled, yet message-passing and over-squashing analyses are typically built on the normalised Laplacian, which adds another layer of disconnect between the theory and the literature it claims to connect to.
Table 3 reports results on "PeMS378 data," but the only datasets described (Table 5) are PeMS03, PeMS04, PeMS07, and Brussels. This is either a typo or carelessness, and it signals weak proofreading.
The main results (Tables 1 and 7) use CRPS, whereas the ablation (Table 3) switches to NLL and MAE, with no standard deviations and no clear dataset. The text also claims a "nearly 30%" MAE reduction, but MAE does not appear in the main results table, so this number is not rigorously supported and rests only on the qualitative figure (Fig. 3).
Corollary 5 and Proposition 7 both state "we show numerically … in Table 1," but the relevant graph measures are in Table 2; Section 3.5 also refers to "Eq. (9)" for the conditional covariance, yet no such equation or derivation is actually provided (it is replaced by "same as TCVAR").
Appendix A ("Motivation") is only a heading with no content.
Quality: 1: poor
Clarity: 1: poor
Significance: 1: poor
Originality: 1: poor
Questions:
None.

Limitations:
Yes

Rating: 1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
