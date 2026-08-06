## Revised problem formulation

**Distributed online traffic forecasting under non-stationary OD drift with budgeted inter-agent communication.**

Setup: a road network $G = (V, E)$ where each node $v$ is an agent observing a local count signal $x_v(t)$. The OD tensor $M(t) \in \mathbb{R}_+^{|V| \times |V|}$ is latent, non-stationary, and only partially observable (probe vehicles, ANPR cameras, transit data). At each step, every agent must:

- produce a horizon-$H$ forecast $\hat{x}_v(t{+}1{:}t{+}H)$,
- decide what OD-derived information to exchange with which neighbors, under a global communication budget $B$,
- detect drift in its own predictive distribution and adapt online,

without centralized retraining.

The headline research question becomes: *given a fixed communication budget, what is the right policy for sharing OD information across nodes, and how should each node use incoming OD signals to adapt its local forecaster online?*


## Methodology

Three coupled components.

**1. OD-conditioned local forecaster.** Each node $v$ maintains a forecaster $f_{\theta_v}$ that takes as input its local history plus an OD-derived context vector $c_v(t)$ summarizing inbound/outbound flows from a subset of partner nodes. The baseline uses $c_v(t) = 0$; the contribution is in how $c_v(t)$ is constructed and kept fresh under a budget. The forecaster itself can be a standard temporal model (TCN, GRU, or a small spatio-temporal GNN over a $k$-hop neighborhood); the novelty isn't the architecture, it's the information flow.

**2. Budgeted OD-sharing policy.** Treat OD-entry transmission as resource allocation. At each communication round, each node $v$ has a budget $b_v$ (bytes or message count) and must select which OD entries — or compressed summaries — to push or pull. Three angles worth comparing as ablations:

- *Influence-based:* rank candidate OD entries by their estimated marginal contribution to local forecast error (e.g., a fast surrogate of $\partial \mathcal{L}_v / \partial M_{uv}$, or a leave-one-out approximation). Share the top-$k$ under budget.
- *Uncertainty-driven:* push OD information when the recipient's predictive uncertainty is high and the sender's signal is informative — a value-of-information criterion.
- *Learned policy:* each node runs a contextual bandit or lightweight RL agent whose actions are "which OD slice to share" and whose reward is downstream forecast improvement (signaled back from recipients via gossip).

The learned-policy version is probably the strongest paper story because it directly instantiates the call's "dynamic interactions among multi-agent systems" and "autonomous self-optimization without human intervention" language.

**3. Drift detection as communication trigger.** Per-node CUSUM (cumulative sum) on the forecast residual or more adcanced methods such as likelihood-ratio test on forecast residuals (or a more modern conformal/martingale-based detector). When a node detects drift, it (a) requests an updated OD slice from its highest-value partners, (b) refreshes $c_v(t)$, and (c) performs a few steps of online gradient descent on its head. No drift → no traffic on the wire. This couples the two main ideas: drift detection *causes* communication, communication *enables* adaptation. It's also the natural answer to "why does this scale?" — bandwidth is consumed only when needed.

A clean slow/fast separation works well: fast timescale is local online adaptation; slow timescale is occasional consensus on shared backbone parameters via gossip or FedAvg-style averaging.

## Evaluation that isolates the novelty

Avoid the trap of a binary "with OD vs. without OD" comparison — it conflates several effects. Instead, run a factorial design:

- **OD axis:** {no OD, full OD broadcast, budgeted OD}
- **Adaptation axis:** {frozen model, periodic retrain, drift-triggered online adaptation}
- **Policy axis (within budgeted OD):** {random, influence-ranked, learned}

The headline figure is a Pareto curve: forecast accuracy vs. communication cost (bytes/second or messages/round), with the learned drift-triggered policy ideally dominating.

Benchmarks: METR-LA, PEMS-BAY for standard comparability, plus at least one dataset with documented non-stationarity — COVID-era PeMS slices, or controlled synthetic regime shifts (event injection, partial sensor failure, gradual OD drift) so you can measure detection latency and recovery time, not just average error.

Metrics beyond MAE/RMSE: communication cost per unit error reduction; drift detection delay; recovery time after a regime shift; per-node fairness (worst-case node error, not just mean).

## Positioning Concerns

- **Differentiation from federated GNN traffic forecasting** (CNFGNN and its descendants). Your differentiators should be sharp and visible in the abstract: (i) drift-*triggered* communication rather than fixed-round federation, (ii) budgeted, content-aware OD sharing rather than parameter averaging, (iii) explicit treatment of non-stationarity with detection-and-recovery metrics.
