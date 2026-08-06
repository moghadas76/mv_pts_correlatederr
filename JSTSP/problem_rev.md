## Revised problem formulation

**End-cloud collaborative online traffic forecasting with budgeted OD distribution from an ANPR coordination platform under non-stationary conditions.**

Setup:

- A set of ANPR cameras $\{v\}$, each running a local forecaster $f_{\theta_v}$ on-board. Camera $v$ observes a local count/classification signal $x_v(t)$ continuously, and reads plates passing through its field of view.
- A central coordination platform M³ that aggregates plate reads from all cameras and reconstructs an OD estimate $\hat{M}(t)$ via cross-camera plate matching. $\hat{M}(t)$ is partial (only camera-to-camera trips are observable), noisy (plate-read failures), and non-stationary.
- Communication between camera $v$ and M³ has cost: uplink bandwidth from camera (plate reads, drift signals, queries), downlink bandwidth from M³ (OD slices, global event notifications, occasional model updates), and M³ server compute for handling queries.
- Each camera produces horizon-$H$ forecasts $\hat{x}_v(t{+}1{:}t{+}H)$ for its local count signal.

The research questions, sharpened:

1. **What should the camera decide locally vs. defer to M³?** The camera has its own observations and on-board GPU; M³ has the cross-camera view. When is local information sufficient, and when is the global view worth the round-trip cost?
2. **What should M³ push down proactively, to which cameras, when?** M³ sees regional events first; how should it distribute that information under a downlink budget?
3. **How do drift signals from the two layers compose?** A camera's local drift detector and M³'s global change-point detector see different things. The system needs to combine them coherently.

The bandwidth budget is now physically grounded: 4G LTE uplink limits matter, M³ query-rate matters, and "communication cost" is no longer an abstraction.

## Revised methodology

Three components, now mapped cleanly to the edge–cloud split.

**1. On-camera forecasting with OD context.** Each camera $v$ runs a compact forecaster on its Jetson TX2 (a small TCN, GRU, or quantized transformer — Jetson TX2 has roughly 1 TFLOPS, enough for inference but not heavy training). Input is local history $x_v(t{-}w{:}t)$ plus an OD context vector $c_v(t)$ supplied by M³ — typically the row and column of $\hat{M}(t)$ corresponding to $v$, possibly compressed. Without $c_v(t)$, the forecaster is a pure local baseline; the contribution is in how $c_v(t)$ is constructed, refreshed, and used.

**2. Two-sided, budgeted OD-sharing policy.**

- *Camera side (pull from M³):* the camera estimates the marginal value of a fresh OD slice for its own forecasting error. When value exceeds the round-trip cost (latency + bandwidth + M³ load), it queries. A natural and clean default trigger: **pull whenever the local drift detector fires.** A more sophisticated version learns a pull policy over time — a contextual bandit whose state includes residual statistics, time-of-day, and time since last pull.
- *M³ side (push to cameras):* M³ has the global OD view and can detect events that no single camera will see in isolation — a sudden rerouting, a corridor closure, a new traffic pattern. When M³ detects such a change, it proactively pushes updated OD slices to the affected cameras. This is exactly the value-add of having a coordinator: it sees what individuals can't.

The push/pull interplay is the multi-agent dynamic, and now it's grounded in a real asymmetry (cameras see local detail; M³ sees global structure), not a contrived one.

**3. Two-level drift detection.**

- *Local* (per camera): residual-based test on $\hat{x}_v$ vs. observed $x_v$ — CUSUM or a conformal detector, running cheaply on the Jetson. Triggers a pull.
- *Global* (at M³): change-point detection on $\hat{M}(t)$ — sudden shifts in OD entries, anomalous plate-match rates, structural breaks. Triggers a push.

Slow/fast separation still applies: cameras adapt locally on the fast clock (a few gradient steps on the local head when drift fires); M³ periodically retrains the shared backbone using accumulated data on the slow clock (hours/days) and broadcasts updated weights to the fleet. The fact that cameras have local storage (256 GB – 1 TB) and meaningful compute makes this practical without constant cloud round-trips.


## Learning problem 1: the local forecaster (definitely a learning problem)

This is the standard one. Each node $v$ has a supervised regression problem:

**Inputs at time $t$ (features):**
- Recent local counts: $x_v(t{-}w), x_v(t{-}w{+}1), \ldots, x_v(t)$ — say, last 24 hours of 5-min observations
- Calendar features: time-of-day, day-of-week, holiday flag
- OD context $c_v(t)$ (only in the OD-augmented variant): e.g., sum of inflows from upstream camera nodes, sum of outflows to downstream camera nodes, derived from the most recent OD slice the node has received

**Target:**
- Future counts: $x_v(t{+}1), \ldots, x_v(t{+}H)$ — predict 15, 30, 60 minutes ahead

**Loss:** standard regression loss (MAE or MSE) summed over the horizon.

**Model class:** lightGBM (or ridge regression, whichever you settled on).

**Training regime — and this is where the online angle comes in:**

You don't train once and freeze. You train in one of two ways:

- *Sliding-window retraining:* every $T$ ticks (or whenever drift fires), retrain on the most recent $W$ ticks of data. Cheap with lightGBM.
- *Warm-start incremental updates:* keep the current model and update it with new data without full retraining (lightGBM supports this with `init_model`; for ridge, recursive least squares does this in closed form).

Drift events are what make this an *online* learning problem rather than a one-shot supervised problem. The world changes, and the model has to keep up.

**This is the one place where supervised learning unambiguously happens.** Inputs, targets, loss, gradient updates, all standard.

## Learning problem 2: the OD pull policy (optional, depends on variant)

This is where the framing has been ambiguous, and I should have been clearer earlier. Among the four pull policies I listed, three are **rule-based** (no learning) and one is **learned**:

- **P0 (local-only):** no policy, no pulls. No learning.
- **P1 (periodic):** pull every $k$ ticks. $k$ is a fixed hyperparameter chosen to meet the budget. No learning.
- **P2 (drift-triggered):** pull when the drift detector fires. The detector itself has parameters ($h$, $k$), but you *set* these, you don't learn them. No learning.
- **P3 (value-of-information / learned policy):** this is the only one that's a learning problem.

If you include P3, you have a second learning problem. Two ways to formulate it, of increasing fanciness:

### P3a: Online value estimation (simple, recommended)

Each time node $v$ pulls an OD slice, observe:

- *Cost:* 1 unit of budget.
- *Benefit:* reduction in forecast error over the next $K$ ticks compared to what the local-only forecaster would have produced (you can compute this counterfactually because the local-only forecaster is cheap to run in parallel).

Maintain a running estimate of expected benefit conditioned on context (e.g., current residual magnitude, time-of-day, ticks since last pull). Pull when expected benefit exceeds a threshold $\tau$, where $\tau$ is calibrated so that the long-run pull rate equals the budget $B_v$.

This is barely "learning" — it's adaptive thresholding. But it's online, data-driven, and improves over time, and it's defensible to call it a learned policy. It has one or two hyperparameters and no neural network.

### P3b: Contextual bandit (fancier, optional)

Frame the pull decision as a contextual bandit:

- *Context $s_t$:* recent residual statistics, time-of-day, ticks since last pull, remaining budget today.
- *Actions:* {pull, don't pull}.
- *Reward:* error reduction over next $K$ ticks if pulled; 0 cost if not pulled; large negative reward if budget exceeded.

Train with LinUCB or Thompson sampling. This is a proper online learning problem with regret bounds you can cite.

**My recommendation: do P3a, not P3b.** P3a is enough to claim a learned policy, it's defensible, it's cheap, and it doesn't drag in RL machinery that distracts from the main story. P3b is a nice extension paragraph in "future work" or a brief ablation.

## Drift detection is *not* a learning problem

Worth being explicit because this confused you and I should have been clearer. CUSUM works like this:

- Compute residual $e_t = x_v(t) - \hat{x}_v(t)$.
- Standardize: $z_t = (e_t - \mu) / \sigma$ using historical residual statistics.
- Maintain a running sum: $S_t = \max(0, S_{t-1} + z_t - k)$.
- Fire when $S_t > h$.

Here $k$ and $h$ are **set by you** based on desired false-alarm rate and detection sensitivity — they are not learned from data via gradient descent. You can sweep them in ablations to show robustness, but that's tuning, not training.

Same for Page-Hinkley. These are statistical-process-control methods from the 1950s; they are deliberately simple and parameter-light.

If you wanted a *learned* drift detector you'd use something like a neural change-point model, but **don't.** It would add a third learning problem to your paper for marginal gain, and the simple detector is exactly what reviewers expect at this point in the methodology.

## So what's the full picture?

Your paper has:

1. **A supervised online regression problem (per node, definitely learned):** lightGBM forecaster, sliding-window retraining, optionally with OD context features. This is your forecasting workhorse.

2. **An adaptive thresholding / online policy problem (optional, lightly learned):** if you include the P3 variant, a simple online estimator of "value per pull" that adapts thresholds to meet a budget. If you only include P0–P2, there is no second learning problem and the paper is *purely* about (1) plus rule-based policies.

3. **A statistical test (not learned):** CUSUM/Page-Hinkley with pre-set parameters.

4. **A data pipeline (not learned, not yours):** M³'s OD reconstruction.

## Should you include P3 or not?

Honest answer: **including P3 makes the paper substantially stronger** because it gives you a second learned component and a cleaner "evolutive optimization" story for the special issue. The call's language about "autonomous self-optimization without human intervention" maps very directly onto a policy that adapts its pull thresholds online. Without P3, your paper is essentially "drift-triggered pulling is better than periodic pulling," which is true but feels thin.

P3a (online value estimation with adaptive thresholding) is the right level of ambition: it adds a learned policy without RL machinery, it gives you one more ablation row, and it makes the contributions list look meatier without much extra implementation cost.

## In one sentence

You are solving a **supervised online time-series forecasting problem** at each node, augmented by a **lightly-learned online policy** that decides when to refresh exogenous (OD) features under a communication budget, with a **non-learned statistical detector** providing the drift signal that the policy can use as one input.
