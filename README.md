# Two neural extensions of factor-augmented endogenous NQAR

These programs are self-contained examples for the factor-augmented NQVAR working paper. We consider **two separate methods**, with shared DGP and quantile-regression routines so they can be compared fairly. 

## Contents

| File | Purpose |
|---|---|
| `nqar_common.py` | Simulate factors, regressors, auxiliary variable, true time-varying spatial weights, and simultaneous outcomes; construct CCE averages; classical OLS auxiliary estimation; exact QR linear-program solver; Monte Carlo metrics. |
| `deep_cfqr_cce.py` | Shared neural network for the auxiliary equation, blocked time-fold cross-fitting of `epsilon_hat`, and node-specific CFQR--CCE. |
| `graph_attention_w.py` | Two-layer graph message-passing network with learned edge attention, masked normalization, graph training, and downstream CFQR--CCE. |
| `tests_smoke.py` | Small automated tests for probability weights, network support, residuals, and QR estimation. |
| `requirements.txt` | Python dependencies. |

## Installation and first runs

Use Python 3.10 or later. Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
python tests_smoke.py
```

**Neural control-function example** (three estimators: classical OLS CFQR--CCE; neural CFQR--CCE; an infeasible true-`epsilon` diagnostic):

```bash
python deep_cfqr_cce.py --n 30 --t 60 --replications 2 --epochs 80 --outdir example_deep
```

**Graph attention example** (three estimators: true-`W` oracle; fixed exponential kernel; learned GNN weights). By default it uses **supervised graph training** with observed network weights available in the simulation:

```bash
python graph_attention_w.py --n 30 --t 60 --replications 2 --epochs 80 --outdir example_gnn
```

**Exploratory graph learning without observed weights as labels** (learn attention from outcome pinball loss plus an optional kernel regularizer):

```bash
python graph_attention_w.py --n 30 --t 60 --train-mode outcome \
    --anchor-strength 0.10 --epochs 150 --outdir example_gnn_outcome
```

This last mode does **not** solve structural identification of the unknown network: joint graph/coefficient fitting can produce prediction-friendly, but structurally incorrect, weights.

Recommended for modest CPU memory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python deep_cfqr_cce.py --n 30 --t 60
```

On Windows simply omit the environment-variable prefix.

**Full or custom experiment (can be computationally expensive):**

```bash
python deep_cfqr_cce.py --n-grid 50 60 100 200 --t-grid 50 60 100 200 \
    --replications 500 --epochs 250 --auxiliary-rule nonlinear --factors 2 \
    --proxy sbar_new --lags 0 --outdir full_deep

python graph_attention_w.py --n-grid 50 60 100 --t-grid 50 60 100 \
    --replications 500 --epochs 250 --network-rule nonlinear --factors 2 \
    --proxy sbar_new --lags 0 --outdir full_gnn
```

These examples describe **how** to run large designs; they are not precomputed results. Full grids with repeated training of neural networks and hundreds of LP fits per replication may take substantial compute resources. Set `--replications 2` or `10` first for debugging.

## DGP:

One outcome regressor `Xy` and one auxiliary regressor `Xz` are used, matching your recent MATLAB setup. Factors follow AR(1) with phi=0.70; `Xy` idiosyncratic innovations have AR(1) coefficient 0.80; `Xz` 0.40; `epsilon` 0.30. Node coefficient centers are `Gamma=0.25`, `lambda=0.35`, `rho=0.15`, `delta0=0.8`, and `beta1=-1`. Structural heterogeneity around the centers is allowed. The simulation uses an expanding 2D lattice (`d0=1.25`, `dc=1.75`) with L-infinity distance and zero diagonal. Default number of factors is `m=2` (change with `--factors 3`). It uses Gaussian outcome innovations shifted by `Phi^{-1}(tau)` to make their **unconditional** tau quantile zero.

The auxiliary DGP is

`Z_it = beta_zi Xz_it + gamma_zi' f_t + epsilon_it`.

The default is `--auxiliary-rule linear`, matching the original auxiliary equation. If you select `--auxiliary-rule nonlinear`, an *additional common nonlinear factor component* is deliberately inserted:

`Z_it += a_i [0.55 sin(f_1,t) + 0.35 f_1,t f_2,t]` (when m>=2).

This is an **extension/violation of the original linear auxiliary specification**, useful for testing whether the neural nuisance model helps under misspecification. The outcome equation remains linear in the latent factors.

The network DGP is, by default in the deep-control-function script,

`W_ijt ∝ exp(-|Z_it-Z_jt|) 1{distance(i,j)<=dc, i!=j}`.

The graph-attention script defaults to an intentionally more complicated true network:

`W_ijt ∝ exp[-|Z_it-Z_jt|-0.65|Xz_it-Xz_jt|+0.40 sin(Z_it+Z_jt)] 1{distance<=dc, i!=j}`.

This lets a flexible learner have something nontrivial to learn. Set `--network-rule kernel` in either program to use the original kernel DGP.

Outcome generation respects contemporaneous simultaneity by solving, at each time,

`(I - diag(Gamma_i) W_t) Y_t = diag(lambda_i) Y_(t-1) + diag(rho_i) W_(t-1)Y_(t-1) + beta_i Xy_t + gamma_y f_t + delta_i epsilon_t + u_t`.

This is not simulated by substituting lagged Y for the contemporaneous network term. An initial 100-period burn-in is discarded (the user's MATLAB uses 200; set `--burn 200` for closer matching).

## The CCE regressor choices

In `cce_design(data, proxy, lags)`, choose any of:

- `rbar`: `[Ybar_t, Xybar_t, Zbar_t, Xzbar_t]`.
- `rbar_new`: `[Ybar_t, Xybar_t, Zbar_t]`.
- `sbar`: `[Xybar_t, Zbar_t, Xzbar_t]`.
- `sbar_new`: `[Xybar_t, Zbar_t]`.

The **active** source MATLAB function `make_cce_design_endog.m` currently has `base=[xybar,zbar]; R=base` with its lag-builder commented out, even though its README describes full `rbar` and `pT=ceil(T^(1/3))`. To reproduce the active code rather than comments, the Python default is `--proxy sbar_new --lags 0`. For lagged CCE use `--lags 1`, `--lags 3`, or `--lags auto`. With `--lags auto`, `pT=ceil(T^(1/3))`. Lags are arranged as `[base_t, base_(t-1),...,base_(t-pT)]`; initial missing lags are zero-padded and the outcome QR discards the first `max(1,pT)` periods. Larger lag orders can cause small-T quantile regressions to be poorly conditioned or unidentified; use longer T.

## Method A: neural control function

1. Form CCE averages `R_t` using only observed variables.
2. Hold out a contiguous block of time periods; remove an extra `--gap` times on either side from the fitting sample.
3. Fit the **shared** DNN to the remaining panel:

   `Z_it = alpha_zi + beta_zi Xz_it + a_zi' h_phi(R_t) + epsilon_it`,

   `h_phi` is one common multilayer perceptron; `alpha_zi`, `beta_zi`, and loading `a_zi` are node-specific trainable coefficients. Its output dimension is `--latent` (default 4) and hidden width `--hidden` (default 20).
4. Predict only the held-out times and calculate residuals `epsilon_hat_it=Z_it-Z_hat_it`.
5. Repeat for `--folds` contiguous blocks, giving out-of-fold residuals for every time. The OLS first-stage benchmark uses **the same held-out blocks and time gap**, for fair comparison.
6. Use `epsilon_hat_it` in the **ordinary node-specific linear quantile regression**:

   `Y_it = alpha_i + Gamma_i (W_tY_t)_i + lambda_i Y_i,t-1 + beta1_i Xy_it + rho_i (W_(t-1)Y_(t-1))_i + Delta_i'R_t + delta_i epsilon_hat_it + error_it`.

The DNN does **not** estimate structural `Gamma`, `lambda`, `rho`, `beta1`, `delta`: QR estimates them as in your paper. Cross-fitting reduces own-observation overfitting but time-series dependence requires more than an arbitrary finite gap to justify asymptotic theory. These folds use training observations on *both sides* of a held-out time block, and are for econometric cross-fitting rather than real-time forecasting. DNN training employs MSE, Adam, weight decay, gradient clipping, a training-time validation subset, and early stopping. `epsilon` prediction error is evaluated against the true epsilon **only in the simulation**.

The three rows are `CFQR_CCE_OLS`, `Deep_CFQR_CCE`, and `Oracle_epsilon_CCE`. The oracle is a diagnostic, not a feasible estimator.

## Method B: GNN / graph attention

1. Construct the **spatial neighbor set** by the same L-infinity lattice cutoff. The GNN receives `Z_it` and `Xz_it`, but **never contemporaneous Y_t** as features used to form W.
2. Apply **two graph message-passing layers**: each node transforms its own attributes and the mean attributes of its admitted neighbors. The result is node embedding `h_it`.
3. For every admissible directed edge `(i,j)`, build an edge feature from `|h_it-h_jt|`, `h_it*h_jt`, `|Z_it-Z_jt|`, `|Xz_it-Xz_jt|`, `sin(Z_it+Z_jt)`, and lattice distance.
4. An attention network learns an edge logit `s_ijt`. The learned neighbor weight is

   `W_hat_ijt = exp(s_ijt) / sum_{k admitted} exp(s_ikt)`,

   with `W_hat_iit=0` and `W_hat_ijt=0` for forbidden spatial links.
5. **Training signal, supervised mode (default):** In these *simulations*, true `W_t` is observable to the trainer because we generated it. The first 70% of dates are used as labeled network observations; a subset of those dates is used for neural early stopping; the last 30% is a held-out **network prediction** test. The graph objective is the cross-entropy between observed row weights and learned row probabilities. The output reports both network MAE and KL divergence on held-out dates.
6. **Alternative outcome mode** (`--train-mode outcome`): Instead of network labels, the attention weights and provisional node-specific coefficients are optimized jointly to reduce observed-outcome quantile loss. A KL penalty toward the known exponential Z-kernel can be set by `--anchor-strength` (default 0.10). True W is **not used for training** in this mode; it is used only to evaluate recovered weights in the simulation. This mode is a proof of computation, not an identified estimator of W or Gamma.
7. Once the graph network is trained, construct learned weights for every sample period. Then refit node-specific outcome CFQR--CCE with these learned weights and the ordinary CCE-OLS control function, separately from the provisional neural outcome-fit coefficients.

The outputs compare `True_W_oracle`, `Known_kernel_reference`, and `GNN_attention_W`. **Supervised** training needs measured weights/edge intensities in empirical data, for example actual transaction counts. If only `Z_t` is available and `h(Z_it,Z_jt)` is assumed known, using a DNN merely to reproduce this kernel is redundant. The **outcome** mode does not require measured W, but learning W jointly with Gamma from a conditional-quantile fit is generally not identified without additional restrictions; its learned adjacency may simply improve in-sample fit.

This is a graph neural network because two layers explicitly aggregate information from neighboring nodes before the attention layer scores links. Its edge weights are allowed to vary freely with the observed node attributes over time, subject to the original geographic support and row-normalization restrictions.

## Output files and statistical definitions

Both programs write:

- `replication_results.csv`: one row per simulated replication, N,T and estimator.
- `monte_carlo_summary.csv`: means across replications and pooled Monte Carlo RMSE.
- `node_estimates_N..._T....csv`: one set of node-level true/estimated coefficients for the first replication in each cell.

The **graph** script also writes a visualization `W_comparison_N..._T....png` and compressed snapshots of true/kernel/learned W matrices `W_snapshots_N..._T....npz`.

For an estimator of parameter `theta_i` and simulation replication `s`, the code reports:

- `Bias_theta = mean_{s,i}(theta_hat_(i,s)-theta_(i,s))`
- `AbsBias_theta = mean_{s,i}|theta_hat_(i,s)-theta_(i,s)|`
- `MSE_theta = mean_{s,i}(theta_hat_(i,s)-theta_(i,s))^2`
- `RMSE_theta = sqrt(MSE_theta)`.

The metrics are provided for `Gamma`, `lambda`, `rho`, `beta1`, `delta`. They evaluate **node-wise coefficient estimation**, not mean-group estimation.

The node-wise QR optimizer is an **exact linear-program formulation of pinball loss** solved by SciPy's HiGHS method (rather than treating gradient descent as exact QR). Each row uses `t>=max(1,pT)` to ensure `W_(t-1)Y_(t-1)` is available from the fitted W series.

## Econometric interpretation and limitations

**1. Two sources of endogeneity.** The endogenous W through epsilon and simultaneous feedback through `(W_tY_t)_i` are not the same problem. Control-function residuals that account for the first do **not automatically** solve the second. This is a simulation/implementation comparison, not established causal inference. Correct IV/identification restrictions or structural assumptions must be supplied for valid estimation of Gamma under simultaneity.

**2. Generated nuisance estimates.** No uniform NED convergence rates, neural equicontinuity, or asymptotic normality are proved here. Standard neural validation or a fixed time embargo does not supply these proofs.

**3. Learned graph identification.** Supervised graph training uses true graph weights as observed labels in the simulation; this is not available automatically when W is unobserved in real data. The optional outcome-trained network does not use true-W labels, but joint recovery of W and Gamma from Y alone remains an unidentified/noncausal fitting exercise unless extra restrictions or instruments are justified. A better held-out outcome fit would not prove structural weight recovery.

**4. Nonlinear alternative DGPs.** Setting `auxiliary_rule=nonlinear` or `network_rule=nonlinear` deliberately moves outside the linear/known-kernel assumptions of the original model. These options are for robustness and potential neural advantages, not exact numerical reproduction of the existing paper.

**5. Simulation sizes.** N=30,T=60 is an easy first test, but complex DNNs can overfit when T is small. Test larger T and a range of N, report both nuisance errors and structural parameter errors, and compare multiple seeds. More epochs or a wider network does not guarantee smaller structural bias.
