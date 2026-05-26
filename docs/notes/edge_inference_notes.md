# Scalable inference of missing edges in the GSNN structural graph

> Status: design notes — chalk-talk synthesis. No production code yet. Intended to be a complete handoff for future development.

## 1. Motivation

GSNN's inductive bias comes from the structural graph `G`. When `G` is incomplete, the model has no mechanism to compensate: there is no path along which information can flow between unconnected nodes. This makes *edge inference* — discovering missing edges from training signal — one of the highest-leverage problems for GSNN.

Two single-side cases are already implemented:

- `gsnn/optim/InputEdgeInferrer.py` — infers `input → function` edges by learning a weight on candidate input edges that, when added via `e0`, reduces validation loss.
- `gsnn/optim/OutputEdgeInferer.py` — infers `function → output` edges by fitting per-(function-node, output) linear maps from intermediate activations to outputs.

Both scale as O(N · K) where one side is anchored to a small set (inputs or outputs). The hard case — inferring `function → function` edges — has been left open because, naively, it scales as O(E²): roughly $10^{10}$ candidate pairs at typical biological scales of $E \sim 10^5$. See `notes.md` 07/18/25 entry.

This document formalizes three families of approximation, in increasing order of complexity and capacity, that bring this problem into the tractable regime.

---

## 2. The intractable target

The "ideal" edge-inference object is described in `notes.md` 07/18/25:

$$\text{score}(e_{ij},\, e_{lm}) \;\propto\; \big| \mathrm{corr}_b\!\big(h_{ij},\; \nabla_{h_{lm}} \mathcal{L}\big) \big|$$

In words: *edge `ij`'s latent state explains the residual gradient on edge `lm`* ⟹ connecting `i` (or `j`) to `m` would help. This is the right object, but quadratic in `E`. Both the memory ($E^2$ scores) and the compute (per-batch correlations) are infeasible at biological scale.

The three approximations below all reduce this to operations linear or quadratic in `N` (function nodes) rather than `E` (edges), and replace edge-level state with node-level summaries. `N` is typically one to two orders of magnitude smaller than `E`.

---

## 3. Why we can hope this works

Three structural observations underlie all three approaches:

1. **Most candidate edges share a node.** Edges `(i,j)` and `(i,k)` both involve node `i`; their relevance to downstream gradients can be largely captured by a per-node summary of `i` rather than per-pair representations.
2. **Activation × gradient is a first-order Taylor probe.** Adding a new edge from node `i` to node `j` with weight `w` produces, to first order:
   $$\Delta \mathcal{L} \;\approx\; w \cdot \langle z_i^\ell,\, \nabla_{z_j^\ell} \mathcal{L} \rangle.$$
   So *activation on the source × negative gradient on the target* is the natural marginal-utility signal.
3. **GSNN's `NodeMLP` is the only parameter-sharing mechanism across function nodes.** Without it, per-node hidden states live in node-specific coordinate frames and cannot be compared directly. With it (and group normalization), there is at least a chance that per-node representations become semantically comparable across nodes. See `gsnn/models/GSNN.py` lines 533–548.

The three tiers below trade off compute against how much of the per-channel structure they preserve.

---

## 4. Tier 0 — magnitude correlation

The cheapest formulation. Reduce every node's per-layer state to a scalar via a fixed magnitude reduction.

### 4.1 Formalization

For function node `i`, layer `ℓ`, sample `b`:

$$x_i^{\ell,b} \;=\; \big\|z_i^{\ell,b}\big\|_1, \qquad y_i^{\ell,b} \;=\; \big\|\gamma_i^{\ell,b}\big\|_1$$

where $z_i^{\ell,b} \in \mathbb{R}^{C_{\text{pn}}}$ are the per-node channel activations at the **pre-norm hook** (post-``lin_in``, before the ResBlock normalization layer; ``ResBlock._last_pre_norm_activation``) and $\gamma_i^{\ell,b} = \nabla_{z_i^{\ell,b}} \mathcal{L}$ is the corresponding loss gradient. Using pre-norm activations preserves magnitude variation even when the model uses layer or RMS normalization. L2 or squared L2 work equally well; L1 is gentlest on outliers.

Score (directed, `i → j`), per adjacent layer pair with activation at `n−1` and gradient at `n`:

$$s_n(i \to j) \;=\; \mathrm{corr}_b\!\big(x_i^{n-1,b},\; y_j^{n,b}\big)$$

aggregated across pairs by mean / max / per-pair head into a final $N \times N$ matrix.

### 4.2 Implementation pattern

Identical machinery to `OutputEdgeInferer.evaluate` (lines 273–405): accumulate streaming sufficient statistics per (layer, source, target) — `sum_x`, `sum_y`, `sum_x²`, `sum_y²`, `sum_xy`, and `n` — and compute correlations + Fisher-z p-values + BH-FDR at the end. The accumulators are 4 × L × N² floats (≈ 2 GB at N=10k, L=5); per-batch updates are O(B · N²) outer products on GPU offloaded to CPU.

### 4.3 What this buys

- **No alignment problem.** Scalars are always comparable across nodes — coordinate frames are irrelevant by construction.
- **Zero parameters.** Pure analysis pass over a trained model.
- **Calibrated significance.** Standard correlation inference with N² hypothesis correction.
- **Reuses existing code structure.** Streaming-statistics pattern is already implemented.

### 4.4 What this costs

- **No sign.** Cannot distinguish activator from inhibitor. Edges are unsigned candidates.
- **No channel structure.** Two nodes with identical `|z|₁` but orthogonal channel patterns are indistinguishable.
- **Confounded co-activity.** Nodes sharing an upstream cause have correlated `|z|`; their `corr(|z_i|, |γ_j|)` will be similar to anything node `j`'s gradient correlates with. Produces *clusters* of candidate edges. Fix: partial correlation, or stagewise greedy selection, or use this as a screen feeding a multivariate verification pass.
- **Direction comes only from the activation-vs-gradient asymmetry.** $s(i \to j) \neq s(j \to i)$ because $x$ and $y$ are different quantities, but the geometric direction signal of the inner product is lost.

### 4.5 The gradient-absorption issue

At equilibrium, if `i → j` already exists in `G`, the model is locally optimal w.r.t. that input, so $|\gamma_j|$ is *small* exactly when $|z_i|$ is large — the gradient has been absorbed. **Existing useful edges therefore score near zero or negative, not high.** This means:

- The correct null hypothesis is "score ≈ 0 for in-graph edges, positive for missing-but-useful edges."
- Validation by "do known edges score high?" is the wrong test.
- Validate via **edge-masking**: zero a sampled subset of existing edges through `GSNN.forward(x, edge_mask=...)` (already supported), recompute the score, and check whether the masked edges re-emerge as positives. The model is no longer at equilibrium for masked edges, so the absorption goes away.

---

## 5. Tier 1–3 — dual encoder

A controlled progression that recovers channel structure at the cost of introducing learned parameters.

### 5.1 Formalization

Two shared, node-agnostic encoders $E_a,\, E_g : \mathbb{R}^{C_{\text{pn}}} \to \mathbb{R}^d$:

$$a_i^\ell = E_a(z_i^\ell), \qquad g_i^\ell = E_g(\gamma_i^\ell)$$

Edge score:

$$s_\ell(i \to j) = \langle a_i^\ell,\, -g_j^\ell \rangle$$

The sign of `-g_j` matches the first-order Taylor argument: $\langle z_i, -\nabla_{z_j} \mathcal{L}\rangle > 0$ means adding an `i → j` contribution would reduce loss.

Full pairwise scoring is O(N²d), or O(Nk) with FAISS / HNSW maximum-inner-product search.

### 5.2 The progression

| Tier | $E_a, E_g$ | Output dim | Parameters | What's recovered |
|---|---|---|---|---|
| 0 | $\|\cdot\|_1$ | 1 | 0 | nothing beyond magnitude |
| 1 | reuse `NodeAttention._last_alpha` | 1 | 0 (existing) | learned, signed scalar (if `node_attn=True`) |
| 2 | learned MLP `ℝ^C → ℝ` | 1 | small | learned scalar reduction |
| 3 | learned MLP `ℝ^C → ℝ^d` | d (≈16) | $O(d \cdot C_{\text{pn}})$ | full channel structure via low-rank inner product |

This is the smallest controlled set of upgrades that lets you measure marginal value at each step.

### 5.3 The alignment problem

For tiers 2 and 3, $\langle a_i, -g_j \rangle$ only means something if $a_i$ and $a_j$ live in a shared latent space. `NodeMLP` is *necessary* (it's the only shared-parameter component) but *not sufficient* (the inputs to `NodeMLP` are still in node-specific coordinate frames).

Fixes, roughly in order of preference:

1. **Concatenate a learned per-node identity embedding $d_i$** before the shared encoder: $a_i = E_a([d_i;\, z_i])$. Identity embeddings ground each node in a common frame; the encoder learns how identity interacts with state. This is exactly the transformer's token-embedding trick (see §6), borrowed back to the dual-encoder setting.
2. **Anchor with pathway co-membership** (the regularizer in `functional_dis_and_similarity.md` §5). If pathway co-members are pulled together in $a$-space, the resulting alignment scaffolding is reusable for edge inference.
3. **Diagnostic probe before committing.** Train a vanilla GSNN with `NodeMLP` + `groupbatch`; fit a linear map $z_j \approx M z_i$ across pairs. If mean test $R^2 \gg 0$, NodeMLP-induced alignment exists and the dual encoder has something to work with. If not, fix (1) or (2) is required.

### 5.4 Training objectives

The encoders need a training signal. Four options, increasing in reliability:

- **(A) Zero extra loss.** Encoders are post-hoc analysis tools applied to a trained model. Quality depends entirely on implicit alignment from norm + NodeMLP. Cheapest, probably weakest signal.
- **(B) Masked-edge reconstruction (InfoNCE).** Mask a random subset of existing edges each batch via `edge_mask`; train encoders so masked edges score higher than random non-edges. Self-supervised, no external supervision needed, sidesteps the gradient-absorption issue by manufacturing non-equilibrium positives.
- **(C) Pathway prior.** Use `PathwayLatentRegularizer` (planned in `functional_dis_and_similarity.md`) to pull pathway co-members together in $a$-space. Provides alignment scaffolding; does not directly train edge inference.
- **(D) Distillation from `InputEdgeInferrer` / `OutputEdgeInferer`.** Use the existing inferrers' scores as labels on a tractable subset, then extrapolate via the encoders.

Recommended combination: **(B) + (C)**. Masked reconstruction is the primary signal; pathway prior makes sure the coordinate system is shared so that signal generalizes.

---

## 6. Tier 4 — transformer surrogate

The full-strength generalization. Per-node tokens, multi-head attention, MLM training.

### 6.1 Formalization

Treat each function node as a token. Initial token representation:

$$h_i^{(0)} \;=\; E_{\text{id}}[i] \,+\, W_a\, z_i^\ell$$

where $E_{\text{id}} \in \mathbb{R}^{N \times d}$ is a learned per-node identity embedding (analogous to BERT's word embedding) and $W_a$ is a shared projection of activations into the embedding space. No positional encoding — the structural graph is unordered, and permutation invariance is the desired inductive bias.

Stack $L_T$ transformer encoder layers. Per layer, multi-head self-attention computes:

$$A^{(h)}[i,j] \;=\; \mathrm{softmax}_j\!\left(\frac{\langle W_q^{(h)} h_i,\; W_k^{(h)} h_j \rangle}{\sqrt{d_h}}\right)$$

**The attention matrix `A` is the edge-inference object.** Top-k entries per row are the inferred parent set of node `i`. Multi-head gives multiple "relation types" simultaneously, which can be aggregated by mean or learned head selection.

### 6.2 Training: masked language modeling for nodes

Random subset of nodes have their activation tokens masked: $h_i^{(0)} \leftarrow E_{\text{id}}[i] + W_a \cdot \mathbf{0}$ for `i ∈ M`. Transformer must reconstruct $z_i$ from the unmasked context.

The transformer can only succeed at reconstruction by routing information through its attention pattern. The attention weights for masked node `i` therefore reveal which nodes the model has learned to use as predictive context — which is exactly the inferred neighborhood of `i`.

Optional auxiliary objective: predict the GSNN's actual gradient $\gamma_i$ from $z_i$ (gradient distillation). This forces the attention pattern to match the GSNN's local dynamics, not just the activation co-occurrence statistics.

### 6.3 What this buys

- **Alignment is solved by construction.** Per-node identity embeddings put every node in a shared latent space without external priors. This was the unresolved foundation problem of the dual encoder.
- **Attention is the right object.** $A[i,j]$ is structurally identical to an edge probability. Stacking layers gives multi-hop reasoning, matching GSNN's L-layer information propagation.
- **MLM sidesteps gradient absorption.** Reconstruction-from-context is a counterfactual probe, not an equilibrium probe. Existing edges are not penalized for being satisfied.
- **Free deconfounding.** Softmax attention competes across the row, so two redundant nodes (shared upstream) will *split* the attention rather than both saturating — a built-in partial-correlation effect.
- **Permutation invariance matches the graph.** No artificial ordering imposed.

### 6.4 What this costs

- **Compute.** Standard self-attention is $O(N^2 d)$ per layer per head. At N=10k, $d_h$=64, h=8, $L_T$=4, B=32: ~10 TFLOPs per batch. Tractable.
- **Memory.** Dense attention matrix per head per layer per sample: at N=10k, B=32, h=8, $L_T$=4 → ~100 GB peak. **Not tractable without sparse attention** (BigBird / Longformer / FlashAttention with masking) or chunked attention computation.
- **Surrogate gap.** The transformer is a learned proxy. Its inferred edges depend on its training distribution and capacity; out-of-distribution generalization is not guaranteed.
- **MLM objective mismatch.** The attention pattern that minimizes MLM loss is not necessarily the edge structure we want. Heads can learn to attend to "easy-to-reconstruct" nodes (low-entropy targets) rather than causal parents. Regularization (entropy penalty, sparsity penalty, alignment to known partial structure) helps.
- **Sparsity prior mismatch.** Biological graphs are sparse; vanilla attention allows global connectivity. L1 / entropy regularization on attention weights, or top-k attention, is necessary to bias toward sparse interpretable patterns.
- **A second model to train.** Adds engineering complexity and a separate training loop. Failure modes are now coupled across the GSNN and the surrogate.

### 6.5 Variants worth flagging

- **Linear attention / Performer.** Reduces compute to O(N · d²), recovers tractability at the cost of explicit attention patterns (the attention matrix is never materialized). Loses the direct edge-inference readout, so probably wrong here.
- **Sparse attention with `G` as prior.** Restrict attention to k-hop neighborhoods in the current `G`. Trades the discovery range against compute. Reasonable middle ground for incremental graph refinement.
- **Cross-attention over (z, γ) pairs.** Replace self-attention on `z` with cross-attention from `z`-queries to `γ`-keys/values. Closer to the dual-encoder semantics: "which nodes' gradients does my activation explain?"
- **Graph-attention hybrid.** Combine learned attention with structural masking (allow attention to existing neighbors + a sampled set of candidates per step). Reduces compute and keeps the discovery channel open.

---

## 7. Cross-cutting limitations

These bite all three tiers; the tiers differ only in how directly they expose them.

### 7.1 Gradient absorption (Tier 0, Tier 2/3 with gradient signal)

Equilibrium ⟹ existing useful edges produce small gradients ⟹ they look uninformative under any score that uses $\gamma$ at equilibrium. **Mitigations:**

- Edge-masking (Tier 0): perturb out of equilibrium before scoring.
- MLM (Tier 4): counterfactual reconstruction is not an equilibrium probe.
- Pre-equilibrium scoring: take measurements from partially-trained checkpoints, not the converged model.

### 7.2 Confounded co-activity (all tiers)

Nodes that share an upstream cause are correlated. Any edge-inference signal that relies on association will produce candidate clusters around true edges, not point sources. **Mitigations:**

- Partial correlation / stagewise greedy selection (Tier 0).
- Softmax attention's inherent normalization (Tier 4).
- Multivariate verification pass on top-k candidates (all tiers).

### 7.3 Identifiability (all tiers)

`notes.md` 07/15/25 explains: many graph configurations yield equivalent or near-equivalent loss. The score recovers *some* useful edges, not necessarily *the* biological ones. **Mitigations:**

- Combine with orthogonal priors: signed-edge consistency (`notes.md` 06/06/25), pathway co-membership (`functional_dis_and_similarity.md`), more output supervision.
- Treat candidates as hypotheses for downstream verification, not ground truth.

### 7.4 Stationarity (Tier 2/3, Tier 4)

Encoders / surrogates trained on a moving target. Embeddings during early training are noise. **Mitigations:**

- Warmup schedule — start edge inference at e.g. 50% of training.
- EMA on embedding statistics.
- Periodic consolidation: top-k candidates get promoted into `G`; refit; repeat.

---

## 8. Validation protocol

The same three-step protocol applies to all tiers; only the score function changes.

1. **Synthetic ground-truth recovery.** Use `gsnn/simulate/` to generate data from a known graph `G*`. Train GSNN on a subset `G_partial ⊊ G*`. Compute scores; check whether held-out edges `G* \ G_partial` rank higher than random non-edges. This is the cleanest test: alignment and absorption are both controlled, ground truth is unambiguous.
2. **Edge-masking self-consistency on real graphs.** Mask sampled subsets of existing edges via `edge_mask`. Recompute scores. Masked edges should rise in rank; in-graph unmasked edges should not. Direct test of whether the score detects edges the model is *not* currently using.
3. **Random-graph control.** Replace `G` with a permuted graph of the same degree distribution. Train and score. The scores should produce no consistent enrichment of the original `G`. This rules out the possibility that the score reflects generic smoothing or capacity rather than structural signal — analogous to the random-pathway control in `functional_dis_and_similarity.md` §8.

Optional: pathway-coherence check on top-k candidates (do inferred edges connect nodes in the same Reactome pathway more often than chance?) provides external biological corroboration.

---

## 9. Recommended progression

Start cheap and walk up the ladder only if the cheaper tier is insufficient. The diagnostic at each step tells you whether the next tier's complexity is warranted.

1. **Tier 0** (magnitude correlation). One streaming pass through training data. If synthetic recovery rate > random by a clear margin, you have a working scalable inferrer with zero training overhead. If recovery is at chance, diagnose: is the issue gradient absorption (try pre-equilibrium scoring) or information loss from the scalar reduction (move to Tier 2)?
2. **Tier 1** (reuse `NodeAttention.α`). Free if `node_attn=True`. Quick A/B against Tier 0.
3. **Tier 2** (learned scalar dual encoder, MLM training). First introduction of training overhead, but minimal parameters. Tests whether *learned* scalars are meaningfully better than fixed magnitudes.
4. **Tier 3** (learned d-dim dual encoder). Adds the channel structure back. Pair with pathway prior (`functional_dis_and_similarity.md` §5) to address alignment.
5. **Tier 4** (transformer surrogate). Only worth the engineering if Tier 3 plateaus *and* you need (a) sparse-attention discovery range, (b) multi-hop reasoning explicit in the surrogate, or (c) MLM as a counterfactual escape from gradient absorption.

The cheaper tiers also serve as calibrated baselines for the more expensive ones — at each step you have a previous-tier reference to attribute improvements against.

---

## 10. Integration points in the existing codebase

| Need | Existing hook | File / line |
|---|---|---|
| Per-layer node activations | `ResBlock._store_activations` / `_last_activation` | `gsnn/models/GSNN.py:766` |
| Channel → node mapping | `channel_groups` buffer on each `ResBlock` | `gsnn/models/GSNN.py:624` |
| Edge masks (for edge-masking validation) | `forward(x, edge_mask=...)` | `gsnn/models/GSNN.py:1097` |
| Sparse weight tensors (for reference) | `ResBlock.lin_in.values`, `ResBlock.lin_out.values` | `gsnn/models/GSNN.py:1071–1072` |
| Streaming sufficient-statistics pattern | `OutputEdgeInferer.evaluate` | `gsnn/optim/OutputEdgeInferer.py:230–543` |
| Shared per-node transformation (alignment substrate) | `NodeMLP` | `gsnn/models/GSNN.py:533–548` |
| Existing scalar per-node summary (Tier 1) | `NodeAttention._last_alpha` | `gsnn/models/GSNN.py:414–526` |
| Single-input edge inference (reference / for Option D distillation) | `InputEdgeInferrer` | `gsnn/optim/InputEdgeInferrer.py` |
| Single-output edge inference (reference / for Option D distillation) | `OutputEdgeInferer` | `gsnn/optim/OutputEdgeInferer.py` |

Suggested module layout for handoff:

- `gsnn/optim/MagnitudeEdgeInferer.py` — Tier 0. Wraps the model, hooks `_store_activations` and `retain_grad`, accumulates streaming statistics, returns the score DataFrame (mirror `OutputEdgeInferer.evaluate`'s output schema).
- `gsnn/optim/DualEncoderEdgeInferer.py` — Tiers 1–3. Two encoders, training loop with masked-edge reconstruction + optional pathway prior hook.
- `gsnn/optim/TransformerEdgeInferer.py` — Tier 4. Per-node-token transformer, MLM training, attention extraction.

All three should share an evaluation harness that runs the §8 validation protocol against the same trained GSNN, so tiers are directly comparable on equal footing.

---

## 11. Open decisions

- [ ] Reduction choice for Tier 0: L1 vs L2 vs squared L2 vs max. Empirical, probably L1.
- [ ] Per-layer scores vs aggregated. Default per-layer with learned softmax across layers.
- [ ] EMA decay rate for gradient statistics in Tier 0 (mitigates batch noise).
- [ ] Masked-edge sampling rate for Tier 2/3 training (matches MLM mask rate; 15% is the BERT default).
- [ ] Whether to use FAISS / HNSW for top-k retrieval at Tier 3/4 inference, or compute the full N×N (cheap at N=10k).
- [ ] Identity embedding initialization for Tier 4 — random vs initialized from a pretrained node2vec / pathway embedding.
- [ ] Sparse attention strategy for Tier 4: top-k attention vs Longformer-style local+global vs FlashAttention with structural mask.
- [ ] Whether tiers should be trained jointly with the main GSNN loss or as a separate post-training pass.

---

## 12. Summary

The intractable O(E²) edge-inference target collapses to a tractable O(N²) or O(Nk) problem when the per-edge state is replaced by per-node summaries. Three families do this with increasing capacity:

- **Tier 0** uses fixed magnitude reductions and Pearson correlation. Zero parameters, no alignment problem, full statistical machinery. Best starting point and permanent baseline.
- **Tier 2/3** uses a learned dual encoder (scalar or d-dim). Recovers channel structure; introduces an alignment problem that NodeMLP makes plausible but does not solve, requiring identity embeddings or pathway-prior scaffolding.
- **Tier 4** uses a transformer surrogate with per-node tokens and MLM training. Solves alignment by construction (identity embeddings), provides multi-hop reasoning (layer stacking), sidesteps gradient absorption (counterfactual reconstruction), and exposes attention as the edge-inference readout. Costs are compute, memory (requires sparse attention at biological scale), and the engineering complexity of a second model.

Universal limitations — gradient absorption at equilibrium, confounded co-activity, structural identifiability, stationarity of the score during training — apply to all three. Mitigations exist for each and are described in §7.

The validation protocol (synthetic ground-truth recovery, edge-masking self-consistency, random-graph control) is the same across tiers, which makes them directly comparable. Start at Tier 0; escalate only when the cheaper tier's diagnostic justifies it.
