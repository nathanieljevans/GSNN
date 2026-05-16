# Encoding prior knowledge of functionally similar / dissimilar pathways

> Status: design note — not yet implemented. Intended to be a complete handoff for future development.

## 1. Motivation

GSNN constrains a neural network's wiring with a biological interaction graph `G` so that we learn structured signaling patterns. In practice this constraint is often **insufficient** for biological identifiability:

- Real interaction graphs have many redundant paths between any two nodes.
- Many graph configurations yield equivalent or near-equivalent loss values (see `notes.md` 07/15/25 entry on graph-equivalence under structural perturbation, and the `04_reinforce.ipynb` results showing that adding a few false edges does not visibly hurt performance).
- The GSNN is therefore free to pick "physically valid but biologically wrong" routings.

Pathway databases (Reactome, KEGG, MSigDB, ...) encode an orthogonal prior: which nodes are part of the same biological program. We want to inject this prior **without** adding pathway nodes/edges directly to `G`, because that would create hub nodes that violate the sparse signaling constraint that gives GSNN its inductive bias in the first place.

Goal: an auxiliary objective (regularizer or auxiliary loss) that biases the model toward solutions consistent with pathway co-membership, while remaining computationally tractable on graphs of ~10k nodes / ~100k edges and remaining empirically testable.

---

## 2. Defining "functional (dis)similarity"

Before picking an objective we have to commit to a notion of similarity. There are three distinct candidates, and the right loss depends on which one we mean:

1. **Co-active in the same sample.** When pathway `P` is "on" in sample `b`, all members of `P` tend to be active. Implies *cross-sample* (batch-wise) correlation of activations.
2. **Functionally substitutable.** Node `i`'s representation can predict node `j`'s. Implies a learned mapping.
3. **Mechanistically equivalent (sign-aware).** Members move in the same direction, not just magnitude.

**Decision for this design:** we target (1), with optional sign-awareness via a signed projection. This matches:

- biological reality (pathways are context-dependent — different cell lines/drugs activate different programs),
- the way pathway scores are typically computed (GSEA-style aggregations across samples),
- and the cheapest tractable formulation (see §5).

### 2.1 Constructing G\*

Let `G` be the GSNN biological interaction network (directed, signed where available). Let `G*` be an **undirected** "pathway-coupling" graph over the same function-node set:

- Embed a pathway × gene bipartite graph (Reactome / KEGG / MSigDB) using node2vec, BiNE, or a simple SVD of the membership matrix.
- For each function node, take its `k`-nearest neighbors in embedding space → similarity edges in `G*`.
- For dissimilarity edges, take `k`-farthest (or sample from low-similarity tail with thresholding to avoid spurious orthogonality).
- Optional: weight edges by similarity strength rather than treating G\* as binary.

In addition to the pairwise `G*`, we also extract **pathway groups**:

- For each pathway `p`, define `V_p ⊆ V` (nodes that are members) and `E_p ⊆ E` (edges of `G` whose endpoints are both in `V_p`). `E_p` is the natural unit when working at the edge level (which fits GSNN's edge-centric architecture).

Both `G*` (pairwise) and `{V_p, E_p}` (groupwise) representations should be derivable from the same pathway DB ingestion step.

---

## 3. Originally considered approaches (kept for reference)

These are the three formulations originally proposed. They are retained here because each highlights a useful intuition, even though we are not pursuing them as primary implementations.

### 3.1 Pairwise predictive layer (rejected as primary)

For each pathway-similar pair `(i, j) ∈ G*`, learn a linear map and minimize:

$$\hat{a}_i^\ell = W_{ij} a_j^\ell, \quad \mathcal{L} = \mathrm{MSE}(a_i^\ell, \hat{a}_i^\ell)$$

For dissimilarity edges, use a gradient-reversal layer (GRL) on the same objective.

**Why rejected:**

- Parameter cost is `O(d^2 · E*)` per layer if the map is full rank, or `O(d · E*)` even for diagonal maps. With `d = node_mlp_hidden = 128` and `E* ~ 50k`, this is intractable (~800M extra params).
- GRL is fragile in practice; competes with primary loss.
- Picks substitutability (notion 2 from §2), not co-activity, which is the wrong target for context-dependent pathways.

### 3.2 mean(|a|) correlation (rejected as primary)

Encourage `mean(|a_i^l|) ≈ mean(|a_j^l|)` for `(i,j) ∈ G*`.

**Why rejected:**

- Collapses each node to a 1-D summary per layer; discards which channels are firing, the sign, and direction.
- With `node_mlp` per-node capacity, two unrelated nodes can have identical `mean(|a|)`.
- Only a degenerate fallback if signed projections are unstable.

### 3.3 Edge-group pairwise MSE (subsumed)

For each pathway `p`, collect `E_p`. For all `(e, f) ∈ E_p × E_p`:

$$\mathcal{L}_p = \sum_{(e,f) \in E_p^2} \mathrm{MSE}(|h_e|, |h_f|)$$

**Why subsumed (not rejected):** the *idea* (edge-level pathway grouping aligned with GSNN's edge-centric machinery) is good and is preserved in §5. The *form* (pairwise MSE on magnitudes) is bad: `O(|E_p|^2)` per pathway, and discards sign. Replace with the latent-factor formulation in §5.

---

## 4. Two structural problems the original proposals share

1. **They ignore the batch dimension.** Pathway co-activity is a *statistical* property over samples, not a within-sample equality. Within-sample MSE is too strict and fights context-dependence.
2. **They scale pairwise** in `E*` or `|E_p|^2`. The cheaper, equivalent move is to introduce **one shared latent per pathway** that members regress against — a one-factor model. Members co-vary by sharing a factor; no pairwise terms.

These two observations motivate the two recommended approaches below.

---

## 5. Recommended primary approach: pathway latent-factor regularizer

### 5.1 Formulation

For each pathway `p`, layer `ℓ`, and minibatch of size `B`:

1. Apply a scalar reduction `φ_ℓ` to each member's activation:
   - **Node-level:** `s_i^{ℓ,b} = φ_ℓ(a_i^{ℓ,b})`, where `a_i^{ℓ,b} ∈ ℝ^{C_pn}` is the per-node channel vector.
   - **Edge-level:** `s_e^{ℓ,b} = φ_ℓ(h_e^{ℓ,b})`, where `h_e^{ℓ,b}` is the latent edge feature.
   - Recommended `φ_ℓ`: a learned linear projection `φ_ℓ(z) = w_ℓ^\top z` (so it is *signed*, not `|·|`). Optionally normalize so `‖w_ℓ‖ = 1`. One `w_ℓ` per layer is enough (or one per pathway × layer if you can spare the params: `P · L · d`).
2. Compute the per-pathway score by aggregating members:

$$S_p^{\ell,b} = \frac{1}{|p|} \sum_{i \in p} s_i^{\ell,b}$$

(weighted mean if pathway membership has confidence scores from G\*.)

3. **Similarity loss (encourage co-variation):** maximize across-batch correlation of each member with the pathway score.

$$\mathcal{L}_{\text{sim}} = -\sum_{p,\ell} \frac{1}{|p|} \sum_{i \in p} \mathrm{corr}_b\!\big(s_i^{\ell,b},\; S_p^{\ell,b}\big)$$

where `corr_b` is the Pearson correlation across the batch dimension.

4. **Dissimilarity loss (encourage orthogonality between dissimilar pathways `p, q`):**

$$\mathcal{L}_{\text{dis}} = \sum_{(p,q) \in D, \ell} \mathrm{corr}_b\!\big(S_p^{\ell,b},\; S_q^{\ell,b}\big)^2$$

`D` is the set of pathway pairs flagged as dissimilar (from the dissimilarity-G\* construction in §2.1).

5. **Total auxiliary loss:**

$$\mathcal{L}_{\text{aux}} = \lambda_{\text{sim}} \mathcal{L}_{\text{sim}} + \lambda_{\text{dis}} \mathcal{L}_{\text{dis}}$$

Add to the primary supervised loss with a small `λ` schedule (warmup-to-`λ_max` after a few epochs is recommended; cold-starting with the regularizer can lock in degenerate solutions).

### 5.2 Why this is the right shape

- **Linear cost.** `O(L · Σ_p |p| · B)` per minibatch — *linear* in pathway sizes, no pairwise terms.
- **Zero (or tiny) extra parameters.** `0` if `φ_ℓ` is a fixed reduction (sum, mean of channels). `L · d` if `φ_ℓ` is a single learned projection per layer; `P · L · d` if per-pathway. All small relative to the main model.
- **Sign-aware.** A learned linear `φ_ℓ` preserves direction — fixes the chief weakness of §3.2 / §3.3.
- **Context-dependent.** Members are encouraged to fire together *only* in samples where the pathway score is non-zero. We are not forcing fixed-magnitude equality.
- **Subsumes §3.3** by replacing the `O(|E_p|^2)` pairwise MSE with a one-factor model that has the same intent.
- **Naturally handles dissimilarity** as an orthogonality penalty between pathway scores rather than a fragile gradient-reversal trick.

### 5.3 Pseudocode

```python
# inputs: node_acts[ℓ] : Tensor (B, N_func, C_pn)   from ResBlock._last_activation
#         pathway_membership : sparse (P, N_func)
#         dissim_pairs : LongTensor (M, 2)
#         phi_proj : nn.Linear(C_pn, 1)              shared or per-layer

def pathway_loss(node_acts, pathway_membership, dissim_pairs, phi_proj):
    L_sim, L_dis = 0.0, 0.0
    for ell, A in enumerate(node_acts):              # A: (B, N, C_pn)
        s = phi_proj(A).squeeze(-1)                  # (B, N)
        # pathway score: (B, P)
        S = (pathway_membership @ s.T).T / pathway_membership.sum(1).clamp_min(1)

        # similarity: per-pathway mean correlation of members with S_p
        # vectorize via standardized vars
        s_std = (s - s.mean(0)) / (s.std(0) + 1e-6)      # (B, N)
        S_std = (S - S.mean(0)) / (S.std(0) + 1e-6)      # (B, P)
        # corr_{p,i in p} = (1/B) <s_std[:,i], S_std[:,p]>
        member_corr = (s_std.T @ S_std) / s_std.size(0)  # (N, P)
        member_corr = member_corr * pathway_membership.T # zero out non-members
        L_sim -= member_corr.sum() / pathway_membership.sum().clamp_min(1)

        # dissimilarity: squared corr between dissimilar pathway score pairs
        Sp = S_std[:, dissim_pairs[:,0]]
        Sq = S_std[:, dissim_pairs[:,1]]
        L_dis += ((Sp * Sq).mean(0) ** 2).mean()
    return L_sim, L_dis
```

The above is **not** production code — it is a reference for the math. See §7 for integration.

### 5.4 Knobs / open design choices

- **Node level vs edge level.** Node-level is simpler. Edge-level (replace `s_i` with `s_e` over `e ∈ E_p`) is more directly aligned with GSNN's edge-centric machinery and lets pathways pin *interactions* not just nodes. Try both. Edge-level has higher variance per sample; consider averaging across layers before computing correlation if so.
- **Across-layer aggregation.** Apply per-layer (richer signal, more variance) or aggregate `s` across layers first (fewer correlation tests, smoother). Default: per-layer.
- **Batch size sensitivity.** Pearson `corr_b` is noisy for small `B`. Use `B ≥ 32`. Optionally use a momentum-buffered running mean/variance for `s` and `S` if memory-bound.
- **Membership weighting.** If `G*` was built with similarity scores, use them as soft membership in `pathway_membership` (a real-valued matrix instead of `{0,1}`).
- **Pathway redundancy.** Reactome has hierarchical / overlapping pathways. Either deduplicate by Jaccard threshold or treat overlap as expected and let the model handle it (recommended; overlap is real biology).

---

## 6. Recommended secondary approach: pathway-aware weight regularization

A complementary, **inference-free** prior that biases the function class itself rather than the activations. Use this either alone (when you don't want auxiliary loss complexity) or combined with §5.

### 6.1 Formulation

For each layer `ℓ` and each pathway `p`, look at the rows/columns of `lin_in` and `lin_out` (in `gsnn/models/SparseLinear.py`) that correspond to edges `e ∈ E_p`. Soft-tie those weights toward the pathway centroid:

$$\mathcal{L}_{\text{w}} = \sum_{p, \ell} \sum_{e \in E_p} \big\| w_e^\ell - \bar w_p^\ell \big\|^2, \quad \bar w_p^\ell = \frac{1}{|E_p|} \sum_{e \in E_p} w_e^\ell$$

This is soft parameter tying / multi-task family regularization: edges in the same pathway are nudged toward a shared template, but free to deviate. Equivalent to a Gaussian prior centered at the pathway centroid.

### 6.2 Properties

- **Cost.** `O(L · Σ_p |E_p| · d)` per regularizer evaluation (once per step, not per sample).
- **No inference-time cost.** Pure training-time regularizer.
- **No gradient through activations.** Side-steps any data-dependent instability.
- **Limitation.** Cannot enforce co-activity that depends on context, only weight similarity. Best as a complement to §5.

### 6.3 Variants

- **Low-rank centroid:** factorize `w_e^ℓ ≈ U_p^ℓ + V_e^ℓ` with `U_p^ℓ` shared per pathway and small per-edge `V_e^ℓ`. Hard parameter sharing of the pathway component.
- **Initialization-only:** drop the running regularizer and instead just initialize edges in the same pathway with correlated weights. Cheapest baseline.
- **Output-edge focus:** apply only to `lin_out` or only to `lin_in`; the layer that gates *which* signals enter the node may be the better target. Empirically determine.

---

## 7. Integration points in the existing codebase

The implementation surface is small because GSNN already exposes everything needed.

| Need | Existing hook | File / line |
|---|---|---|
| Per-layer node activations | `ResBlock._store_activations` / `_last_activation` | `gsnn/models/GSNN.py:766` |
| Channel → node mapping | `channel_groups` buffer on each `ResBlock` | `gsnn/models/GSNN.py:624` |
| Edge masks / edge views | `function_edge_mask`, `input_edge_mask`, `output_edge_mask` | `gsnn/models/GSNN.py:911-913` |
| Sparse weight tensors (for §6) | `ResBlock.lin_in.values`, `ResBlock.lin_out.values` | `gsnn/models/GSNN.py:1071-1072` (used in `prune`) |
| Existing auxiliary structural loss pattern (signed edges) | `SignedMessagePassing` + `notes.md` 06/06/25 | `gsnn/models/GSNN.py:360` |

Suggested module layout (for handoff):

- `gsnn/proc/pathways.py` — pathway DB ingestion → `pathway_membership` matrix, `E_p` groups, `G*` similarity / dissimilarity edges. Cache to disk.
- `gsnn/models/PathwayRegularizer.py` — class with two methods:
  - `activation_loss(model, last_activations) → (L_sim, L_dis)` for §5.
  - `weight_loss(model) → L_w` for §6.
- Training-loop addition: enable `_store_activations = True`, run forward, collect `_last_activation` per `ResBlock`, call regularizer, add to loss.

The signed-edge auxiliary loss already in the codebase is the closest analog and a good template for how to wire this in without disrupting the main forward path.

---

## 8. Verification plan

Performance alone will not separate "the model is using pathway info" from "the model just got more capacity / a different inductive bias." Each test below is intentionally orthogonal to validation accuracy.

1. **Pathway enrichment in learned activations.** Cluster node activations (`get_node_activations`) post-training. Compute Adjusted Rand Index against pathway membership labels for the regularized vs unregularized model. Expectation: ARI ↑ with no test-loss degradation.
2. **Held-out pathway recovery.** Train with §5 but with 20% of pathway memberships hidden from the loss. After training, check whether held-out members still co-vary more with their (unseen) pathway score than random-pair baselines. Direct test of generalization of the prior.
3. **Counterfactual / occlusion coherence.** Use `gsnn/interpret/OcclusionExplainer.py` and `gsnn/interpret/CounterfactualExplainer.py`. For each input perturbation, compute pathway enrichment (hypergeometric / GSEA) of the top-`k` attributed nodes. Pathway-regularized models should produce more pathway-coherent attributions.
4. **Identifiability under graph perturbation.** Replicating the `04_reinforce.ipynb` setup: inject `k` spurious edges into `G`. Train models with and without §5/§6. Measure attribution alignment with ground-truth signaling for synthetic data. Pathway-regularized model should be more robust (smaller drop).
5. **Drug → pathway consistency.** For drugs with known target pathway `P`, perturbing the drug input should produce larger downstream changes in `V_p` than in non-`V_p` nodes. Compute the ratio; should increase with regularization.
6. **Ablations.**
   - λ-sweep for §5 (`λ_sim`, `λ_dis`).
   - §5 alone vs §6 alone vs both.
   - Node-level §5 vs edge-level §5.
   - Random-pathway control: replace true pathways with random partitions of equal size; the prior should *not* help under this control. This is the most important single sanity check — if it does help with random pathways, the regularizer is just acting as a smoothing prior, not encoding pathway info.

The random-pathway control in (6) is the load-bearing experiment. Plan for it from day one.

---

## 9. Decisions still open (TODO before implementation)

- [ ] Pick pathway DB(s) and version(s) (Reactome current-release recommended; MSigDB Hallmark as a smaller-scale sanity set).
- [ ] Decide soft vs binary pathway membership.
- [ ] Decide per-layer vs aggregated-across-layers `φ`.
- [ ] Decide node-level vs edge-level for §5 (or implement both behind a flag).
- [ ] Decide whether `φ_ℓ` is fixed (mean of channels) or learned (linear projection). Default learned.
- [ ] Decide on dissimilarity edge construction — k-farthest in embedding vs explicit "antagonistic pathways" annotation. Empirical question.
- [ ] Decide caching strategy — pathway membership matrices can be precomputed and registered as model buffers so they ride with the checkpoint.

---

## 10. Summary

Drop the pairwise predictive layer (3.1) and the magnitude-only correlation (3.2). Keep the edge-grouping intuition from (3.3) but replace its `O(|E_p|^2)` pairwise MSE with a one-factor model.

Primary mechanism: a **per-pathway latent score** that members are encouraged to correlate with **across the batch dimension**, with a separate **orthogonality penalty** between dissimilar pathway scores. Cost is linear in pathway size, parameter count is `O(L · d)` or `O(P · L · d)`, sign is preserved via a learned linear projection, and context-dependence is built in.

Secondary mechanism: **soft parameter tying** of `lin_in` / `lin_out` weights for edges in the same pathway, toward the pathway centroid. Free at inference, complementary to the primary mechanism.

Verify with pathway enrichment in activations, held-out pathway recovery, attribution coherence, identifiability under graph perturbation, and — critically — a random-pathway control to rule out generic smoothing effects.
