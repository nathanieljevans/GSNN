# GSNN Test Suite TODO

Comprehensive test plan for the `gsnn/` package, ranked from **highest** to **lowest** priority. Tests are grouped by priority tier; within each tier, items are ordered roughly by dependency order (foundational utilities before consumers).

**Existing coverage:** `test_extract_entity_function.py` — smoke test for `extract_entity_function` across three norm modes.

**Suggested conventions:**
- Place unit tests under `gsnn/tests/`
- Name files `test_<module>.py`
- Use small synthetic graphs (2–5 nodes) for model tests; use `simulate_3_in_3_out` for integration tests
- Mark slow/integration tests with `@pytest.mark.slow`
- Prefer deterministic seeds for simulation and explainer tests

---

## P0 — Critical core (model math & graph indexing)

These modules define GSNN correctness. Bugs here silently break training, inference, and all downstream explainers.

### `models/SparseLinear.py` → `test_sparse_linear.py`

| Test | Description |
|------|-------------|
| `test_conv_message_passing` | `Conv` aggregates weighted neighbor features; bias applied when present |
| `test_batch_graphs_offsets` | Batched `edge_index` offsets are correct for B>1 bipartite graphs |
| `test_sparse_linear_forward_shape` | Output shape `(B, M, 1)` for input `(B, N, 1)` |
| `test_sparse_linear_matches_dense` | Forward pass matches equivalent dense matmul on toy COO |
| `test_sparse_linear_no_bias` | Forward works when `bias=False`; no bias parameter registered |
| `test_init_schemes` | Each init (`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `uniform`, `normal`, `degree_normalized`, `zeros`) runs without error |
| `test_init_invalid_raises` | Unknown `init` raises `ValueError` |
| `test_sparse_linear_batched_indices_cache` | Custom `batched_indices` produces same result as auto-computed |
| `test_prune_reduces_edges` | `prune(idxs)` shrinks `values` and `indices` consistently |
| `test_init_helpers_finite` | `xavier_*`, `kaiming_*`, `uniform`, `normal` return finite tensors of correct shape |

### `models/utils.py` → `test_models_utils.py`

| Test | Description |
|------|-------------|
| `test_hetero2homo_offsets` | Input/output node index offsets and edge concatenation order |
| `test_hetero2homo_masks` | `input_node_mask` / `output_node_mask` identify correct nodes |
| `test_hetero2homo_edge_weights` | Edge weights concatenated in same order as edges when provided |
| `test_get_Win_indices` | COO indices match expected row/col for uniform and per-node channel counts |
| `test_get_Wout_indices` | COO indices for W_out align with function-node out-edges |
| `test_get_conv_indices` | Tuple sizes, channel_groups length, w_in/w_out dimensions consistent |
| `test_node2edge` | Source node features broadcast to outgoing edges |
| `test_edge2node_single_in_degree` | Single incoming edge per output node |
| `test_edge2node_multi_in_degree` | Multiple edges summed and sqrt-degree normalized |
| `test_apply_norm_and_nonlin_order` | `norm_first=True/False` applies ops in correct order |
| `test_corr_score_pearson` | Known correlation on synthetic data |
| `test_corr_score_constant_column` | Zero-variance column returns 0 |
| `test_corr_score_invalid_method` | Unknown method raises `ValueError` |
| `test_predict_gsnn` | Mock loader returns stacked y, yhat, sig_ids |

### `models/GSNN.py` → `test_gsnn.py`

| Test | Description |
|------|-------------|
| `test_gsnn_forward_shape` | Output shape `(B, num_outputs)` on minimal graph |
| `test_gsnn_ret_edge_out` | `ret_edge_out=True` returns edge-level features |
| `test_gsnn_share_layers` | `share_layers=True` reuses same SparseLinear weights across layers |
| `test_gsnn_unshared_layers` | `share_layers=False` creates independent layer weights |
| `test_gsnn_self_edges` | `add_function_self_edges=True` augments edge_index |
| `test_gsnn_edge_mask` | `edge_mask` zeros masked edges during forward |
| `test_gsnn_node_mask` | `node_mask` suppresses function-node channels |
| `test_gsnn_node_activity_requires_x_fn` | `node_activity=True` without `x_fn` raises `ValueError` |
| `test_gsnn_node_activity_per_node` | Gate applied; output shape unchanged |
| `test_gsnn_node_activity_per_channel` | `node_activity_mode='per-channel'` runs |
| `test_gsnn_get_batch_params_caching` | Same B returns cached batch indices |
| `test_gsnn_get_batch_params_different_B` | Different B recomputes indices |
| `test_gsnn_prune` | `prune()` removes small-magnitude edges; returns param count |
| `test_gsnn_checkpoint_forward` | `checkpoint=True` in train mode completes forward/backward |
| `test_gsnn_node_errs_length` | Wrong-length `node_errs` raises `ValueError` |
| `test_gsnn_get_node_activations_agg` | Each `agg` mode (`sum`, `mean`, `max`, `last`, `all`) returns expected dict keys/shapes |
| `test_gsnn_get_node_attention` | Requires `node_attn=True`; returns per-node `(L, B)` tensors |
| `test_gsnn_edge_weight_dict` | Optional edge weights propagate through ResBlocks |
| `test_gsnn_norm_variants` | Parametrize over norm types used in ResBlock |
| `test_gsnn_gradient_flow` | Loss backward reaches SparseLinear parameters |

### `models/ResBlock.py` → `test_resblock.py`

| Test | Description |
|------|-------------|
| `test_resblock_forward_shape` | Preserves `(B, num_edges)` |
| `test_resblock_residual_off` | `residual=False` skips skip connection |
| `test_resblock_node_mask` | `set_node_mask` zeros masked channels |
| `test_resblock_fn_activity` | External activity gate multiplies channels |
| `test_resblock_node_err` | `node_err` added after lin_in |
| `test_resblock_store_activations` | `_store_activations=True` populates `_last_activation` |
| `test_resblock_norm_variants` | Each supported norm type runs forward |
| `test_resblock_invalid_norm` | Unknown norm raises `ValueError` |
| `test_resblock_node_mlp` | `node_mlp=True` applies per-node MLP |
| `test_resblock_node_attn` | `node_attn=True` applies NodeAttention |
| `test_resblock_shared_lin` | Pre-built `lin_in`/`lin_out` reused when passed in |

---

## P1 — Core model components

Building blocks used inside GSNN/ResBlock; failures cause subtle shape or semantics bugs.

### `models/SignedMessagePassing.py` → `test_signed_message_passing.py`

| Test | Description |
|------|-------------|
| `test_signed_mp_forward_shape` | `(B, N_fn)` in/out |
| `test_signed_mp_sign_flip` | Negative edge weight inverts neighbor contribution |
| `test_signed_mp_no_function_edges` | Identity-like behavior when no func→func edges |

### `models/NodeAttention.py` → `test_node_attention.py`

| Test | Description |
|------|-------------|
| `test_node_attention_forward_shape` | Output same shape as input |
| `test_node_attention_return_alpha` | `return_alpha=True` returns `(out, alpha)` with `(B, n_nodes)` |
| `test_node_attention_wrong_channels` | Mismatched channel count raises `ValueError` |
| `test_node_attention_with_signed_mp` | Signed message passing path when edge_index/weight provided |
| `test_node_attention_stores_last_alpha` | `_last_alpha` set after forward |

### `models/NodeActivity.py` → `test_node_activity.py`

| Test | Description |
|------|-------------|
| `test_node_activity_per_node_mode` | Output shape `(B, Ncg)` |
| `test_node_activity_per_channel_mode` | Output shape `(B, Ncg)` with per-channel gates |
| `test_node_activity_2d_input` | `(B, Nf)` accepted when `activity_dim=1` |
| `test_node_activity_wrong_Nf` | Mismatched function node count raises `ValueError` |
| `test_node_activity_wrong_dim` | 2-D input with `activity_dim>1` raises `ValueError` |
| `test_node_activity_temperature` | Lower temperature → sharper sigmoid gates |
| `test_get_alpha_mean_before_forward` | Raises if forward not run |

### `models/NodeMLP.py` → `test_node_mlp.py`

| Test | Description |
|------|-------------|
| `test_node_mlp_forward_shape` | `(B, N, C)` preserved |
| `test_node_mlp_dropout_train_eval` | Dropout active in train, off in eval |

### Normalization layers → `test_norm_layers.py`

| Module | Tests |
|--------|-------|
| `GroupLayerNorm` | Forward shape; per-group mean≈0, std≈1; affine gamma/beta |
| `GroupBatchNorm` | Forward shape; running stats update in train mode |
| `GroupRMSNorm` | Forward shape; RMS normalization |
| `SoftmaxGroupNorm` | Weights sum to 1 within each group |
| `GroupEMANorm` | EMA stats update; eval uses running stats |
| `ChannelEMANorm` | Per-channel EMA on edge/channel dim |

### `models/NN.py` → `test_nn.py`

| Test | Description |
|------|-------------|
| `test_nn_forward_shape` | `(B, in) → (B, out)` |
| `test_nn_layers_depth` | `layers=1` vs `layers=3` parameter count |
| `test_nn_no_norm` | `norm=None` skips normalization |

### `models/PathwayLatentRegularizer.py` → `test_pathway_latent_regularizer.py`

| Test | Description |
|------|-------------|
| `test_enable_disable_hooks` | `enable`/`disable` register/remove forward hooks |
| `test_loss_nonnegative` | `loss(model)` ≥ 0 on toy GSNN |
| `test_loss_zero_without_hooks` | No loss when disabled |

---

## P2 — Simulation & graph I/O

Enables reproducible training benchmarks and tutorial workflows.

### `simulate/nx2pyg.py` → `test_nx2pyg.py`

| Test | Description |
|------|-------------|
| `test_nx2pyg_edge_categorization` | Edges sorted into input/function/output buckets |
| `test_nx2pyg_with_weights` | `weight_attr` populates `edge_weight_dict` |
| `test_pyg2nx_roundtrip` | `pyg2nx(nx2pyg(G))` preserves edges on toy graph |

### `simulate/utils.py` → `test_simulate_utils.py`

| Test | Description |
|------|-------------|
| `test_nx_to_pyro_model_samples` | Model returns dict with output node keys |
| `test_nx_to_pyro_special_functions` | Custom node function overrides linear default |
| `test_nx_to_pyro_signed_edges` | Signed edge weights affect conditional means |

### `simulate/simulate.py` → `test_simulate.py`

| Test | Description |
|------|-------------|
| `test_simulate_shapes` | `(x_train, x_test, y_train, y_test)` shapes match n_train/n_test and node counts |
| `test_simulate_reproducibility` | Fixed seed gives identical samples (if seeded) |
| `test_simulate_sde_shapes` | SDE path returns correct array shapes |
| `test_simulate_sde_seed` | `seed` argument fixes output |
| `test_simulate_sde_return_order` | **Verify** return tuple order matches docstring (currently may differ) |

### `simulate/datasets.py` → `test_simulate_datasets.py`

| Test | Description |
|------|-------------|
| `test_simulate_3_in_3_out` | Returns graph, tensors, node lists; tensor shapes |
| `test_simulate_3_in_3_out_zscorey` | `zscorey=True` normalizes y_train/y_test |
| `test_simulate_10_in_25_func_10_out_cyclic` | Large cyclic graph builds and simulates |
| `test_gsnn_trainable_on_simulated_data` | End-to-end: build GSNN from nx2pyg data, one training step reduces loss |

### `simulate/graph_comparison.py` → `test_graph_comparison.py`

| Test | Description |
|------|-------------|
| `test_graph_comparison_identical` | Same graph → precision/recall = 1 |
| `test_graph_comparison_disjoint` | No shared deps → TP=0 |
| `test_get_dependency_details` | Returns expected dependency sets |

---

## P3 — Graph construction & preprocessing

Pipeline code for real-world constraint networks.

### `proc/subset.py` → `test_proc_subset.py`

| Test | Description |
|------|-------------|
| `test_bfs_distance` | Distances match hand-computed toy graph |
| `test_get_all_possible_paths_set` | SPL through each node on known DAG |
| `test_subset_graph_keeps_viable_paths` | Nodes on root→leaf paths retained |
| `test_subset_graph_prunes_dead_ends` | Off-path nodes removed |
| `test_build_nx` | DataFrame edges → correct DiGraph node typing |

### `proc/construct.py` → `test_proc_construct.py`

| Test | Description |
|------|-------------|
| `test_gsnn_network_constructor_build` | Minimal edge tables → valid `HeteroData` |
| `test_constructor_prunes_unreachable` | Mediator/function nodes off all paths dropped |
| `test_constructor_force_include_names` | Fixed name lists preserved even without edges |
| `test_constructor_graph_summary` | `graph_summary` contains expected keys/counts |

### `proc/coarsen.py` → `test_proc_coarsen.py`

| Test | Description |
|------|-------------|
| `test_io_equivalence` | Equivalent I/O behavior groups nodes correctly |
| `test_diff_equivalence` | Diffusion-based equivalence on toy graph |
| `test_diff_io_equivalence` | Combined I/O + diff equivalence |

### `proc/bio.py` → `test_proc_bio.py` *(optional data fixtures)*

| Test | Description |
|------|-------------|
| `test_uniprot2symbol` | ID mapping with `allow='1:m'` |
| `test_symbol2uniprot` | Reverse mapping |
| `test_ensg2symbol` | Ensembl → symbol |
| `test_build_uniprot_symbol_map` | Map built from mock func_edges DataFrame |
| `test_get_bio_interactions_smoke` | Runs with small mock/cached tables (mark `@pytest.mark.integration`) |
| `test_complex_handling_modes` | `_apply_complex_handling` expand vs collapse |

---

## P4 — Interpretability

User-facing explanation APIs; depend on P0/P1 model correctness.

### `interpret/_kwargs_utils.py` → `test_kwargs_utils.py`

| Test | Description |
|------|-------------|
| `test_normalize_model_kwargs_none` | Returns `{}` |
| `test_slice_per_sample` | Row `i` extracted; dim-1 tensors broadcast |
| `test_repeat_batch_from_1` | `(1, ...)` → `(n, ...)` |
| `test_repeat_batch_passthrough` | Already `(n, ...)` unchanged |
| `test_repeat_batch_invalid_dim` | Wrong leading dim raises `ValueError` |
| `test_tile_for_grid` | `(B, ...)` → `(outer*B, ...)` |
| `test_concat_pair` | Matching tensor keys concatenated on dim 0 |

### `interpret/extract_entity_function.py` → extend `test_extract_entity_function.py`

| Test | Description |
|------|-------------|
| `test_extract_entity_function_runs` | ✅ **Exists** (parametrized norm smoke) |
| `test_extract_matches_gsnn_subgraph` | Extracted module output matches GSNN internal activations for same node |
| `test_extract_invalid_node` | Unknown node name raises |
| `test_extract_layer_index` | Different `layer` values extract different weights |
| `test_dense_func_node_forward` | `dense_func_node` norm/nonlinearity ordering |
| `test_extract_unsupported_norm_raises` | `groupbatch`/`edgebatch` raise `NotImplementedError` |

### `interpret/GSNNExplainer.py` → `test_gsnn_explainer.py`

| Test | Description |
|------|-------------|
| `test_gsnn_explainer_edge_scores` | `explain(..., target='edge')` returns DataFrame with score column |
| `test_gsnn_explainer_node_scores` | `target='node'` mode |
| `test_gsnn_explainer_model_frozen` | Model params unchanged after explain |
| `test_gsnn_explainer_with_model_kwargs` | `x_fn` passed when `node_activity=True` |
| `test_gsnn_explainer_tune` | `tune()` finds beta meeting min_r2 |

### `interpret/IGExplainer.py` → `test_ig_explainer.py`

| Test | Description |
|------|-------------|
| `test_ig_explainer_edge_attributions` | Non-zero attributions sum sensibly |
| `test_ig_explainer_node_attributions` | Node mode runs |
| `test_ig_explainer_baseline` | Custom baseline changes attributions |
| `test_ig_explainer_model_kwargs` | Side inputs reshaped with x |

### `interpret/OcclusionExplainer.py` → `test_occlusion_explainer.py`

| Test | Description |
|------|-------------|
| `test_occlusion_edge_importance` | Masking high-importance edge hurts prediction more |
| `test_occlusion_node_importance` | Node occlusion mode |
| `test_occlusion_element_mask` | Subset mask restricts occluded elements |

### Contrastive explainers → `test_contrastive_explainers.py`

| Class | Tests |
|-------|-------|
| `ContrastiveGSNNExplainer` | Joint x1/x2 explain; edge and node modes; `tune()` |
| `ContrastiveIGExplainer` | Contrastive IG attributions differ for x1 vs x2 |
| `ContrastiveOcclusionExplainer` | Diff-based occlusion scores |

### Other interpret modules → `test_interpret_misc.py`

| Module | Tests |
|--------|-------|
| `CounterfactualExplainer` | Returns counterfactual x with changed prediction |
| `NoiseTunnel` | Aggregated scores over noisy samples |
| `interpret/utils.py` | `plot_edge_importance` / `plot_node_importance` run without error (smoke + save fig to tmp) |
| `plot_explanation_graph.py` | `plot_explanation_graph`, `plot_hairball`, `adjust_label_positions` smoke |

---

## P5 — Training & optimization utilities

Supporting training loops, diagnostics, and edge inference.

### `optim/EarlyStopper.py` → `test_early_stopper.py`

| Test | Description |
|------|-------------|
| `test_early_stop_improvement_resets` | Improvement resets counter |
| `test_early_stop_triggers` | Stops after `patience` epochs without improvement |
| `test_early_stop_min_delta` | Small improvements below min_delta don't reset |

### `optim/RewardScaler.py` → `test_reward_scaler.py`

| Test | Description |
|------|-------------|
| `test_edw_weights` | EDW weights sum appropriately |
| `test_reward_scaler_warmup` | Warmup period behavior |
| `test_reward_scaler_scale_clip` | Clipping at configured bound |

### `optim/Environment.py` → `test_environment.py`

| Test | Description |
|------|-------------|
| `test_augment_edge_index` | Action adds/removes edges correctly |
| `test_train_validate_run` | Smoke: one train/val step on tiny dataset |

### `optim/REINFORCE.py` → `test_reinforce.py`

| Test | Description |
|------|-------------|
| `test_reinforce_sample` | Action sampled from policy |
| `test_reinforce_update` | Policy update changes log-probs |
| `test_get_edge_probs` | Probabilities sum to 1 over actions |

### `optim/OutputEdgeInferer.py` → `test_output_edge_inferer.py`

| Test | Description |
|------|-------------|
| `test_output_edge_inferer_fit` | `fit()` completes on tiny loader |
| `test_output_edge_inferer_forward` | `forward(a)` shape matches activations |
| `test_output_edge_inferer_evaluate` | Metrics returned from evaluate |
| `test_safe_corrcoef` | Edge cases (constant vectors) |

### `optim/InputEdgeInferrer.py` → `test_input_edge_inferrer.py`

| Test | Description |
|------|-------------|
| `test_input_edge_inferrer_infer` | Inferred edge weights shape matches targets |
| `test_input_edge_inferrer_bootstrap` | `infer_bs` runs n_bootstrap iterations |

### `optim/utils.py` → `test_optim_utils.py`

| Test | Description |
|------|-------------|
| `test_compute_picp` | Coverage near nominal alpha on synthetic dist |
| `test_compute_ECE` | ECE in [0, 1] |
| `test_dbscan_silhouette_score` | Returns float or NaN gracefully |
| `test_neighborhood_preservation_score` | Score in reasonable range |

### Diagnostics → `test_diagnostics.py`

| Class | Tests |
|-------|-------|
| `GradDiagnostics` | `analyze`, `update`, `get_summary`, `reset` |
| `TrainingDiagnostics` | Hook registration, `update`, `get_summary`, `reset` |

### `optim/BayesOpt.py` → `test_bayes_opt.py`

| Test | Description |
|------|-------------|
| `test_bayes_opt_step` | One optimization step updates best action |
| `test_get_best_action` | Returns action from search space |

---

## P6 — OT, external, and specialized research code

Lower priority unless actively used in current workflows.

### `ot/SHD.py`, `ot/NOT.py`, `ot/OTICNN.py` → `test_ot.py`

| Test | Description |
|------|-------------|
| `test_not_step_smoke` | One training step without crash |
| `test_oticnn_get_T` | Transporter retrievable after init |
| `test_shd_step_smoke` | One SHD step |
| `test_runtime_scaler` | Scale/inv_scale roundtrip |

### `ot/utils.py`, `ot/mmd.py` → `test_ot_utils.py`

| Test | Description |
|------|-------------|
| `test_freeze_unfreeze` | Param `requires_grad` toggled |
| `test_mmd_distance` | MMD ≥ 0; identical samples → ~0 |
| `test_compute_scalar_mmd` | Scalar aggregation over gammas |

### `external/cellot.py`, `external/mmd.py` → `test_external.py`

| Test | Description |
|------|-------------|
| `test_icnn_convexity` | Convexity check on random inputs |
| `test_nonnegative_linear` | Weights stay ≥ 0 after forward |
| `test_compute_w2_distance` | W2 distance finite |

---

## Recommended implementation order

1. **P0** — `test_sparse_linear.py`, `test_models_utils.py`, `test_resblock.py`, `test_gsnn.py`
2. **P1** — norm layers, NodeAttention/Activity/MLP, SignedMessagePassing
3. **P2** — nx2pyg roundtrip, simulate, end-to-end train smoke
4. **P3** — subset, construct, coarsen
5. **P4** — `_kwargs_utils`, explainers (build on trained toy model fixture)
6. **P5** — EarlyStopper, optim utils, inferrers
7. **P6** — OT/external as needed

---

## Shared fixtures to add (`conftest.py`)

| Fixture | Purpose |
|---------|---------|
| `minimal_edge_index_dict` | 1-input, 1-function, 1-output graph |
| `small_gsnn_model` | GSNN built from minimal graph, eval mode |
| `small_gsnn_data` | HeteroData from `nx2pyg` on same graph |
| `simulated_3x3_batch` | Output of `simulate_3_in_3_out(n_train=32, n_test=8)` |
| `trained_gsnn_toy` | Few-step trained model for explainer tests |
| `device` | `cpu` default; optional `cuda` skip marker |

---

## Coverage gaps & known risks to test explicitly

- **`GroupLayerNorm.forward`**: uses `mean` before assignment (line 36) — add test that catches NaN after backward
- **`simulate_sde` return order**: docstring says `(x_train, x_test, y_train, y_test)` but implementation may return `(x_train, y_train, x_test, y_test)` — regression test
- **`GSNN.init` doc vs code**: doc mentions `'xavier'`/`'kaiming'` but SparseLinear expects full init names — test accepted values
- **Checkpoint + node_activity**: combined forward/backward under gradient checkpointing
- **Multi-output graphs**: edge2node with several outputs and varying in-degrees

---

## Summary counts

| Priority | Modules | Approx. test cases |
|----------|---------|-------------------|
| P0 | 4 | ~45 |
| P1 | 10 | ~35 |
| P2 | 5 | ~20 |
| P3 | 4 | ~18 |
| P4 | 12 | ~30 |
| P5 | 10 | ~25 |
| P6 | 5 | ~12 |
| **Total** | **~50 files** | **~185** |
