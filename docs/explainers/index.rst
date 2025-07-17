Explainers
==========

The GSNN library provides a comprehensive suite of explainability methods designed to interpret predictions from Graph Structured Neural Networks. These methods help researchers understand *which edges contribute to predictions*, *how perturbations affect outcomes*, and *what minimal changes would lead to different results*. Each explainer is tailored to different types of questions and provides unique insights into model behavior.

.. note::
    Unlike traditional neural network explainers that focus on input features, GSNN explainers operate on the **edge space** of the graph. This allows direct interpretation of how specific molecular interactions or pathway connections contribute to predictions.

Edge Attribution Methods
------------------------

**Integrated Gradients (IG) Explainer**

The Integrated Gradients explainer computes per-edge attributions by integrating gradients along a straight-line path in **feature space** from a baseline input to the target observation. This method satisfies the completeness axiom, ensuring that the sum of all edge attributions equals the difference between the prediction and baseline prediction.

*What the results indicate:*
  * **Positive scores** indicate edges that contribute positively to the prediction
  * **Negative scores** indicate edges that contribute negatively  
  * **Near-zero scores** indicate edges with minimal impact

*When to use:*
  * Understanding how individual edges contribute to a single prediction
  * When baseline comparisons are meaningful (e.g., comparing to "no activity" state)
  * For generating faithful attributions with theoretical guarantees

*Strengths:*
  * Mathematically principled with completeness guarantees
  * Relatively stable across different baseline choices
  * Computationally efficient for single predictions

*Weaknesses:*
  * Requires careful baseline selection
  * May struggle with highly nonlinear interactions
  * Limited to single-input explanations


**Contrastive Integrated Gradients Explainer**

The Contrastive IG explainer extends Integrated Gradients to **contrastive questions** by attributing the prediction difference Δf = f(x₁) - f(x₂) to individual edges. This method integrates along mask paths while keeping both inputs fixed, making it ideal for understanding why the model predicts differently for two related observations.

*What the results indicate:*
  * **Positive scores** indicate edges that increase the absolute prediction difference |Δf|
  * **Negative scores** indicate edges that decrease the absolute prediction difference
  * **Near-zero scores** indicate edges irrelevant to the contrast

*When to use:*
  * Comparing predictions between related samples (e.g., diseased vs. healthy)
  * Understanding differential pathway activation
  * When interested in *relative* rather than absolute importance

*Strengths:*
  * Principled approach to contrastive explanations
  * Maintains completeness axiom for prediction differences
  * Excellent for comparative analysis

*Weaknesses:*
  * Requires paired observations for meaningful interpretation
  * More computationally expensive than single-input methods
  * May be sensitive to input selection


Direct Perturbation Methods
---------------------------

**Occlusion Explainer**

The Occlusion explainer provides a direct, model-agnostic measure of edge importance by systematically removing each edge and measuring the resulting change in prediction. This "knock-out" approach offers intuitive interpretability by directly quantifying the effect of completely removing each edge.

*What the results indicate:*
  * **Positive scores** indicate edges that contribute positively (removal decreases prediction)
  * **Negative scores** indicate edges that inhibit the prediction (removal increases prediction)
  * **Near-zero scores** indicate edges with no measurable impact

*When to use:*
  * When direct causal interpretation is needed
  * For validating other explanation methods
  * When computational budget allows exhaustive perturbation testing
  * For non-differentiable models or when gradients are unreliable

*Strengths:*
  * Intuitive and direct interpretation
  * Model-agnostic (works with any architecture)
  * Provides clear causal insights
  * No baseline selection required

*Weaknesses:*
  * Computationally expensive (scales linearly with number of edges)
  * May miss interaction effects between edges
  * Can be unstable for models with sharp decision boundaries


**Contrastive Occlusion Explainer**

The Contrastive Occlusion explainer extends the occlusion approach to contrastive scenarios by measuring how edge removal affects the absolute prediction difference between two inputs. This method identifies edges that specifically contribute to differential predictions.

*What the results indicate:*
  * **Positive scores** indicate edges whose removal decreases |Δf| (edge contributes to difference)
  * **Negative scores** indicate edges whose removal increases |Δf| (edge reduces difference)
  * **Near-zero scores** indicate edges with no impact on the prediction difference

*When to use:*
  * Identifying pathway differences between conditions
  * Understanding mechanism-specific effects
  * When gradient-based contrastive methods are not applicable

*Strengths:*
  * Direct measurement of differential importance
  * Model-agnostic approach
  * Clear interpretation for comparative studies

*Weaknesses:*
  * Computationally intensive (quadratic in number of comparisons)
  * Limited to pairwise comparisons
  * May miss subtle interaction effects


Optimization-Based Methods
--------------------------

**GSNN Explainer**

The GSNN explainer learns a sparse binary edge mask that maximizes fidelity to the original prediction while minimizing the number of active edges. Using a differentiable Gumbel-Softmax relaxation, this method identifies the minimal set of edges necessary to reproduce the target prediction.

*What the results indicate:*
  * **Scores near 1** indicate edges essential for reproducing the original prediction
  * **Scores near 0** indicate edges that can be removed with minimal impact
  * The overall mask reveals the **minimal sufficient subgraph** for the prediction

*When to use:*
  * Identifying core pathways or mechanisms
  * When sparsity is desired (e.g., for downstream analysis or intervention)
  * Understanding model redundancy and robustness
  * Generating simplified explanatory models

*Strengths:*
  * Produces inherently sparse explanations
  * Balances fidelity with simplicity
  * Differentiable optimization allows flexible objective functions
  * Can incorporate domain knowledge through constraints

*Weaknesses:*
  * Requires careful hyperparameter tuning (sparsity vs. fidelity trade-off)
  * May converge to local optima
  * Computational overhead for iterative optimization
  * Binary masks may miss nuanced importance gradients


**Counterfactual Explainer**

The Counterfactual explainer learns minimal perturbations to input features that achieve a target prediction. Using gradient descent with L2 regularization, this method answers "what is the smallest change needed to reach a desired outcome?" This approach is particularly valuable for understanding model decision boundaries and generating actionable insights.

*What the results indicate:*
  * **Positive perturbations** indicate features that need to be increased to reach the target
  * **Negative perturbations** indicate features that need to be decreased
  * **Near-zero perturbations** indicate features irrelevant for achieving the target
  * The magnitude indicates how much change is needed

*When to use:*
  * Understanding how to achieve desired outcomes (e.g., therapeutic targets)
  * Identifying minimal interventions
  * Exploring model decision boundaries
  * Generating "what-if" scenarios for intervention planning

*Strengths:*
  * Directly actionable insights for intervention
  * Incorporates minimality constraint naturally
  * Flexible targeting (specific outputs or full prediction vectors)
  * Supports feature masking for constrained optimization

*Weaknesses:*
  * May find local rather than global minima
  * Requires differentiable models
  * Sensitive to hyperparameter choice (learning rate, weight decay)
  * Limited to continuous perturbations


Robustness and Stability Methods
--------------------------------

**Noise Tunnel**

The Noise Tunnel method enhances the stability and robustness of gradient-based explainers by running them multiple times with Gaussian noise injected into the edge-mask space, then aggregating the results. This approach is inspired by SmoothGrad but adapted specifically for GSNN's edge-based architecture.

*What the results indicate:*
  * **Smoothed attribution scores** that are more robust to model sensitivity
  * **Confidence intervals** through multiple noisy samples
  * **Stable rankings** of edge importance less susceptible to noise

*When to use:*
  * When base explainer results are noisy or unstable
  * For more reliable feature selection based on explanations
  * When model has sharp gradients or discontinuities
  * For producing confidence estimates on attributions

*Strengths:*
  * Significantly improves stability of gradient-based methods
  * Provides uncertainty quantification for explanations
  * Can be applied to any gradient-based explainer
  * Helps identify robust vs. artifact attributions

*Weaknesses:*
  * Computationally expensive (multiple runs required)
  * May over-smooth important sharp transitions
  * Requires careful noise level selection
  * Limited to methods that accept noise injection


Choosing the Right Explainer
----------------------------

**For Single Predictions:**
  * Use **IG Explainer** for theoretically grounded attributions with completeness guarantees
  * Use **Occlusion Explainer** for direct, model-agnostic importance measures
  * Use **GSNN Explainer** when you need sparse, minimal explanations
  * Use **Counterfactual Explainer** for actionable intervention insights

**For Comparative Analysis:**
  * Use **Contrastive IG Explainer** for principled differential attribution
  * Use **Contrastive Occlusion Explainer** for model-agnostic comparative analysis

**For Robust Explanations:**
  * Wrap gradient-based methods with **Noise Tunnel** for stability
  * Compare results across multiple explainers for validation

**Computational Considerations:**
  * **Fastest:** IG Explainer, Counterfactual Explainer
  * **Moderate:** GSNN Explainer, Noise Tunnel
  * **Slowest:** Occlusion-based methods (scale with graph size)

.. note::
    **Best Practice**: For critical applications, we recommend using multiple complementary explainers and comparing their results. Convergent findings across different methods provide stronger evidence for interpretation validity. 