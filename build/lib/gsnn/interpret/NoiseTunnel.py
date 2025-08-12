from typing import Optional

import numpy as np
import pandas as pd
import torch


class NoiseTunnel:
    """Edge-level *NoiseTunnel* wrapper for :class:`IGExplainer` and
    :class:`ContrastiveIGExplainer`.

    This module runs the wrapped explainer multiple times while injecting
    Gaussian noise in the *edge-mask space* and finally aggregates the
    obtained attributions.  The procedure is inspired by *SmoothGrad* /
    *NoiseTunnel* (Smilkov *et al.* 2017) but adapted to GSNNs where the
    *inputs* are the **edge weights** rather than node features.

    Parameters
    ----------
    explainer : IGExplainer or ContrastiveIGExplainer
        A *configured* explainer instance whose ``explain`` method will be
        executed repeatedly.  The explainer **must** expose the underlying
        GSNN model via the attribute ``model``.
    n_samples : int, optional (default=20)
        Number of noisy repetitions.
    noise_std : float, optional (default=0.05)
        Standard deviation of the Gaussian noise added to the edge weights.
    agg : {'mean', 'median'}, optional (default='mean')
        Aggregation statistic used to combine the per-sample attributions.

    Notes
    -----
    1.  For :class:`IGExplainer` we add noise to its *baseline* edge-mask
        (``explainer.baseline``).  This is equivalent to sampling different
        straight-line paths *m(α) = α·(1 + ε)* where ε ~ 𝓝(0, σ²).
    2.  :class:`ContrastiveIGExplainer` does **not** expose a baseline.
        Therefore we perturb the *terminal* mask ``m=1`` only, which yields a
        noisy path *m(α)=α·(1+ε)*.  The implementation copies the internal
        logic of the contrastive explainer because the original method does
        not accept external masks.
    3.  The injected noise is clipped to the valid range \[0, 1\].

    Example
    -------
    >>> ig = ContrastiveIGExplainer(model, data, n_steps=64)
    >>> nt = NoiseTunnel(ig, n_samples=30, noise_std=0.1)
    >>> df = nt.explain(x1, x2, target_idx=0)
    >>> df.sort_values('score', ascending=False).head()
    """

    def __init__(
        self,
        explainer,
        n_samples: int = 20,
        noise_std: float = 0.05,
        agg: str = "mean",
    ) -> None:
        self.explainer = explainer
        self.n_samples = int(n_samples)
        self.noise_std = float(noise_std)
        if agg not in {"mean", "median"}:
            raise ValueError("agg must be 'mean' or 'median'.")
        self.agg = agg

        # Convenience aliases
        self.model = explainer.model
        self.device = next(self.model.parameters()).device
        self.E = self.model.edge_index.size(1)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _sample_noise(self) -> torch.Tensor:
        """Sample a *single* Gaussian noise mask of shape ``(1,E)``."""
        eps = torch.randn((1, self.E), device=self.device) * self.noise_std
        # Clip so the final mask remains inside [0,1]
        return eps

    def _aggregate(self, scores: np.ndarray) -> np.ndarray:
        """Aggregate along axis=0 (samples)."""
        if self.agg == "mean":
            return scores.mean(axis=0)
        else:  # median
            return np.median(scores, axis=0)

    # ------------------------------------------------------------------
    # Public API – mirrors the wrapped explainer
    # ------------------------------------------------------------------
    def explain(self, *args, **kwargs) -> pd.DataFrame:  # noqa: D401
        """Compute *noise-tunnel* edge attributions.

        The positional / keyword arguments are forwarded verbatim to the
        wrapped explainer's ``explain`` method.
        """

        # Container for per-sample scores (n_samples x E)
        all_scores = []

        for _ in range(self.n_samples):
            # ----------------------------------------------------------
            # 1) Inject noise into the edge-mask space
            # ----------------------------------------------------------
            noise = self._sample_noise()

            # Forward noise to the base explainer via the ``jitter`` kwarg
            df = self.explainer.explain(*args, jitter=noise, **kwargs)

            all_scores.append(df["score"].to_numpy(copy=True))

        # --------------------------------------------------------------
        # Aggregate across noise samples & return DataFrame
        # --------------------------------------------------------------
        score_mat = np.stack(all_scores, axis=0)  # (n_samples , E)
        agg_scores = self._aggregate(score_mat)

        res = df.copy(deep=True)
        res["score"] = agg_scores
        return res 