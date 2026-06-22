"""
Frame-level importance weighting for TTT3R state updates.

Computes a scalar weight per frame from frozen encoder patch tokens:
    w_t = alpha * w_freq + (1 - alpha) * w_surprise
"""

import torch


class FrameImportanceEstimator:
    """
    Training-free frame-level importance estimator.

    The estimator expects spatial patch tokens only, before pose/register tokens are
    appended elsewhere in the model.
    """

    def __init__(
        self,
        alpha=0.5,
        device="cuda",
        max_history=None,
        min_weight=0.05,
    ):
        """
        Args:
            alpha: balance between frequency importance and surprise, in [0, 1].
            device: torch device.
            max_history: kept for CLI compatibility; unused.
            min_weight: lower clamp so the state never fully stops updating.
        """
        self.alpha = float(alpha)
        self.device = device
        self.min_weight = float(min_weight)

        self.last_stats = {}

    def reset(self):
        """Reset estimator state."""
        self.last_stats = {}

    def compute_frequency_importance(self, tokens):
        """
        Compute texture/spatial diversity from patch token variance.

        Args:
            tokens: spatial encoder patch tokens, shape [B, P, D].

        Returns:
            Scalar tensor in [0, 1].
        """
        token_mean = tokens.mean(dim=1, keepdim=True)
        variance = ((tokens - token_mean) ** 2).mean()
        return torch.sigmoid(variance.sqrt() - 1.0)

    def combine_weight(self, w_freq, w_surprise):
        """Combine frequency importance and decoder surprise into final w_t."""
        w_surprise = w_surprise.to(device=w_freq.device, dtype=w_freq.dtype)
        w_t = self.alpha * w_freq + (1.0 - self.alpha) * w_surprise
        return w_t.clamp(self.min_weight, 1.0)

    def compute_weight(self, tokens, w_surprise):
        """
        Compute combined frame-level importance.

        Args:
            tokens: spatial encoder patch tokens, shape [B, P, D].
            w_surprise: scalar tensor from decoder reconstruction surprise.

        Returns:
            Scalar tensor in [min_weight, 1].
        """
        with torch.no_grad():
            tokens = tokens.detach()
            w_freq = self.compute_frequency_importance(tokens)
            w_t = self.combine_weight(w_freq, w_surprise)

            self.last_stats = {
                "w_freq": float(w_freq.detach().cpu()),
                "w_surprise": float(w_surprise.detach().cpu()),
                "w_t": float(w_t.detach().cpu()),
            }

        return w_t
