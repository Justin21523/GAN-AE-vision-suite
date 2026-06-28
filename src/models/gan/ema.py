"""
Exponential Moving Average (EMA) helper.

EMA is a common stabilization technique for GANs:
- During training, keep a smoothed copy of generator weights.
- During sampling/evaluation, temporarily swap to EMA weights for higher quality.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        # Keep a full `state_dict` shadow (params + buffers) so sampling can use
        # consistent BatchNorm running stats as well.
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}

    def register(self):
        """Initialize EMA shadow weights from the model's current parameters."""
        self.shadow = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def update(self):
        """Update EMA shadow weights after each optimizer step."""
        if not self.shadow:
            self.register()

        current = self.model.state_dict()
        for key, value in current.items():
            if key not in self.shadow:
                self.shadow[key] = value.detach().clone()
                continue

            if not torch.is_floating_point(value):
                self.shadow[key] = value.detach().clone()
                continue

            new_avg = (1.0 - self.decay) * value.detach() + self.decay * self.shadow[key]
            self.shadow[key] = new_avg.clone()

    def apply_shadow(self):
        """Swap model parameters/buffers to their EMA (shadow) values."""
        if not self.shadow:
            self.register()
        self.backup = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.model.load_state_dict(self.shadow, strict=False)

    def restore(self):
        """Restore original (non-EMA) parameters after `apply_shadow()`."""
        if self.backup:
            self.model.load_state_dict(self.backup, strict=False)
        self.backup = {}

    def load_shadow(self, shadow: Dict[str, torch.Tensor], device: Optional[torch.device] = None):
        """
        Load a previously-saved EMA shadow state.

        Args:
            shadow: A `state_dict`-like mapping.
            device: If provided, move shadow tensors to this device.
        """
        dev = device or next(self.model.parameters()).device
        self.shadow = {k: v.detach().to(dev) for k, v in shadow.items()}
