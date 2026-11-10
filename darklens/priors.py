"""Box-uniform priors with a smooth bijection to R^d.

The normalising flows model densities on R^d, but physical parameters live in a
bounded prior box. We therefore train the flows on

    z = logit((theta - low) / (high - low))

and convert densities back with the change-of-variables formula

    log p(theta | x) = log q(z | x) + sum_i log |dz_i / dtheta_i|.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

_CLAMP = 1e-6


@dataclass
class BoxUniform:
    names: list[str]
    low: Tensor
    high: Tensor
    labels: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, spec: dict[str, tuple[float, float]], labels: dict[str, str] | None = None):
        names = list(spec)
        low = torch.tensor([spec[k][0] for k in names], dtype=torch.float32)
        high = torch.tensor([spec[k][1] for k in names], dtype=torch.float32)
        labels = labels or {}
        return cls(names, low, high, [labels.get(k, k) for k in names])

    @property
    def dim(self) -> int:
        return len(self.names)

    def _lh(self, ref: Tensor):
        return self.low.to(ref.device, ref.dtype), self.high.to(ref.device, ref.dtype)

    def sample(self, n: int, generator: torch.Generator | None = None) -> Tensor:
        u = torch.rand(n, self.dim, generator=generator)
        return self.low + u * (self.high - self.low)

    def log_prob(self, theta: Tensor) -> Tensor:
        low, high = self._lh(theta)
        inside = ((theta >= low) & (theta <= high)).all(-1)
        lp = -torch.log(high - low).sum()
        return torch.where(inside, lp, torch.full_like(lp, -float("inf")).expand_as(inside))

    def to_unbounded(self, theta: Tensor) -> Tensor:
        low, high = self._lh(theta)
        u = ((theta - low) / (high - low)).clamp(_CLAMP, 1 - _CLAMP)
        return torch.log(u) - torch.log1p(-u)

    def from_unbounded(self, z: Tensor) -> Tensor:
        low, high = self._lh(z)
        return low + torch.sigmoid(z) * (high - low)

    def log_abs_det_dz_dtheta(self, theta: Tensor) -> Tensor:
        """sum_i log |dz_i/dtheta_i| for the logit map."""
        low, high = self._lh(theta)
        u = ((theta - low) / (high - low)).clamp(_CLAMP, 1 - _CLAMP)
        return (-torch.log(high - low) - torch.log(u) - torch.log1p(-u)).sum(-1)
