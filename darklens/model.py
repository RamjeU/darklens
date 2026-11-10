"""Neural posterior estimator: a ResNet image encoder feeding two conditional spline flows.

Two posterior heads share one image embedding:

* the **dark-matter head** models q(f_sub, log10 M_hm | image). This is the science
  target. Because it is trained on only these two parameters, it directly learns
  the posterior *marginalised* over every nuisance parameter: source light, the
  exact subhalo realisation, noise, and the macro-model.
* the **macro head** models q(theta_E, e1, e2, gamma1, gamma2, x_L, y_L | image).

Training minimises the expected negative log posterior density
E_{theta ~ prior, x ~ simulator(theta)}[-log q(theta | x)]. Its minimiser is the true
posterior (Papamakarios & Murray 2016; Lueckmann et al. 2017; Greenberg et al. 2019).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .flows import ConditionalSplineFlow
from .priors import BoxUniform


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, c_out)
        self.skip = (nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.GroupNorm(8, c_out))
                     if (stride != 1 or c_in != c_out) else nn.Identity())

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        return F.gelu(h + self.skip(x))


class LensEncoder(nn.Module):
    """ResNet-style encoder. It flattens the final feature map instead of pooling it,
    so the embedding keeps the absolute position information that the lens centre
    and orientation depend on."""

    def __init__(self, npix: int = 64, widths=(32, 64, 128, 256), embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 3, 1, 1, bias=False), nn.GroupNorm(8, widths[0]), nn.GELU())
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks += [ResBlock(c, w, stride=1 if i == 0 else 2), ResBlock(w, w)]
            c = w
        self.blocks = nn.Sequential(*blocks)
        side = npix // 2 ** (len(widths) - 1)
        self.reduce = nn.Conv2d(c, 64, 1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * side * side, 512),
                                  nn.GELU(), nn.Linear(512, embed_dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.reduce(self.blocks(self.stem(x))))


class DarkLensNPE(nn.Module):
    def __init__(self, dm_prior: BoxUniform, macro_prior: BoxUniform, npix: int = 64,
                 embed_dim: int = 256, flow_layers_dm: int = 6, flow_layers_macro: int = 8,
                 flow_hidden: int = 128, n_bins: int = 8, encoder_widths=(32, 64, 128, 256),
                 tail_bound: float = 10.0):
        super().__init__()
        self.dm_prior, self.macro_prior = dm_prior, macro_prior
        self.encoder = LensEncoder(npix, tuple(encoder_widths), embed_dim)
        self.flow_dm = ConditionalSplineFlow(dm_prior.dim, embed_dim, flow_layers_dm, flow_hidden, n_bins, tail_bound)
        self.flow_macro = ConditionalSplineFlow(macro_prior.dim, embed_dim, flow_layers_macro, flow_hidden, n_bins,
                                                tail_bound)
        self.tail_bound = tail_bound

    # -------------------------------------------------------------- training
    def loss(self, x: Tensor, theta_dm: Tensor, theta_macro: Tensor, dm_weight: float = 1.0) -> dict[str, Tensor]:
        """Negative log-density in the unbounded (logit) space, averaged over the batch.

        ``dm_weight`` up-weights the dark-matter head's gradient into the shared encoder.
        Each head is still minimised by its own true posterior, so the weight changes
        which features the encoder prioritises, not what the heads converge to.
        """
        h = self.encoder(x)
        z_dm = self.dm_prior.to_unbounded(theta_dm)
        z_ma = self.macro_prior.to_unbounded(theta_macro)
        nll_dm = -self.flow_dm.log_prob(z_dm, h).mean()
        nll_ma = -self.flow_macro.log_prob(z_ma, h).mean()
        return {"loss": dm_weight * nll_dm + nll_ma, "nll_dm": nll_dm, "nll_macro": nll_ma}

    def prior_baseline(self) -> dict[str, float]:
        """Lowest NLL a head can reach while *ignoring the image* (i.e. predicting the prior).

        A uniform prior becomes a standard logistic in logit space, but beyond
        ``tail_bound`` the flow falls back to its Gaussian base, so the floor is
        slightly above the logistic entropy (2 nats per dimension). A head's
        validation NLL must drop below this number before it has learned anything
        from the images.
        """
        B = self.tail_bound
        z = np.linspace(-B, B, 200_001)
        p = 1.0 / (4 * np.cosh(z / 2) ** 2)                     # logistic pdf
        m = p.sum() * (z[1] - z[0])
        inside = -(p * np.log(p / m)).sum() * (z[1] - z[0])
        zt = np.linspace(B, B + 60, 200_001)
        pt = 1.0 / (4 * np.cosh(zt / 2) ** 2)
        outside = 2 * (pt * (0.5 * zt**2 + 0.5 * np.log(2 * np.pi))).sum() * (zt[1] - zt[0])
        per_dim = float(inside + outside)
        return {"dm": per_dim * self.dm_prior.dim, "macro": per_dim * self.macro_prior.dim}

    # ------------------------------------------------------------- inference
    def embed(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def log_prob_dm(self, theta_dm: Tensor, h: Tensor) -> Tensor:
        """log q(theta_dm | x) in *physical* parameter space."""
        z = self.dm_prior.to_unbounded(theta_dm)
        return self.flow_dm.log_prob(z, h) + self.dm_prior.log_abs_det_dz_dtheta(theta_dm)

    def log_prob_macro(self, theta_macro: Tensor, h: Tensor) -> Tensor:
        z = self.macro_prior.to_unbounded(theta_macro)
        return self.flow_macro.log_prob(z, h) + self.macro_prior.log_abs_det_dz_dtheta(theta_macro)

    @torch.no_grad()
    def sample_dm(self, h: Tensor, n: int) -> Tensor:
        return self.dm_prior.from_unbounded(self.flow_dm.sample(n, h))

    @torch.no_grad()
    def sample_macro(self, h: Tensor, n: int) -> Tensor:
        return self.macro_prior.from_unbounded(self.flow_macro.sample(n, h))
