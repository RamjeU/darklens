"""Conditional neural spline flows, written from scratch.

The flow is a stack of affine-free *rational-quadratic spline coupling* layers
(Durkan, Bekasov, Murray & Papamakarios 2019, "Neural Spline Flows"). Each layer
leaves half of the dimensions unchanged and transforms the other half with a
monotone piecewise rational-quadratic spline. The spline's knots are predicted by an
MLP that sees the unchanged half and the context vector (the image embedding).
Outside ``[-tail_bound, tail_bound]`` the transform is the identity, so it is a
bijection on all of R.

The spline follows the equations of Durkan et al. (2019) and the conventions of the
authors' MIT-licensed reference library ``nflows`` (github.com/bayesiains/nflows):
minimum bin sizes, and boundary derivatives padded so the slope is exactly 1 where
the spline meets the identity tails. The code here was written independently.

Direction convention: ``forward`` maps data z -> base noise u. It is used for
density evaluation, which is what training needs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ----------------------------------------------------------------------------
# Rational-quadratic spline (Gregory & Delbourgo 1982; Durkan et al. 2019)
# ----------------------------------------------------------------------------
def _search(bin_edges: Tensor, x: Tensor) -> Tensor:
    # Comparison-sum instead of torch.searchsorted, so it runs on every backend (incl. MPS).
    return (x[..., None] >= bin_edges).sum(-1) - 1


def rq_spline(x: Tensor, w: Tensor, h: Tensor, d: Tensor, inverse: bool = False,
              tail_bound: float = 5.0, min_w: float = 1e-3, min_h: float = 1e-3,
              min_d: float = 1e-3) -> tuple[Tensor, Tensor]:
    """Element-wise monotone RQ spline with identity tails.

    x: (..., ) values; w, h: (..., K) unnormalised widths and heights;
    d: (..., K - 1) unnormalised interior derivatives.
    Returns (y, log|dy/dx|), with the log-determinant always taken in the
    *forward* direction's sign convention for the direction requested.
    """
    K = w.shape[-1]
    inside = (x > -tail_bound) & (x < tail_bound)
    out = x.clone()
    logdet = torch.zeros_like(x)
    if not inside.any():
        return out, logdet

    xi, w, h, d = x[inside], w[inside], h[inside], d[inside]
    B = tail_bound

    widths = min_w + (1 - min_w * K) * F.softmax(w, -1)
    cw = F.pad(torch.cumsum(widths, -1), (1, 0))
    cw = 2 * B * cw - B
    cw[..., 0], cw[..., -1] = -B, B
    widths = cw[..., 1:] - cw[..., :-1]

    heights = min_h + (1 - min_h * K) * F.softmax(h, -1)
    ch = F.pad(torch.cumsum(heights, -1), (1, 0))
    ch = 2 * B * ch - B
    ch[..., 0], ch[..., -1] = -B, B
    heights = ch[..., 1:] - ch[..., :-1]

    # Boundary derivatives are fixed to 1 so the spline joins the identity tails smoothly.
    const = math.log(math.exp(1 - min_d) - 1)
    d = F.pad(d, (1, 1), value=const)
    derivs = min_d + F.softplus(d)

    edges = ch if inverse else cw
    edges_search = edges.clone()
    edges_search[..., -1] += 1e-6
    idx = _search(edges_search, xi).clamp(0, K - 1)[..., None]

    xk = cw.gather(-1, idx)[..., 0]
    wk = widths.gather(-1, idx)[..., 0]
    yk = ch.gather(-1, idx)[..., 0]
    hk = heights.gather(-1, idx)[..., 0]
    dk = derivs.gather(-1, idx)[..., 0]
    dk1 = derivs[..., 1:].gather(-1, idx)[..., 0]
    delta = hk / wk

    if not inverse:
        t = (xi - xk) / wk
        t1t = t * (1 - t)
        num = hk * (delta * t**2 + dk * t1t)
        den = delta + (dk + dk1 - 2 * delta) * t1t
        y = yk + num / den
        dnum = delta**2 * (dk1 * t**2 + 2 * delta * t1t + dk * (1 - t) ** 2)
        ld = torch.log(dnum) - 2 * torch.log(den)
    else:
        dy = xi - yk
        a = dy * (dk + dk1 - 2 * delta) + hk * (delta - dk)
        b = hk * dk - dy * (dk + dk1 - 2 * delta)
        c = -delta * dy
        disc = (b**2 - 4 * a * c).clamp(min=0)
        t = (2 * c) / (-b - torch.sqrt(disc))
        y = t * wk + xk
        t1t = t * (1 - t)
        den = delta + (dk + dk1 - 2 * delta) * t1t
        dnum = delta**2 * (dk1 * t**2 + 2 * delta * t1t + dk * (1 - t) ** 2)
        ld = -(torch.log(dnum) - 2 * torch.log(den))

    out[inside] = y
    logdet[inside] = ld
    return out, logdet


# ----------------------------------------------------------------------------
# Coupling layer and flow
# ----------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int, n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden),
                          nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, d_out)
        nn.init.zeros_(self.out.weight)   # start near the identity transform
        nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor) -> Tensor:
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out(F.gelu(h))


class SplineCoupling(nn.Module):
    def __init__(self, dim: int, context_dim: int, mask: Tensor, hidden: int = 128,
                 n_bins: int = 8, tail_bound: float = 5.0, n_blocks: int = 2):
        super().__init__()
        self.register_buffer("mask", mask.bool())
        self.n_id = int(self.mask.sum())
        self.n_tr = dim - self.n_id
        self.K = n_bins
        self.tail = tail_bound
        self.net = ResidualMLP(self.n_id + context_dim, self.n_tr * (3 * n_bins - 1), hidden, n_blocks)

    def _params(self, z_id: Tensor, ctx: Tensor):
        p = self.net(torch.cat([z_id, ctx], -1)).view(-1, self.n_tr, 3 * self.K - 1)
        K = self.K
        scale = math.sqrt(self.net.inp.out_features)  # keep softmax logits O(1)
        return p[..., :K] / scale, p[..., K:2 * K] / scale, p[..., 2 * K:]

    def _transform(self, z: Tensor, ctx: Tensor, inverse: bool):
        z_id, z_tr = z[:, self.mask], z[:, ~self.mask]
        w, h, d = self._params(z_id, ctx)
        y_tr, ld = rq_spline(z_tr, w, h, d, inverse=inverse, tail_bound=self.tail)
        out = torch.empty_like(z)
        out[:, self.mask] = z_id
        out[:, ~self.mask] = y_tr
        return out, ld.sum(-1)

    def forward(self, z, ctx):
        return self._transform(z, ctx, inverse=False)

    def inverse(self, u, ctx):
        return self._transform(u, ctx, inverse=True)


class ConditionalSplineFlow(nn.Module):
    """q(z | context) as a stack of spline couplings over a standard normal base."""

    def __init__(self, dim: int, context_dim: int, n_layers: int = 6, hidden: int = 128,
                 n_bins: int = 8, tail_bound: float = 5.0):
        super().__init__()
        self.dim = dim
        layers = []
        for i in range(n_layers):
            if dim < 2:
                raise ValueError("coupling flows need dim >= 2")
            # Alternate checkerboard and half-split masks (each followed by its
            # complement) so every dimension is transformed and all pairs interact.
            if i % 4 in (0, 1):
                mask = (torch.arange(dim) % 2 == i % 2).float()
            else:
                mask = (torch.arange(dim) < dim // 2).float()
                if i % 2:
                    mask = 1 - mask
            layers.append(SplineCoupling(dim, context_dim, mask, hidden, n_bins, tail_bound))
        self.layers = nn.ModuleList(layers)

    def log_prob(self, z: Tensor, ctx: Tensor) -> Tensor:
        total = torch.zeros(z.shape[0], device=z.device)
        for layer in self.layers:
            z, ld = layer(z, ctx)
            total = total + ld
        base = -0.5 * (z**2).sum(-1) - 0.5 * self.dim * math.log(2 * math.pi)
        return base + total

    @torch.no_grad()
    def sample(self, n: int, ctx: Tensor) -> Tensor:
        """Draw n samples per context row. Returns (B, n, dim)."""
        B = ctx.shape[0]
        ctx_rep = ctx.repeat_interleave(n, 0)
        u = torch.randn(B * n, self.dim, device=ctx.device)
        for layer in reversed(self.layers):
            u, _ = layer.inverse(u, ctx_rep)
        return u.view(B, n, self.dim)
