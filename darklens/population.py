"""Hierarchical (population-level) inference of dark-matter properties.

A single lens constrains warm dark matter only weakly. Many lenses together can rule
models out. Following the importance-reweighting approach for amortised posteriors
(e.g. Wagner-Carena et al. 2021, ApJ 909, 187), each lens's NPE posterior, which was
trained under the interim prior pi, is turned into a likelihood ratio and multiplied
across lenses:

    p(Lambda | {x_i}) ∝ p(Lambda) * prod_i ∫ [q(f, m | x_i) / pi(f, m)] p(f, m | Lambda) df dm

Population model, with hyper-parameters Lambda = (mu_f, log10 M_hm):

* the particle-physics parameter log10 M_hm is **shared** by every lens;
* the substructure fraction varies from lens to lens:
  f_i ~ TruncNormal(mu_f, sigma_f) on the prior range.

Everything is evaluated exactly on a 2-D grid, since the flow gives normalised densities.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .model import DarkLensNPE
from .simulator import LensSimulator
from .substructure import wdm_mass_from_half_mode


@torch.no_grad()
def lens_log_density_grid(model: DarkLensNPE, sim: LensSimulator, images: torch.Tensor,
                          n_f: int = 100, n_m: int = 100, bs: int = 8) -> dict:
    """log q(f, m | x_i) on a (n_f, n_m) grid of cell centres, for every image."""
    dev = next(model.parameters()).device
    lo, hi = sim.dm_prior.low.numpy(), sim.dm_prior.high.numpy()
    f_grid = lo[0] + (np.arange(n_f) + 0.5) / n_f * (hi[0] - lo[0])
    m_grid = lo[1] + (np.arange(n_m) + 0.5) / n_m * (hi[1] - lo[1])
    F_, M_ = np.meshgrid(f_grid, m_grid, indexing="ij")
    pts = torch.tensor(np.stack([F_.ravel(), M_.ravel()], -1), dtype=torch.float32, device=dev)
    G = pts.shape[0]
    out = []
    for i in range(0, len(images), bs):
        x = sim.normalise(images[i:i + bs].to(dev))
        h = model.embed(x)
        b = h.shape[0]
        lp = model.log_prob_dm(pts.repeat(b, 1), h.repeat_interleave(G, 0)).view(b, n_f, n_m)
        out.append(lp.cpu())
    logq = torch.cat(out).double().numpy()
    log_prior = -np.log((hi - lo).prod())
    return {"f_grid": f_grid, "m_grid": m_grid, "logq": logq, "log_interim": log_prior}


def _log_trunc_normal(f: np.ndarray, mu: np.ndarray, sigma: float, lo: float, hi: float) -> np.ndarray:
    """log N_trunc(f | mu, sigma) on [lo, hi]; returns (len(f), len(mu))."""
    from math import erf, sqrt
    z = (f[:, None] - mu[None, :]) / sigma
    norm = np.array([0.5 * (erf((hi - m) / (sigma * sqrt(2))) - erf((lo - m) / (sigma * sqrt(2)))) for m in mu])
    return -0.5 * z**2 - math.log(sigma * math.sqrt(2 * math.pi)) - np.log(norm)[None, :]


def hierarchical_posterior(grid: dict, sigma_f: float, lens_subset: np.ndarray | None = None) -> dict:
    """Posterior over (mu_f, log10 M_hm) on the grid, with a flat hyper-prior.

    sigma_f = 0 means every lens has the same f_sub.
    """
    f, m = grid["f_grid"], grid["m_grid"]
    logq = grid["logq"] if lens_subset is None else grid["logq"][lens_subset]
    log_ratio = logq - grid["log_interim"]                     # (N, n_f, n_m)
    df = f[1] - f[0]
    if sigma_f <= 0:
        per_lens = log_ratio                                    # mu_f == f_i
    else:
        K = _log_trunc_normal(f, f, sigma_f, f[0] - df / 2, f[-1] + df / 2) + np.log(df)  # (f_i, mu)
        # log ∫ exp(log_ratio(f_i, m)) N(f_i | mu) df_i  via log-sum-exp over f_i
        per_lens = np.empty_like(log_ratio)
        for i in range(len(log_ratio)):                        # loop keeps memory O(n_f^2 n_m)
            a = log_ratio[i][:, None, :] + K[:, :, None]       # (f_i, mu, m)
            amax = a.max(0, keepdims=True)
            per_lens[i] = (amax + np.log(np.exp(a - amax).sum(0, keepdims=True)))[0]
    logpost = per_lens.sum(0)
    logpost -= logpost.max()
    post = np.exp(logpost)
    post /= post.sum() * df * (m[1] - m[0])
    return {"mu_f": f, "log10_mhm": m, "posterior": post,
            "marg_mhm": post.sum(0) * df, "marg_f": post.sum(1) * (m[1] - m[0])}


def credible_upper_limit(x: np.ndarray, pdf: np.ndarray, level: float = 0.95) -> float:
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(np.interp(level, cdf, x))


def credible_interval(x: np.ndarray, pdf: np.ndarray, level: float = 0.68) -> tuple[float, float]:
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return float(np.interp((1 - level) / 2, cdf, x)), float(np.interp((1 + level) / 2, cdf, x))


@torch.no_grad()
def simulate_population(sim: LensSimulator, n_lenses: int, mu_f: float, sigma_f: float,
                        log10_mhm: float, seed: int = 7, bs: int = 64) -> dict:
    """Mock survey: every lens shares log10 M_hm, and f_sub_i ~ TruncNormal(mu_f, sigma_f)."""
    g = torch.Generator().manual_seed(seed)
    lo, hi = float(sim.dm_prior.low[0]), float(sim.dm_prior.high[0])
    f = torch.empty(0)
    while len(f) < n_lenses:  # rejection sampling of the truncated normal
        draw = mu_f + sigma_f * torch.randn(4 * n_lenses, generator=g)
        f = torch.cat([f, draw[(draw >= lo) & (draw <= hi)]])
    f = f[:n_lenses]
    imgs = []
    for i in range(0, n_lenses, bs):
        b = min(bs, n_lenses - i)
        out = sim.simulate(b, g, overrides={"f_sub": f[i:i + b], "log10_mhm": log10_mhm})
        imgs.append(out["image"].cpu())
    return {"images": torch.cat(imgs), "f_sub": f.numpy(), "log10_mhm": log10_mhm,
            "mu_f": mu_f, "sigma_f": sigma_f}


def mhm_limit_to_wdm_mass(log10_mhm_upper: float) -> float:
    """An upper limit on M_hm becomes a *lower* limit on the thermal-relic WDM mass [keV]."""
    return float(wdm_mass_from_half_mode(10**log10_mhm_upper))
