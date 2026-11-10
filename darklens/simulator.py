"""Batched, differentiable strong-lensing simulator in PyTorch.

Forward model for one lens system:

1. **Main deflector**: a singular isothermal ellipsoid (SIE) plus external shear.
2. **Dark-matter substructure**: Poisson-sampled pseudo-Jaffe subhalos drawn from a
   CDM mass function and thinned by warm-dark-matter suppression (see
   ``substructure.py``).
3. **Ray tracing**: the lens equation beta = theta - alpha(theta), evaluated on a
   supersampled pixel grid.
4. **Source**: an elliptical Sersic galaxy plus optional compact star-forming clumps.
5. **Instrument**: pixel integration (average pooling), Gaussian PSF, and
   heteroscedastic noise (sky background plus Poisson-like source noise).

Every step is a differentiable PyTorch operation, so gradients flow from pixels back
to the lens parameters. Sampling the subhalo realisation is the only discrete step.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .cosmology import LensGeometry
from .priors import BoxUniform
from .profiles import (gaussian_psf_kernel, pseudo_jaffe_deflection, sersic,
                       shear_deflection, sie_deflection, ellipticity_to_q_phi)
from .substructure import SubhaloBatch, SubhaloPopulation

# Parameters the network is asked to infer, grouped by posterior head.
DM_PARAMS = {
    "f_sub": (0.0, 0.05),          # CDM-equivalent substructure mass fraction
    "log10_mhm": (6.0, 10.0),      # WDM half-mode mass [log10 M_sun]
}
MACRO_PARAMS = {
    "theta_E": (0.8, 1.4),         # Einstein radius [arcsec]
    "e1": (-0.25, 0.25),
    "e2": (-0.25, 0.25),
    "gamma1": (-0.08, 0.08),       # external shear
    "gamma2": (-0.08, 0.08),
    "x_lens": (-0.08, 0.08),       # lens centre [arcsec]
    "y_lens": (-0.08, 0.08),
}
# Nuisance parameters: these are simulated and then marginalised over by the network.
NUISANCE_PARAMS = {
    "x_src": (-0.2, 0.2),          # source position relative to lens centre [arcsec]
    "y_src": (-0.2, 0.2),
    "R_e_src": (0.10, 0.35),       # Sersic half-light radius [arcsec]
    "n_src": (0.7, 2.5),           # Sersic index
    "e1_src": (-0.3, 0.3),
    "e2_src": (-0.3, 0.3),
    "log10_amp_src": (0.6, 1.5),   # surface brightness at R_e [log10 counts/pixel]
}
LABELS = {
    "f_sub": r"$f_{\rm sub}$", "log10_mhm": r"$\log_{10} M_{\rm hm}$",
    "theta_E": r"$\theta_E$", "e1": r"$e_1$", "e2": r"$e_2$",
    "gamma1": r"$\gamma_1$", "gamma2": r"$\gamma_2$",
    "x_lens": r"$x_{\rm L}$", "y_lens": r"$y_{\rm L}$",
}


@dataclass
class SimConfig:
    npix: int = 64
    pixel_scale: float = 0.065        # arcsec / pixel  (HST WFC3/IR drizzled ~ 0.06-0.08)
    supersample: int = 2
    psf_fwhm: float = 0.08            # arcsec
    z_lens: float = 0.5
    z_source: float = 2.0
    sky_sigma: float = 1.0            # background noise per pixel [counts]
    gain: float = 4.0                 # Poisson term: var = sky^2 + signal / gain
    max_clumps: int = 2               # compact clumps added to the Sersic source
    m_min: float = 1e7
    m_max: float = 1e10
    mf_slope: float = -1.9
    R_render: float = 2.0
    max_subhalos: int = 512
    dm_prior: dict = field(default_factory=lambda: dict(DM_PARAMS))
    macro_prior: dict = field(default_factory=lambda: dict(MACRO_PARAMS))
    nuisance_prior: dict = field(default_factory=lambda: dict(NUISANCE_PARAMS))
    chunk_elems: int = 2**25          # memory budget for the subhalo deflection sum

    @classmethod
    def from_dict(cls, d: dict | None) -> "SimConfig":
        d = dict(d or {})
        for k in ("dm_prior", "macro_prior", "nuisance_prior"):
            if k in d:
                base = dict(getattr(cls(), k))
                base.update({kk: tuple(v) for kk, v in d[k].items()})
                d[k] = base
        return cls(**d)


class LensSimulator:
    def __init__(self, config: SimConfig | None = None, device: str | torch.device = "cpu"):
        self.cfg = cfg = config or SimConfig()
        self.device = torch.device(device)
        self.geometry = LensGeometry(cfg.z_lens, cfg.z_source)
        self.population = SubhaloPopulation(self.geometry, cfg.m_min, cfg.m_max, cfg.mf_slope,
                                            cfg.R_render, max_subhalos=cfg.max_subhalos)
        self.dm_prior = BoxUniform.from_dict(cfg.dm_prior, LABELS)
        self.macro_prior = BoxUniform.from_dict(cfg.macro_prior, LABELS)
        self.nuisance_prior = BoxUniform.from_dict(cfg.nuisance_prior)

        # Supersampled pixel-centre grid, flattened to (1, P).
        n = cfg.npix * cfg.supersample
        d = cfg.pixel_scale / cfg.supersample
        ax = (torch.arange(n, dtype=torch.float32) - (n - 1) / 2) * d
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        self.gx = xx.reshape(1, -1).to(self.device)
        self.gy = yy.reshape(1, -1).to(self.device)
        self.psf = gaussian_psf_kernel(cfg.psf_fwhm, cfg.pixel_scale).to(self.device)
        self.fov = cfg.npix * cfg.pixel_scale

    # ------------------------------------------------------------------ priors
    def sample_parameters(self, n: int, generator: torch.Generator | None = None,
                          overrides: dict[str, float | Tensor] | None = None) -> dict[str, Tensor]:
        """Sample all parameters from the prior (CPU). ``overrides`` fixes chosen ones."""
        out: dict[str, Tensor] = {}
        for prior in (self.dm_prior, self.macro_prior, self.nuisance_prior):
            s = prior.sample(n, generator)
            out.update({k: s[:, i] for i, k in enumerate(prior.names)})
        for k, v in (overrides or {}).items():
            out[k] = torch.as_tensor(v, dtype=torch.float32).expand(n).clone()
        # Source clumps: up to `max_clumps` compact knots near the source centre.
        C = self.cfg.max_clumps
        if C > 0:
            g = generator
            out["clump_on"] = (torch.rand(n, C, generator=g) < 0.5).float()
            out["clump_dx"] = (torch.rand(n, C, generator=g) - 0.5) * 0.4
            out["clump_dy"] = (torch.rand(n, C, generator=g) - 0.5) * 0.4
            out["clump_Re"] = 0.02 + 0.05 * torch.rand(n, C, generator=g)
            out["clump_amp"] = 0.5 + 1.5 * torch.rand(n, C, generator=g)  # relative to host
        return out

    def theta_dm(self, p: dict[str, Tensor]) -> Tensor:
        return torch.stack([p[k] for k in self.dm_prior.names], -1)

    def theta_macro(self, p: dict[str, Tensor]) -> Tensor:
        return torch.stack([p[k] for k in self.macro_prior.names], -1)

    def sample_subhalos(self, p: dict[str, Tensor], generator=None) -> SubhaloBatch:
        return self.population.sample(p["theta_E"], p["f_sub"], p["log10_mhm"],
                                      p["x_lens"], p["y_lens"], generator)

    # --------------------------------------------------------------- physics
    def deflection(self, p: dict[str, Tensor], subs: SubhaloBatch | None) -> tuple[Tensor, Tensor]:
        c = lambda k: p[k].to(self.device)[:, None]  # noqa: E731  (B, 1)
        x, y = self.gx, self.gy
        ax, ay = sie_deflection(x, y, c("theta_E"), c("e1"), c("e2"), c("x_lens"), c("y_lens"))
        sx, sy = shear_deflection(x - c("x_lens"), y - c("y_lens"), c("gamma1"), c("gamma2"))
        ax, ay = ax + sx, ay + sy
        if subs is not None and subs.mask.sum() > 0:
            # Pack the padded (B, S) subhalo arrays into a flat list of K live
            # subhalos, so the cost scales with the real count rather than B * max(n).
            # Each subhalo's deflection is then scattered back to its own image.
            B = ax.shape[0]
            ax, ay = ax.expand(B, -1), ay.expand(B, -1)
            live = subs.mask.bool()
            bi = live.nonzero()[:, 0].to(self.device)
            sb, sa = subs.b[live].to(self.device), subs.a[live].to(self.device)
            sx, sy = subs.x[live].to(self.device), subs.y[live].to(self.device)
            K, P = sb.shape[0], x.shape[1]
            chunk = max(1, self.cfg.chunk_elems // P)
            for k0 in range(0, K, chunk):
                sl = slice(k0, k0 + chunk)
                dx, dy = pseudo_jaffe_deflection(x, y, sb[sl, None], sa[sl, None], sx[sl, None], sy[sl, None])
                ax = ax.index_add(0, bi[sl], dx)
                ay = ay.index_add(0, bi[sl], dy)
        return ax, ay

    def source_brightness(self, p: dict[str, Tensor], bx: Tensor, by: Tensor) -> Tensor:
        c = lambda k: p[k].to(self.device)[:, None]  # noqa: E731
        sx = c("x_lens") + c("x_src")
        sy = c("y_lens") + c("y_src")
        amp = 10 ** c("log10_amp_src")
        img = sersic(bx, by, amp, c("R_e_src"), c("n_src"), c("e1_src"), c("e2_src"), sx, sy)
        if self.cfg.max_clumps > 0 and "clump_on" in p:
            for j in range(self.cfg.max_clumps):
                cj = lambda k: p[k][:, j].to(self.device)[:, None]  # noqa: E731
                zero = torch.zeros_like(sx)
                img = img + cj("clump_on") * sersic(
                    bx, by, amp * cj("clump_amp"), cj("clump_Re"), torch.ones_like(sx),
                    zero, zero, sx + cj("clump_dx"), sy + cj("clump_dy"))
        return img

    def render(self, p: dict[str, Tensor], subs: SubhaloBatch | None = None, noise: bool = True,
               generator: torch.Generator | None = None) -> Tensor:
        """Return observed images (B, 1, npix, npix) in counts."""
        cfg = self.cfg
        ax, ay = self.deflection(p, subs)
        bx, by = self.gx - ax, self.gy - ay                      # lens equation
        hi = self.source_brightness(p, bx, by)
        B = hi.shape[0]
        n_hi = cfg.npix * cfg.supersample
        img = hi.view(B, 1, n_hi, n_hi)
        img = F.avg_pool2d(img, cfg.supersample)                   # pixel integration
        img = F.conv2d(img, self.psf, padding=self.psf.shape[-1] // 2)
        if noise:
            var = cfg.sky_sigma**2 + img.clamp(min=0) / cfg.gain
            eps = torch.randn(img.shape, generator=generator).to(self.device)
            img = img + var.sqrt() * eps
        return img

    def convergence(self, p: dict[str, Tensor], subs: SubhaloBatch | None = None) -> Tensor:
        """Analytic convergence map kappa (B, 1, npix*ss, npix*ss) for visualisation."""
        c = lambda k: p[k].to(self.device)[:, None]  # noqa: E731
        x, y = self.gx, self.gy
        q, phi = ellipticity_to_q_phi(c("e1"), c("e2"))
        dx, dy = x - c("x_lens"), y - c("y_lens")
        co, si = torch.cos(phi), torch.sin(phi)
        xr, yr = co * dx + si * dy, -si * dx + co * dy
        kap = c("theta_E") / (2 * torch.sqrt(q * xr**2 + yr**2 / q + 1e-6))
        if subs is not None:
            subs = subs.to(self.device)
            r = torch.sqrt((x[:, None] - subs.x[..., None]) ** 2 + (y[:, None] - subs.y[..., None]) ** 2 + 1e-8)
            a = subs.a[..., None]
            kap = kap + (subs.b[..., None] / 2 * (1 / r - 1 / torch.sqrt(r**2 + a**2))).sum(1)
        n = self.cfg.npix * self.cfg.supersample
        return kap.view(-1, 1, n, n)

    # ------------------------------------------------------------ high level
    @torch.no_grad()
    def simulate(self, n: int, generator: torch.Generator | None = None,
                 overrides: dict | None = None, noise: bool = True) -> dict:
        p = self.sample_parameters(n, generator, overrides)
        subs = self.sample_subhalos(p, generator)
        img = self.render(p, subs, noise=noise, generator=generator)
        return {"image": img, "params": p, "subhalos": subs,
                "theta_dm": self.theta_dm(p), "theta_macro": self.theta_macro(p)}

    def normalise(self, img: Tensor) -> Tensor:
        """Network input transform: noise units, then a gentle asinh stretch."""
        return torch.asinh(img / self.cfg.sky_sigma / 3.0)
