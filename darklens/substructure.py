"""Dark-matter subhalo populations, including warm-dark-matter suppression.

Model
-----
CDM subhalos follow a power-law mass function, dN/dm ∝ m^gamma with gamma = -1.9
(Springel et al. 2008), between ``m_min`` and ``m_max``. They are scattered
uniformly over a disk of radius ``R_render`` centred on the main deflector.

The normalisation is the *CDM-equivalent projected substructure mass fraction*

    f_sub = M_sub,CDM(< R_render) / M_main(< R_render),

so f_sub = 0.01 means that 1% of the projected mass near the lens would sit in
subhalos if dark matter were cold.

Warm dark matter (WDM) erases small-scale power below the free-streaming scale. Its
effect on the subhalo mass function is modelled as a multiplicative suppression,
fitted to WDM N-body simulations by Lovell (2020, ApJ 897, 147):

    (dN/dm)_WDM = (dN/dm)_CDM * [1 + (alpha M_hm / m)^beta]^gamma,
    alpha = 4.2, beta = 2.5, gamma = -0.2   (subhalos),

where M_hm is the *half-mode mass*. CDM corresponds to M_hm -> 0. The suppression
is applied by thinning a CDM draw, so f_sub and M_hm are separate parameters.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .cosmology import FlatLCDM, LensGeometry


# ----------------------------------------------------------------------------
# Mass function helpers
# ----------------------------------------------------------------------------
def powerlaw_icdf(u: Tensor, m_min: float, m_max: float, slope: float) -> Tensor:
    """Inverse CDF of dN/dm ∝ m^slope on [m_min, m_max] (slope != -1)."""
    s = slope + 1.0
    lo, hi = m_min**s, m_max**s
    return (lo + u * (hi - lo)) ** (1.0 / s)


def powerlaw_mean_mass(m_min: float, m_max: float, slope: float) -> float:
    s1, s2 = slope + 1.0, slope + 2.0
    num = (m_max**s2 - m_min**s2) / s2
    den = (m_max**s1 - m_min**s1) / s1
    return num / den


def wdm_suppression(m: Tensor, log10_mhm: Tensor, alpha: float = 4.2, beta: float = 2.5,
                    gamma: float = -0.2) -> Tensor:
    """Fraction of CDM subhalos of mass m that survive in WDM (Lovell 2020 subhalo fit)."""
    ratio = alpha * 10.0 ** (log10_mhm - torch.log10(m))
    return (1.0 + ratio**beta) ** gamma


# ----------------------------------------------------------------------------
# Half-mode mass <-> thermal-relic WDM particle mass
# ----------------------------------------------------------------------------
def half_mode_mass(m_wdm_keV: float | np.ndarray, cosmo: FlatLCDM = FlatLCDM()) -> np.ndarray:
    """Half-mode mass [M_sun] for a thermal-relic WDM particle of mass m_wdm [keV].

    Uses the transfer-function fit of Viel et al. (2005) and the half-mode
    definition of Schneider et al. (2012):
        alpha   = 0.049 (m/keV)^-1.11 (Omega_wdm/0.25)^0.11 (h/0.7)^1.22  [Mpc/h]
        lambda_hm = 2 pi alpha (2^(nu/5) - 1)^(-1/(2 nu)),  nu = 1.12
        M_hm    = (4 pi / 3) rho_m (lambda_hm / 2)^3
    """
    m = np.asarray(m_wdm_keV, dtype=np.float64)
    h = cosmo.H0 / 100.0
    omega_dm = cosmo.Om0 - 0.049  # subtract baryons
    nu = 1.12
    alpha = 0.049 * m**-1.11 * (omega_dm / 0.25) ** 0.11 * (h / 0.7) ** 1.22 / h  # [Mpc]
    lam_hm = 2 * np.pi * alpha * (2 ** (nu / 5) - 1) ** (-1 / (2 * nu))
    rho_m = cosmo.Om0 * 2.775e11 * h**2  # [M_sun / Mpc^3]
    return (4 * np.pi / 3) * rho_m * (lam_hm / 2) ** 3


def wdm_mass_from_half_mode(m_hm: float | np.ndarray, cosmo: FlatLCDM = FlatLCDM()) -> np.ndarray:
    """Invert ``half_mode_mass`` (it is a pure power law in m_wdm)."""
    ref = half_mode_mass(1.0, cosmo)
    return (np.asarray(m_hm, dtype=np.float64) / ref) ** (1.0 / (-3 * 1.11))


# ----------------------------------------------------------------------------
# Subhalo population sampler
# ----------------------------------------------------------------------------
@dataclass
class SubhaloBatch:
    """Padded batch of subhalos. Unused slots have ``mask == 0`` and b == 0."""

    x: Tensor      # (B, S) arcsec
    y: Tensor      # (B, S) arcsec
    b: Tensor      # (B, S) pseudo-Jaffe Einstein radius [arcsec]
    a: Tensor      # (B, S) truncation radius [arcsec]
    mass: Tensor   # (B, S) M_sun
    mask: Tensor   # (B, S) {0, 1}

    @property
    def n(self) -> Tensor:
        return self.mask.sum(-1)

    def to(self, device) -> "SubhaloBatch":
        return SubhaloBatch(*(t.to(device) for t in (self.x, self.y, self.b, self.a, self.mass, self.mask)))


@dataclass
class SubhaloPopulation:
    geometry: LensGeometry
    m_min: float = 1e7
    m_max: float = 1e10
    slope: float = -1.9
    R_render: float = 2.0       # arcsec; disk of subhalos around the lens centre
    max_subhalos: int = 512

    def __post_init__(self):
        self._sigma = self.geometry.sigma_crit_arcsec2
        self._mean_m = powerlaw_mean_mass(self.m_min, self.m_max, self.slope)

    def expected_number(self, theta_E: Tensor, f_sub: Tensor) -> Tensor:
        """Expected CDM subhalo count inside R_render.

        The main-deflector mass inside R is approximated by its isothermal value
        M(<R) = pi * theta_E * R * Sigma_cr.
        """
        m_main = math.pi * theta_E * self.R_render * self._sigma
        return f_sub * m_main / self._mean_m

    def mass_to_pseudo_jaffe(self, mass: Tensor, theta_E: Tensor) -> tuple[Tensor, Tensor]:
        """Tidal truncation a = sqrt(b * theta_E) (Metcalf & Madau 2001), so that
        M = pi Sigma_cr b a  ->  b = (M / (pi Sigma_cr sqrt(theta_E)))^(2/3)."""
        b = (mass / (math.pi * self._sigma * torch.sqrt(theta_E))) ** (2.0 / 3.0)
        a = torch.sqrt(b * theta_E)
        return b, a

    @torch.no_grad()
    def sample(self, theta_E: Tensor, f_sub: Tensor, log10_mhm: Tensor, cx: Tensor, cy: Tensor,
               generator: torch.Generator | None = None) -> SubhaloBatch:
        """Draw one subhalo realisation per batch element (on CPU)."""
        B = theta_E.shape[0]
        lam = self.expected_number(theta_E, f_sub)
        n = torch.poisson(lam, generator=generator).long().clamp(max=self.max_subhalos)
        S = max(int(n.max().item()), 1)
        slot = torch.arange(S)[None, :]
        mask = (slot < n[:, None]).float()

        u = torch.rand(B, S, generator=generator)
        mass = powerlaw_icdf(u, self.m_min, self.m_max, self.slope)
        keep = torch.rand(B, S, generator=generator) < wdm_suppression(mass, log10_mhm[:, None])
        mask = mask * keep.float()

        r = self.R_render * torch.sqrt(torch.rand(B, S, generator=generator))
        ang = 2 * math.pi * torch.rand(B, S, generator=generator)
        x = cx[:, None] + r * torch.cos(ang)
        y = cy[:, None] + r * torch.sin(ang)

        b, a = self.mass_to_pseudo_jaffe(mass, theta_E[:, None])
        b = b * mask
        return SubhaloBatch(x=x, y=y, b=b, a=a, mass=mass * mask, mask=mask)
