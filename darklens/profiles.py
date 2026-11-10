"""Differentiable lens mass profiles and source light profiles (PyTorch).

Conventions
-----------
* Angles are in arcseconds on the sky.
* Every function broadcasts: grid coordinates are typically shaped ``(B, P)`` or
  ``(1, P)`` and parameters ``(B, 1)``.
* Ellipticities use the complex convention ``e = e1 + i e2``, with axis ratio
  ``q = (1 - |e|) / (1 + |e|)`` and position angle ``phi = arg(e) / 2``.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-6


def ellipticity_to_q_phi(e1: Tensor, e2: Tensor, q_max: float = 0.999) -> tuple[Tensor, Tensor]:
    e = torch.sqrt(e1**2 + e2**2).clamp(max=0.95)
    q = ((1 - e) / (1 + e)).clamp(min=0.05, max=q_max)
    phi = 0.5 * torch.atan2(e2, e1)
    return q, phi


def _rotate(x: Tensor, y: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    c, s = torch.cos(phi), torch.sin(phi)
    return c * x + s * y, -s * x + c * y


def _rotate_back(x: Tensor, y: Tensor, phi: Tensor) -> tuple[Tensor, Tensor]:
    c, s = torch.cos(phi), torch.sin(phi)
    return c * x - s * y, s * x + c * y


# ----------------------------------------------------------------------------
# Mass profiles: each returns the deflection field (alpha_x, alpha_y)
# ----------------------------------------------------------------------------
def sie_deflection(x: Tensor, y: Tensor, theta_E: Tensor, e1: Tensor, e2: Tensor,
                   cx: Tensor, cy: Tensor) -> tuple[Tensor, Tensor]:
    """Singular isothermal ellipsoid (Kormann et al. 1994; Keeton 2001).

    Convergence: kappa = theta_E / (2 sqrt(q x'^2 + y'^2 / q)), so ``theta_E`` is
    the Einstein radius along the intermediate axis and the profile reduces to a
    singular isothermal sphere when q -> 1.
    """
    q, phi = ellipticity_to_q_phi(e1, e2)
    xr, yr = _rotate(x - cx, y - cy, phi)
    b = theta_E / torch.sqrt(q)                 # Keeton's normalisation
    f = torch.sqrt(1 - q**2)
    psi = torch.sqrt(q**2 * xr**2 + yr**2 + _EPS**2)
    pref = b * q / f
    ax = pref * torch.atan(f * xr / psi)
    ay = pref * torch.atanh((f * yr / psi).clamp(-1 + 1e-6, 1 - 1e-6))
    return _rotate_back(ax, ay, phi)


def shear_deflection(x: Tensor, y: Tensor, g1: Tensor, g2: Tensor) -> tuple[Tensor, Tensor]:
    """External shear from the lens environment: psi = (g1 (x^2 - y^2) + 2 g2 x y) / 2."""
    return g1 * x + g2 * y, g2 * x - g1 * y


def pseudo_jaffe_deflection(x: Tensor, y: Tensor, b: Tensor, a: Tensor,
                            cx: Tensor, cy: Tensor) -> tuple[Tensor, Tensor]:
    """Truncated singular isothermal sphere (pseudo-Jaffe), used for subhalos.

    kappa(r) = (b / 2) (1/r - 1/sqrt(r^2 + a^2)),
    alpha(r) = b [1 - r / (sqrt(r^2 + a^2) + a)]   (radial).

    The total projected mass is finite: M = pi * b * a * Sigma_cr.
    """
    dx, dy = x - cx, y - cy
    r = torch.sqrt(dx**2 + dy**2 + _EPS**2)
    s = torch.sqrt(r**2 + a**2) + a
    radial = b * (1.0 / r - 1.0 / s)          # alpha(r) / r, written stably
    return radial * dx, radial * dy


def sis_deflection(x: Tensor, y: Tensor, theta_E: Tensor, cx: Tensor, cy: Tensor):
    dx, dy = x - cx, y - cy
    r = torch.sqrt(dx**2 + dy**2 + _EPS**2)
    return theta_E * dx / r, theta_E * dy / r


# ----------------------------------------------------------------------------
# Light profiles
# ----------------------------------------------------------------------------
def sersic_bn(n: Tensor) -> Tensor:
    """Ciotti & Bertin (1999) asymptotic expansion of the Sersic b_n constant."""
    return 2 * n - 1.0 / 3 + 4.0 / (405 * n) + 46.0 / (25515 * n**2)


def sersic(x: Tensor, y: Tensor, amp: Tensor, R_e: Tensor, n: Tensor, e1: Tensor,
           e2: Tensor, cx: Tensor, cy: Tensor) -> Tensor:
    """Elliptical Sersic surface brightness; ``amp`` is the brightness at R_e."""
    q, phi = ellipticity_to_q_phi(e1, e2, q_max=1.0)
    xr, yr = _rotate(x - cx, y - cy, phi)
    R = torch.sqrt(q * xr**2 + yr**2 / q + _EPS**2)
    return amp * torch.exp(-sersic_bn(n) * ((R / R_e) ** (1.0 / n) - 1.0))


def gaussian_psf_kernel(fwhm_arcsec: float, pixel_scale: float, size: int | None = None) -> Tensor:
    """Normalised 2-D Gaussian point-spread function, shape (1, 1, k, k)."""
    sigma_pix = fwhm_arcsec / (2 * math.sqrt(2 * math.log(2))) / pixel_scale
    if size is None:
        size = int(2 * math.ceil(3 * sigma_pix) + 1)
    r = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    g = torch.exp(-0.5 * (r / sigma_pix) ** 2)
    k = torch.outer(g, g)
    return (k / k.sum())[None, None]
