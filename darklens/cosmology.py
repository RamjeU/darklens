"""Flat Lambda-CDM distances and the lensing critical surface density.

Everything here is computed once in float64 NumPy (it does not need to be
differentiable) and handed to the PyTorch simulator as scalars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C_KM_S = 299_792.458                # speed of light [km/s]
G_MPC_MSUN = 4.30091727e-9          # Newton's G [Mpc (km/s)^2 / M_sun]
ARCSEC = np.pi / (180.0 * 3600.0)   # one arcsecond in radians

_trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz


@dataclass(frozen=True)
class FlatLCDM:
    """Flat Lambda-CDM cosmology (defaults: Planck 2018, rounded)."""

    H0: float = 67.4
    Om0: float = 0.315

    def comoving_distance(self, z: float, n: int = 4096) -> float:
        """Line-of-sight comoving distance to redshift z [Mpc]."""
        if z <= 0:
            return 0.0
        zz = np.linspace(0.0, z, n)
        Ez = np.sqrt(self.Om0 * (1 + zz) ** 3 + (1 - self.Om0))
        return (C_KM_S / self.H0) * _trapz(1.0 / Ez, zz)

    def angular_diameter_distance(self, z: float) -> float:
        return self.comoving_distance(z) / (1 + z)

    def angular_diameter_distance_z1z2(self, z1: float, z2: float) -> float:
        """D_A between z1 < z2 (valid because the universe is flat)."""
        return (self.comoving_distance(z2) - self.comoving_distance(z1)) / (1 + z2)


@dataclass(frozen=True)
class LensGeometry:
    """Distances and unit conversions for a single lens/source redshift pair."""

    z_lens: float = 0.5
    z_source: float = 2.0
    cosmo: FlatLCDM = FlatLCDM()

    @property
    def D_l(self) -> float:
        return self.cosmo.angular_diameter_distance(self.z_lens)

    @property
    def D_s(self) -> float:
        return self.cosmo.angular_diameter_distance(self.z_source)

    @property
    def D_ls(self) -> float:
        return self.cosmo.angular_diameter_distance_z1z2(self.z_lens, self.z_source)

    @property
    def sigma_crit(self) -> float:
        """Critical surface density Sigma_cr = c^2 D_s / (4 pi G D_l D_ls) [M_sun / Mpc^2]."""
        return C_KM_S**2 * self.D_s / (4 * np.pi * G_MPC_MSUN * self.D_l * self.D_ls)

    @property
    def sigma_crit_arcsec2(self) -> float:
        """Sigma_cr expressed per square arcsecond on the lens plane [M_sun / arcsec^2]."""
        return self.sigma_crit * (self.D_l * ARCSEC) ** 2

    @property
    def kpc_per_arcsec(self) -> float:
        return self.D_l * ARCSEC * 1e3

    def einstein_mass(self, theta_E: float) -> float:
        """Projected mass inside the Einstein radius of an isothermal lens [M_sun]."""
        return np.pi * theta_E**2 * self.sigma_crit_arcsec2
