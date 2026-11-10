"""Physics checks: every profile must satisfy the lensing identities exactly."""
import math

import numpy as np
import pytest
import torch

from darklens.cosmology import LensGeometry
from darklens.profiles import ellipticity_to_q_phi, pseudo_jaffe_deflection, sie_deflection, sersic
from darklens.substructure import (SubhaloPopulation, half_mode_mass, powerlaw_icdf,
                                   powerlaw_mean_mass, wdm_mass_from_half_mode)



@pytest.fixture(autouse=True)
def _float64():
    """Run the physics checks in double precision, then restore the default."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


T = lambda v: torch.tensor(float(v))  # noqa: E731


def _div_curl(fn, x, y, h=1e-5):
    axp, ayp = fn(x + h, y); axm, aym = fn(x - h, y)
    bxp, byp = fn(x, y + h); bxm, bym = fn(x, y - h)
    div = (axp - axm + byp - bym) / (2 * h)
    curl = (ayp - aym - bxp + bxm) / (2 * h)
    return div, curl


@pytest.mark.parametrize("e1,e2", [(0.0, 0.0), (0.2, -0.1), (-0.15, 0.22)])
def test_sie_poisson_equation(e1, e2):
    """div(alpha) = 2 kappa and curl(alpha) = 0 (alpha is the gradient of a potential)."""
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(50, generator=g) * 4 - 2, torch.rand(50, generator=g) * 4 - 2
    fn = lambda x, y: sie_deflection(x, y, T(1.1), T(e1), T(e2), T(0.03), T(-0.02))  # noqa: E731
    div, curl = _div_curl(fn, x, y)
    q, phi = ellipticity_to_q_phi(T(e1), T(e2))
    c, s = torch.cos(phi), torch.sin(phi)
    dx, dy = x - 0.03, y + 0.02
    xr, yr = c * dx + s * dy, -s * dx + c * dy
    kappa = 1.1 / (2 * torch.sqrt(q * xr**2 + yr**2 / q))
    assert torch.allclose(div / 2, kappa, rtol=1e-5)
    assert curl.abs().max() < 1e-6


def test_sie_einstein_radius_circular():
    """For a circular SIE (= SIS) the deflection magnitude equals theta_E everywhere."""
    x = torch.linspace(-2, 2, 17); y = torch.full_like(x, 0.7)
    ax, ay = sie_deflection(x, y, T(1.3), T(0.0), T(0.0), T(0.0), T(0.0))
    assert torch.allclose(torch.sqrt(ax**2 + ay**2), torch.full_like(x, 1.3), rtol=2e-3)


def test_pseudo_jaffe_convergence_and_mass():
    b, a = T(0.05), T(0.3)
    x = torch.tensor([0.05, 0.2, 0.8, 2.0]); y = torch.zeros(4)
    fn = lambda x, y: pseudo_jaffe_deflection(x, y, b, a, T(0), T(0))  # noqa: E731
    div, curl = _div_curl(fn, x, y)
    kappa = b / 2 * (1 / x - 1 / torch.sqrt(x**2 + a**2))
    assert torch.allclose(div / 2, kappa, rtol=1e-4)
    # total mass: alpha(r) * r -> b * a at large r  (M = pi b a Sigma_cr)
    ax, _ = fn(torch.tensor([1e4]), torch.tensor([0.0]))
    assert abs(ax.item() * 1e4 - (b * a).item()) / (b * a).item() < 1e-3


def test_subhalo_mass_inversion():
    geo = LensGeometry()
    pop = SubhaloPopulation(geo)
    m = torch.tensor([1e7, 1e8, 1e9, 1e10]); thE = T(1.2)
    b, a = pop.mass_to_pseudo_jaffe(m, thE)
    m_back = math.pi * b * a * geo.sigma_crit_arcsec2
    assert torch.allclose(m_back, m, rtol=1e-6)


def test_powerlaw_sampler_mean():
    u = torch.rand(400_000, generator=torch.Generator().manual_seed(1))
    m = powerlaw_icdf(u, 1e7, 1e10, -1.9)
    assert m.min() >= 1e7 and m.max() <= 1e10
    assert abs(m.mean().item() / powerlaw_mean_mass(1e7, 1e10, -1.9) - 1) < 0.03


def test_expected_subhalo_count_matches_poisson_mean():
    pop = SubhaloPopulation(LensGeometry())
    n = 4000
    thE, f = torch.full((n,), 1.1), torch.full((n,), 0.02)
    subs = pop.sample(thE, f, torch.full((n,), 5.0), torch.zeros(n), torch.zeros(n), torch.Generator().manual_seed(2))
    lam = pop.expected_number(thE[:1], f[:1]).item()
    # log M_hm = 5 is below m_min, so WDM thinning removes almost nothing
    assert abs(subs.n.mean().item() / lam - 1) < 0.03


def test_wdm_mass_round_trip_and_literature_value():
    for m in (1.0, 3.0, 5.2, 9.0):
        assert abs(wdm_mass_from_half_mode(half_mode_mass(m)) - m) < 1e-8
    # Gilman et al. (2020) quote m_hm < 10^7.8 M_sun  <->  m_WDM > 5.2 keV
    assert abs(np.log10(half_mode_mass(5.2)) - 7.8) < 0.2


def test_critical_density_reasonable():
    geo = LensGeometry(0.5, 2.0)
    # Einstein mass of a 1" lens at these redshifts is ~1e11 M_sun
    assert 5e10 < geo.einstein_mass(1.0) < 5e11
    assert 6.0 < geo.kpc_per_arcsec < 6.6


def test_sersic_half_light():
    """R_e encloses half the flux of a circular Sersic profile."""
    n = T(1.7)
    r = torch.linspace(1e-4, 40, 400_000)
    I = sersic(r, torch.zeros_like(r), T(1.0), T(1.0), n, T(0), T(0), T(0), T(0))
    cum = torch.cumsum(I * r, 0)
    half_r = r[torch.searchsorted(cum, cum[-1] / 2)]
    assert abs(half_r.item() - 1.0) < 0.01
