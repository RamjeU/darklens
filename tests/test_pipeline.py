"""End-to-end smoke tests: simulate -> one training step -> sample -> population grid."""
import torch

from darklens.model import DarkLensNPE
from darklens.population import hierarchical_posterior, lens_log_density_grid
from darklens.simulator import LensSimulator, SimConfig


def _tiny():
    sim = LensSimulator(SimConfig(npix=32, pixel_scale=0.13, supersample=1))
    model = DarkLensNPE(sim.dm_prior, sim.macro_prior, npix=32, embed_dim=32, flow_hidden=32,
                        flow_layers_dm=2, flow_layers_macro=2, encoder_widths=(8, 16, 16, 16))
    return sim, model


def test_simulator_shapes_and_finiteness():
    sim, _ = _tiny()
    out = sim.simulate(4, torch.Generator().manual_seed(0))
    assert out["image"].shape == (4, 1, 32, 32)
    assert torch.isfinite(out["image"]).all()
    assert out["theta_dm"].shape == (4, 2) and out["theta_macro"].shape == (4, 7)


def test_simulator_is_differentiable():
    sim, _ = _tiny()
    p = sim.sample_parameters(2, torch.Generator().manual_seed(1))
    p["theta_E"] = p["theta_E"].clone().requires_grad_()
    img = sim.render(p, sim.sample_subhalos(p), noise=False)
    img.sum().backward()
    assert torch.isfinite(p["theta_E"].grad).all() and p["theta_E"].grad.abs().sum() > 0


def test_training_step_and_sampling():
    sim, model = _tiny()
    out = sim.simulate(8, torch.Generator().manual_seed(2))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    loss = model.loss(sim.normalise(out["image"]), out["theta_dm"], out["theta_macro"])["loss"]
    loss.backward(); opt.step()
    assert torch.isfinite(loss)
    model.eval()
    h = model.embed(sim.normalise(out["image"]))
    s = model.sample_dm(h, 16)
    lo, hi = sim.dm_prior.low, sim.dm_prior.high
    assert s.shape == (8, 16, 2) and (s >= lo).all() and (s <= hi).all()


def test_population_grid_is_normalised():
    sim, model = _tiny()
    model.eval()
    imgs = sim.simulate(3, torch.Generator().manual_seed(3))["image"]
    grid = lens_log_density_grid(model, sim, imgs, n_f=20, n_m=20)
    res = hierarchical_posterior(grid, sigma_f=0.005)
    df = res["mu_f"][1] - res["mu_f"][0]; dm = res["log10_mhm"][1] - res["log10_mhm"][0]
    assert abs(res["posterior"].sum() * df * dm - 1) < 1e-6
