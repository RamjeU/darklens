"""Normalising-flow correctness: invertibility, exact log-determinants, normalisation."""
import torch

from darklens.flows import ConditionalSplineFlow, rq_spline
from darklens.priors import BoxUniform


def test_spline_inverse_and_logdet():
    torch.manual_seed(0)
    x = torch.randn(2000) * 2.5
    w, h, d = torch.randn(2000, 8), torch.randn(2000, 8), torch.randn(2000, 7)
    y, ld = rq_spline(x, w, h, d)
    xr, ild = rq_spline(y, w, h, d, inverse=True)
    assert (x - xr).abs().max() < 1e-3
    assert (ld + ild).abs().max() < 1e-3
    xx = x.clone().requires_grad_()
    yy, _ = rq_spline(xx, w, h, d)
    (g,) = torch.autograd.grad(yy.sum(), xx)
    assert (g.log() - ld).abs().max() < 1e-4


def test_spline_identity_tails():
    x = torch.tensor([-7.0, 6.0, 9.0])
    y, ld = rq_spline(x, torch.randn(3, 8), torch.randn(3, 8), torch.randn(3, 7), tail_bound=5.0)
    assert torch.equal(x, y) and torch.all(ld == 0)


@torch.no_grad()
def test_flow_density_integrates_to_one():
    torch.manual_seed(1)
    flow = ConditionalSplineFlow(2, 8, n_layers=4)
    for p in flow.parameters():
        p.add_(0.1 * torch.randn_like(p))
    ctx = torch.randn(1, 8)
    g = torch.linspace(-9, 9, 241)
    X, Y = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([X.ravel(), Y.ravel()], -1)
    lp = flow.log_prob(pts, ctx.expand(len(pts), -1))
    assert abs((lp.exp().sum() * (g[1] - g[0]) ** 2).item() - 1) < 5e-3


@torch.no_grad()
def test_flow_sample_inverts_density_path():
    torch.manual_seed(2)
    flow = ConditionalSplineFlow(7, 8, n_layers=6)
    for p in flow.parameters():
        p.add_(0.1 * torch.randn_like(p))
    ctx = torch.randn(3, 8)
    u = torch.randn(3, 7)
    z = u.clone()
    for layer in reversed(flow.layers):
        z, _ = layer.inverse(z, ctx)
    back = z.clone()
    for layer in flow.layers:
        back, _ = layer(back, ctx)
    assert (back - u).abs().max() < 1e-4


def test_prior_bijection_and_jacobian():
    prior = BoxUniform.from_dict({"a": (0.0, 0.05), "b": (6.0, 10.0)})
    th = prior.sample(100, torch.Generator().manual_seed(0))
    z = prior.to_unbounded(th)
    assert torch.allclose(prior.from_unbounded(z), th, atol=1e-5)
    thg = th.clone().requires_grad_()
    zz = prior.to_unbounded(thg)
    jac = torch.stack([torch.autograd.grad(zz[:, i].sum(), thg, retain_graph=True)[0][:, i] for i in range(2)], -1)
    assert torch.allclose(jac.abs().log().sum(-1), prior.log_abs_det_dz_dtheta(th), atol=1e-3)
