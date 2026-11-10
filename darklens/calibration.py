"""Posterior calibration diagnostics for amortised posteriors.

A posterior estimator can be sharp but wrong. These tests check whether its credible
regions contain the truth as often as they claim to, using simulations whose ground
truth is known.

* **Expected coverage of highest-density regions (HPD).** For each test lens we
  compute the credibility level of the true parameters under the flow's own
  density. If the posterior is calibrated, those levels are uniform on [0, 1]
  (Hermans et al. 2022, "A Trust Crisis in Simulation-Based Inference?").
* **Simulation-based calibration (SBC) ranks** of each 1-D marginal
  (Talts et al. 2018). Ranks are uniform if the posterior is calibrated.
* **TARP**, Tests of Accuracy with Random Points (Lemos et al. 2023, ICML). This is a
  necessary *and sufficient* coverage test for the full joint posterior, and it uses
  samples only.
"""
from __future__ import annotations

import numpy as np
import torch

from .model import DarkLensNPE
from .simulator import LensSimulator


@torch.no_grad()
def collect_posteriors(model: DarkLensNPE, sim: LensSimulator, n_tests: int = 500, n_samples: int = 1000,
                       seed: int = 123, bs: int = 50) -> dict:
    """Simulate test lenses and draw posterior samples and densities for both heads."""
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ("dm_true", "macro_true", "dm_samples", "macro_samples",
                           "dm_hpd_level", "macro_hpd_level", "n_sub")}
    for i in range(0, n_tests, bs):
        b = min(bs, n_tests - i)
        sim_out = sim.simulate(b, g)
        h = model.embed(sim.normalise(sim_out["image"]))
        for head, true_key in (("dm", "theta_dm"), ("macro", "theta_macro")):
            true = sim_out[true_key].to(dev)
            sample_fn = model.sample_dm if head == "dm" else model.sample_macro
            logp_fn = model.log_prob_dm if head == "dm" else model.log_prob_macro
            s = sample_fn(h, n_samples)                                   # (b, n, d)
            lp_true = logp_fn(true, h)                                    # (b,)
            lp_s = logp_fn(s.reshape(b * n_samples, -1), h.repeat_interleave(n_samples, 0)).view(b, n_samples)
            # HPD credibility of the truth = fraction of posterior mass with higher density.
            level = (lp_s > lp_true[:, None]).float().mean(1)
            out[f"{head}_true"].append(true.cpu())
            out[f"{head}_samples"].append(s.cpu())
            out[f"{head}_hpd_level"].append(level.cpu())
        out["n_sub"].append(sim_out["subhalos"].n)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def expected_coverage(levels: np.ndarray, n_grid: int = 101) -> tuple[np.ndarray, np.ndarray]:
    """Empirical coverage of the alpha-HPD region as a function of nominal alpha."""
    alpha = np.linspace(0, 1, n_grid)
    cov = (levels[None, :] <= alpha[:, None]).mean(1)
    return alpha, cov


def sbc_ranks(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Rank of the truth among posterior samples, per dimension: (N, d) in [0, n]."""
    return (samples < truth[:, None, :]).sum(1)


def tarp(samples: np.ndarray, truth: np.ndarray, low: np.ndarray, high: np.ndarray,
         n_bootstrap: int = 100, seed: int = 0) -> dict:
    """TARP expected-coverage curve with bootstrap uncertainty (Lemos et al. 2023).

    samples: (N, n, d); truth: (N, d). Parameters are first rescaled to the unit prior box.
    """
    rng = np.random.default_rng(seed)
    span = high - low
    s = (samples - low) / span
    t = (truth - low) / span
    N = len(t)
    ref = rng.uniform(0, 1, size=t.shape)                         # random reference points
    d_samp = np.linalg.norm(s - ref[:, None, :], axis=-1)        # (N, n)
    d_true = np.linalg.norm(t - ref, axis=-1)                    # (N,)
    f = (d_samp < d_true[:, None]).mean(1)                       # (N,)
    alpha = np.linspace(0, 1, 101)

    def ecp(fv):
        return (fv[None, :] < alpha[:, None]).mean(1)

    curve = ecp(f)
    boots = np.stack([ecp(f[rng.integers(0, N, N)]) for _ in range(n_bootstrap)])
    return {"alpha": alpha, "ecp": curve, "ecp_lo": np.percentile(boots, 2.5, 0),
            "ecp_hi": np.percentile(boots, 97.5, 0),
            "atc": float(np.abs(curve - alpha).mean())}  # mean |deviation| from the diagonal


def posterior_summary(samples: np.ndarray, truth: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Per-parameter accuracy metrics: RMSE of the posterior mean, mean 68% interval
    width, and how much the posterior shrinks relative to the prior (1 = fully pinned)."""
    mean = samples.mean(1)
    std = samples.std(1)
    lo, hi = np.percentile(samples, [16, 84], axis=1)
    prior_std = (high - low) / np.sqrt(12)
    return {
        "rmse": np.sqrt(((mean - truth) ** 2).mean(0)),
        "width68": (hi - lo).mean(0),
        "contraction": 1 - (std**2).mean(0) / prior_std**2,
        "coverage68": ((truth >= lo) & (truth <= hi)).mean(0),
    }
