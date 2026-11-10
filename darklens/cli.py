"""Command-line interface:  python -m darklens <command> [options]

Commands
--------
gallery      Draw simulator figures (no training needed).
train        Train the neural posterior estimator (on-the-fly simulation).
evaluate     Calibration tests (coverage, SBC, TARP), parameter recovery, example posteriors, saliency.
population   Hierarchical inference of warm dark matter from a mock survey of many lenses.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def cmd_gallery(a):
    from . import plotting as P
    from .simulator import LensSimulator, SimConfig
    from .utils import get_device, load_config

    cfg = load_config(a.config)
    sim = LensSimulator(SimConfig.from_dict(cfg["sim"]), get_device(a.device))
    out = Path(a.out)
    print(P.plot_gallery(sim, out / "gallery.png"))
    print(P.plot_substructure_effect(sim, out / "cdm_vs_wdm.png"))
    print(P.plot_mass_function(out / "mass_function.png", sim.cfg.m_min, sim.cfg.m_max, sim.cfg.mf_slope))


def cmd_train(a):
    from .train import train
    best = train(a.config, a.run, device=a.device, resume=a.resume, max_minutes=a.max_minutes)
    print(f"[darklens] best checkpoint: {best}")


def cmd_evaluate(a):
    from . import plotting as P
    from .calibration import collect_posteriors, expected_coverage, posterior_summary, sbc_ranks, tarp
    from .utils import load_trained

    run = Path(a.run)
    out = Path(a.out or run / "figures")
    model, sim, cfg = load_trained(run / "best.pt", a.device)
    print(f"[darklens] collecting posteriors for {a.n_tests} test lenses ...")
    res = collect_posteriors(model, sim, a.n_tests, a.n_samples, seed=a.seed)

    heads = {"dm": sim.dm_prior, "macro": sim.macro_prior}
    metrics = {}
    tarps = {}
    for head, prior in heads.items():
        lo, hi = prior.low.numpy(), prior.high.numpy()
        s, t = res[f"{head}_samples"], res[f"{head}_true"]
        summ = posterior_summary(s, t, lo, hi)
        tarps[head] = tarp(s, t, lo, hi)
        metrics[head] = {name: {k: float(v[i]) for k, v in summ.items()} for i, name in enumerate(prior.names)}
        metrics[head]["_tarp_mean_abs_deviation"] = tarps[head]["atc"]
        ranges = list(zip(lo, hi))
        P.plot_recovery(s, t, prior.labels, ranges, out / f"recovery_{head}.png")
        P.plot_sbc(sbc_ranks(s, t), prior.labels, s.shape[1], out / f"sbc_{head}.png")
        # example posteriors: pick lenses with many / few subhalos for the DM head
        picks = [int(np.argmax(res["n_sub"])), int(np.argmin(res["n_sub"]))] if head == "dm" else [0, 1]
        for k, j in enumerate(picks):
            P.corner(s[j], t[j], prior.labels, ranges, out / f"posterior_{head}_example{k + 1}.png",
                     title=f"Posterior for test lens {j}  (N_sub = {int(res['n_sub'][j])})")
    P.plot_coverage(expected_coverage(res["dm_hpd_level"]), expected_coverage(res["macro_hpd_level"]),
                    tarps["dm"], tarps["macro"], out / "calibration.png")
    if (run / "train_log.csv").exists():
        P.plot_training_curve(run / "train_log.csv", out / "training_curve.png")

    # Saliency: which pixels drive the dark-matter posterior?
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(a.seed + 1)
    sim_out = sim.simulate(1, g, overrides={"f_sub": 0.04, "log10_mhm": 6.5})
    x = sim.normalise(sim_out["image"]).clone().requires_grad_(True)
    lp = model.log_prob_dm(sim_out["theta_dm"].to(dev), model.embed(x)).sum()
    (grad,) = torch.autograd.grad(lp, x)
    P.plot_saliency(sim_out["image"][0, 0].detach().cpu(), grad.abs()[0, 0].cpu(), out / "saliency.png")

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[darklens] figures written to {out}")


def cmd_population(a):
    from . import plotting as P
    from .population import (credible_upper_limit, hierarchical_posterior, lens_log_density_grid,
                             mhm_limit_to_wdm_mass, simulate_population)
    from .utils import load_trained

    run = Path(a.run)
    out = Path(a.out or run / "figures")
    model, sim, _ = load_trained(run / "best.pt", a.device)
    summary = {}
    for name, mhm in (("cdm", a.cdm_mhm), ("wdm", a.wdm_mhm)):
        print(f"[darklens] mock survey '{name}': {a.n_lenses} lenses, log10 M_hm = {mhm}")
        pop = simulate_population(sim, a.n_lenses, a.mu_f, a.sigma_f, mhm, seed=a.seed + (0 if name == "cdm" else 1))
        grid = lens_log_density_grid(model, sim, pop["images"])
        # Information check: mean KL(q || prior) per lens on the grid. If the head has not
        # learned anything yet, multiplying N near-prior posteriors only amplifies small
        # density errors at the prior edges, and the result is meaningless.
        lr = grid["logq"] - grid["log_interim"]
        cell = (grid["f_grid"][1] - grid["f_grid"][0]) * (grid["m_grid"][1] - grid["m_grid"][0])
        kl = float((np.exp(grid["logq"]) * lr * cell).sum((1, 2)).mean())
        print(f"[darklens] mean per-lens information KL(q||prior) = {kl:.3f} nats")
        if kl < 0.05:
            print("[darklens] WARNING: the dark-matter head is close to the prior (undertrained). "
                  "The hierarchical result below mostly reflects estimator error; train longer.")
        Ns = [n for n in (1, 5, 10, 25, 50, 100, 200) if n <= a.n_lenses]
        if Ns[-1] != a.n_lenses:
            Ns.append(a.n_lenses)
        results = [(n, hierarchical_posterior(grid, a.sigma_f, np.arange(n))) for n in Ns]
        final = results[-1][1]
        ul = credible_upper_limit(final["log10_mhm"], final["marg_mhm"], 0.95)
        lim = mhm_limit_to_wdm_mass(ul)
        summary[name] = {"true_log10_mhm": mhm, "n_lenses": a.n_lenses, "mean_kl_per_lens": kl,
                         "log10_mhm_95_upper": ul, "m_wdm_95_lower_keV": lim,
                         "log10_mhm_map": float(final["log10_mhm"][np.argmax(final["marg_mhm"])])}
        P.plot_population(results, a.mu_f, mhm, out / f"population_{name}.png", wdm_limit=lim)
    (out / "population.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="darklens", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gallery")
    g.add_argument("--config", default=None)
    g.add_argument("--out", default="figures")
    g.set_defaults(fn=cmd_gallery)

    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/default.yaml")
    t.add_argument("--run", default="runs/default")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--max-minutes", type=float, default=None, help="stop (and save) after this wall time")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--run", default="runs/default")
    e.add_argument("--out", default=None)
    e.add_argument("--n-tests", type=int, default=500)
    e.add_argument("--n-samples", type=int, default=1000)
    e.add_argument("--seed", type=int, default=123)
    e.set_defaults(fn=cmd_evaluate)

    q = sub.add_parser("population")
    q.add_argument("--run", default="runs/default")
    q.add_argument("--out", default=None)
    q.add_argument("--n-lenses", type=int, default=100)
    q.add_argument("--mu-f", type=float, default=0.025, help="true population-mean f_sub")
    q.add_argument("--sigma-f", type=float, default=0.005, help="lens-to-lens scatter in f_sub")
    q.add_argument("--cdm-mhm", type=float, default=6.5, help="true log10 M_hm for the cold-DM-like survey")
    q.add_argument("--wdm-mhm", type=float, default=9.0, help="true log10 M_hm for the warm-DM survey")
    q.add_argument("--seed", type=int, default=2024)
    q.set_defaults(fn=cmd_population)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
