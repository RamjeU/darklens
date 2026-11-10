"""Figures for the README and paper-style outputs (matplotlib only, no extra deps)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .substructure import half_mode_mass, wdm_suppression  # noqa: E402

# Categorical slots in fixed order (validated colour-blind-safe palette).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#1f1f1d", "#6b6a64", "#e1e0d9"
IMG_CMAP = "magma"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _img(ax, im, title=None, cmap=IMG_CMAP, **kw):
    ax.imshow(im, origin="lower", cmap=cmap, **kw)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


# ----------------------------------------------------------------------------
# Simulator figures
# ----------------------------------------------------------------------------
@torch.no_grad()
def plot_gallery(sim, path, n=18, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = sim.simulate(n, g)
    p = out["params"]
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(1.75 * cols, 1.9 * np.ceil(n / cols)))
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        _img(ax, out["image"][i, 0].cpu())
        ax.set_title(f"$f_{{sub}}$={p['f_sub'][i]:.3f}\n$\\log M_{{hm}}$={p['log10_mhm'][i]:.1f}  "
                     f"N={int(out['subhalos'].n[i])}", fontsize=7)
    fig.suptitle("Simulated strong lenses drawn from the training prior", y=1.0)
    return _save(fig, path)


@torch.no_grad()
def plot_substructure_effect(sim, path, seed=3):
    """CDM vs WDM: the same lens and source, with different dark-matter physics."""
    g = torch.Generator().manual_seed(seed)
    p = sim.sample_parameters(1, g, overrides={"f_sub": 0.04, "theta_E": 1.15, "x_src": 0.05, "y_src": 0.03})
    rows = []
    for label, mhm in (("Cold dark matter\n$\\log M_{hm}=6$", 6.0),
                       ("Warm dark matter\n$\\log M_{hm}=9.5$", 9.5)):
        pp = dict(p)
        pp["log10_mhm"] = torch.tensor([mhm])
        subs = sim.population.sample(pp["theta_E"], pp["f_sub"], pp["log10_mhm"], pp["x_lens"],
                                     pp["y_lens"], torch.Generator().manual_seed(11))
        rows.append((label, subs, pp))
    smooth = sim.render(p, None, noise=False)[0, 0].cpu()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    lim = None
    for r, (label, subs, pp) in enumerate(rows):
        kap = sim.convergence(pp, subs)[0, 0].cpu()
        clean = sim.render(pp, subs, noise=False)[0, 0].cpu()
        noisy = sim.render(pp, subs, noise=True, generator=torch.Generator().manual_seed(5))[0, 0].cpu()
        _img(axes[r, 0], np.log10(kap.clamp(min=1e-2)), cmap="cividis", vmin=-1.0, vmax=1.0)
        axes[r, 0].set_ylabel(label)
        axes[r, 0].set_title(f"convergence $\\log_{{10}}\\kappa$  (N={int(subs.n)})" if r == 0 else f"N={int(subs.n)}")
        _img(axes[r, 1], clean, "noise-free image" if r == 0 else None)
        diff = (clean - smooth).numpy()
        if lim is None:  # share the colour scale between rows
            lim = float(np.abs(diff).max())
        _img(axes[r, 2], diff, "image minus smooth-lens image" if r == 0 else None, cmap="RdBu_r", vmin=-lim, vmax=lim)
        _img(axes[r, 3], noisy, "observed (PSF + noise)" if r == 0 else None)
    for r in range(2):
        axes[r, 0].set_ylabel(rows[r][0], fontsize=9)
    fig.suptitle("The same lens under two dark-matter models: WDM removes the small subhalos", y=1.01)
    return _save(fig, path)


def plot_mass_function(path, m_min=1e7, m_max=1e10, slope=-1.9):
    m = np.logspace(np.log10(m_min), np.log10(m_max), 200)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(m, m ** (slope + 1) / m_min ** (slope + 1), color=INK, lw=2, label="CDM")
    for i, mw in enumerate([2.0, 3.5, 6.0]):
        mhm = float(half_mode_mass(mw))
        sup = wdm_suppression(torch.tensor(m), torch.tensor(np.log10(mhm))).numpy()
        ax.plot(m, sup * m ** (slope + 1) / m_min ** (slope + 1), color=C[i], lw=2,
                label=f"WDM {mw:g} keV  ($\\log M_{{hm}}$={np.log10(mhm):.1f})")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("subhalo mass $m$ [$M_\\odot$]")
    ax.set_ylabel("$m\\,dN/dm$  (normalised)")
    ax.set_title("Subhalo mass function: warm dark matter suppresses low-mass halos")
    ax.legend(fontsize=7.5)
    ax.grid(True, color=GRID, lw=0.6, which="major")
    return _save(fig, path)


# ----------------------------------------------------------------------------
# Training / inference figures
# ----------------------------------------------------------------------------
def plot_training_curve(log_csv, path):
    rows = list(csv.DictReader(open(log_csv)))
    s = np.array([float(r["step"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(s, [float(r["train_loss"]) for r in rows], color=C[0], lw=2, label="train")
    axes[0].plot(s, [float(r["val_loss"]) for r in rows], color=C[1], lw=2, label="validation")
    axes[0].set_title("Total negative log posterior")
    axes[0].legend()
    axes[1].plot(s, [float(r["val_nll_dm"]) for r in rows], color=C[0], lw=2, label="dark-matter head")
    axes[1].plot(s, [float(r["val_nll_macro"]) for r in rows], color=C[2], lw=2, label="macro-model head")
    axes[1].set_title("Validation NLL by head (logit space)")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(True, color=GRID, lw=0.6)
    return _save(fig, path)


def corner(samples, truth, labels, ranges, path, title=None, color=C[0], bins=40):
    d = samples.shape[1]
    fig, axes = plt.subplots(d, d, figsize=(1.55 * d + 0.6, 1.55 * d + 0.6))
    axes = np.atleast_2d(axes)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                ax.hist(samples[:, i], bins=bins, range=ranges[i], color=color, alpha=0.85, histtype="stepfilled")
                ax.axvline(truth[i], color=C[1], lw=1.5)
                ax.set_yticks([])
                ax.set_xlim(ranges[i])
            else:
                H, xe, ye = np.histogram2d(samples[:, j], samples[:, i], bins=bins, range=[ranges[j], ranges[i]])
                H = H.T
                ax.imshow(H, origin="lower", extent=[*ranges[j], *ranges[i]], aspect="auto", cmap="Blues")
                hs = np.sort(H.ravel())[::-1]
                cs = np.cumsum(hs) / hs.sum()
                levels = sorted({hs[np.searchsorted(cs, lv)] for lv in (0.95, 0.68)})
                if len(levels) == 2 and levels[0] < levels[1]:
                    ax.contour(H, levels=levels, extent=[*ranges[j], *ranges[i]], colors=[C[0]], linewidths=0.9)
                ax.plot(truth[j], truth[i], marker="*", color=C[1], ms=9, mec="white", mew=0.8)
                ax.axvline(truth[j], color=C[1], lw=0.8, alpha=0.6)
                ax.axhline(truth[i], color=C[1], lw=0.8, alpha=0.6)
            if i == d - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif i != j:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6.5)
    if title:
        fig.suptitle(title, y=1.0)
    return _save(fig, path)


def plot_coverage(cov_dm, cov_macro, tarp_dm, tarp_macro, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.plot(*cov_dm, color=C[0], lw=2, label="dark-matter head")
    ax.plot(*cov_macro, color=C[2], lw=2, label="macro-model head")
    ax.set_title("Expected coverage of HPD regions")
    ax.set_xlabel("credibility level")
    ax.set_ylabel("empirical coverage")
    ax.text(0.04, 0.9, "above: conservative\nbelow: overconfident", fontsize=7, color=MUTED, va="top")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    for t, c, lab in ((tarp_dm, C[0], "dark-matter head"), (tarp_macro, C[2], "macro-model head")):
        ax.fill_between(t["alpha"], t["ecp_lo"], t["ecp_hi"], color=c, alpha=0.2, lw=0)
        ax.plot(t["alpha"], t["ecp"], color=c, lw=2, label=f"{lab}  (mean |dev| {t['atc']:.3f})")
    ax.set_title("TARP coverage test (Lemos et al. 2023)")
    ax.set_xlabel("credibility level $1-\\alpha$")
    ax.set_ylabel("expected coverage probability")
    ax.legend(loc="lower right", fontsize=7.5)
    for a in axes:
        a.set_xlim(0, 1), a.set_ylim(0, 1), a.set_aspect("equal")
        a.grid(True, color=GRID, lw=0.6)
    return _save(fig, path)


def plot_sbc(ranks, labels, n_samples, path, n_bins=20):
    d = ranks.shape[1]
    cols = min(d, 5)
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.8 * rows), squeeze=False)
    N = ranks.shape[0]
    expected = N / n_bins
    band = 2 * np.sqrt(expected * (1 - 1 / n_bins))
    for k, ax in enumerate(axes.flat):
        if k >= d:
            ax.axis("off")
            continue
        ax.axhspan(expected - band, expected + band, color=GRID, lw=0)
        ax.hist(ranks[:, k], bins=n_bins, range=(0, n_samples), color=C[0], alpha=0.9, histtype="stepfilled")
        ax.set_title(labels[k])
        ax.set_yticks([]), ax.set_xticks([])
    fig.suptitle("SBC rank histograms: flat means calibrated (grey band = ±2σ)", y=1.02)
    return _save(fig, path)


def plot_population(results_by_n, truth_mu_f, truth_mhm, path, wdm_limit=None):
    """results_by_n: list of (N, hierarchical_posterior dict)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for k, (N, res) in enumerate(results_by_n):
        ax.plot(res["log10_mhm"], res["marg_mhm"], color=C[k % len(C)], lw=2, label=f"N = {N} lenses")
    ax.axvline(truth_mhm, color=INK, lw=1.2, ls="--", label="truth")
    ax.set_xlabel("$\\log_{10} M_{hm}$ [$M_\\odot$]")
    ax.set_ylabel("posterior density")
    ax.set_title("Shared half-mode mass sharpens as lenses are combined")
    ax.legend(fontsize=7.5)
    ax.grid(True, color=GRID, lw=0.6)
    N, res = results_by_n[-1]
    ax = axes[1]
    ext = [res["log10_mhm"][0], res["log10_mhm"][-1], res["mu_f"][0], res["mu_f"][-1]]
    ax.imshow(res["posterior"], origin="lower", extent=ext, aspect="auto", cmap="Blues")
    P = res["posterior"]
    hs = np.sort(P.ravel())[::-1]
    cs = np.cumsum(hs) / hs.sum()
    lv = sorted({hs[min(np.searchsorted(cs, q), len(hs) - 1)] for q in (0.95, 0.68)})
    if len(lv) == 2 and lv[0] < lv[1]:
        ax.contour(res["log10_mhm"], res["mu_f"], P, levels=lv, colors=[C[0]], linewidths=1.2)
    ax.plot(truth_mhm, truth_mu_f, marker="*", color=C[1], ms=12, mec="white")
    ax.set_xlabel("$\\log_{10} M_{hm}$")
    ax.set_ylabel("population mean $\\mu_{f_{sub}}$")
    title = f"Joint hyper-posterior, N = {N}"
    if wdm_limit is not None:
        title += f"\n95% limit: $m_{{WDM}}$ > {wdm_limit:.1f} keV"
    ax.set_title(title)
    return _save(fig, path)


def plot_saliency(image, saliency, path, title="Where the dark-matter posterior looks"):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3.1))
    _img(axes[0], image, "observed image")
    _img(axes[1], saliency, "|∂ log q(f_sub, M_hm | x) / ∂x|", cmap="viridis")
    fig.suptitle(title, y=1.02)
    return _save(fig, path)


def plot_recovery(samples, truth, labels, ranges, path, max_points=300):
    """Posterior mean (with 68% interval) against the true value, one panel per parameter."""
    d = truth.shape[1]
    cols = min(d, 5)
    rows = int(np.ceil(d / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.3 * rows), squeeze=False)
    idx = np.arange(min(len(truth), max_points))
    mean = samples[idx].mean(1)
    lo, hi = np.percentile(samples[idx], [16, 84], axis=1)
    for k, ax in enumerate(axes.flat):
        if k >= d:
            ax.axis("off")
            continue
        r = ranges[k]
        ax.plot(r, r, color=MUTED, lw=1, ls="--")
        ax.errorbar(truth[idx, k], mean[:, k], yerr=[mean[:, k] - lo[:, k], hi[:, k] - mean[:, k]],
                    fmt="o", ms=2.2, color=C[0], ecolor=C[0], elinewidth=0.5, alpha=0.55)
        ax.set_xlim(r), ax.set_ylim(r)
        ax.set_title(labels[k])
        ax.tick_params(labelsize=6.5)
        ax.grid(True, color=GRID, lw=0.5)
    fig.supxlabel("true value", fontsize=9)
    fig.supylabel("posterior mean ± 68%", fontsize=9)
    return _save(fig, path)
