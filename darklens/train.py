"""Amortised neural posterior estimation with on-the-fly simulation.

Every gradient step uses a fresh batch from the simulator, so the network never sees
the same lens twice. The training set is effectively infinite and cannot be
overfit. A fixed validation set, simulated once with its own seed, tracks progress.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch

from .model import DarkLensNPE
from .simulator import LensSimulator, SimConfig
from .utils import get_device, load_config, seed_everything


def _lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return f


@torch.no_grad()
def _simulate_validation(sim: LensSimulator, n: int, seed: int, bs: int = 128):
    g = torch.Generator().manual_seed(10_000 + seed)
    xs, dms, mas = [], [], []
    for i in range(0, n, bs):
        out = sim.simulate(min(bs, n - i), g)
        xs.append(sim.normalise(out["image"]).cpu())
        dms.append(out["theta_dm"])
        mas.append(out["theta_macro"])
    return torch.cat(xs), torch.cat(dms), torch.cat(mas)


@torch.no_grad()
def evaluate_loss(model: DarkLensNPE, x, dm, ma, device, bs: int = 256) -> dict[str, float]:
    model.eval()
    tot = {"loss": 0.0, "nll_dm": 0.0, "nll_macro": 0.0}
    for i in range(0, len(x), bs):
        l = model.loss(x[i:i + bs].to(device), dm[i:i + bs].to(device), ma[i:i + bs].to(device))
        for k in tot:
            tot[k] += l[k].item() * len(x[i:i + bs])
    model.train()
    return {k: v / len(x) for k, v in tot.items()}


def train(config_path: str | None, out_dir: str, device: str | None = None, resume: bool = False,
          max_minutes: float | None = None) -> Path:
    cfg = load_config(config_path)
    tc = cfg["train"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = get_device(device)
    gen = seed_everything(tc["seed"])

    sim = LensSimulator(SimConfig.from_dict(cfg["sim"]), dev)
    model = DarkLensNPE(sim.dm_prior, sim.macro_prior, sim.cfg.npix, **cfg["model"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(tc["warmup"], tc["steps"]))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[darklens] device={dev}  parameters={n_params / 1e6:.2f}M  steps={tc['steps']}")
    base = model.prior_baseline()
    print(f"[darklens] prior-only baselines (an image-blind head scores this): "
          f"dm {base['dm']:.3f}, macro {base['macro']:.3f}. "
          f"Learning from images = val below these numbers.")
    dm_w = float(tc.get("dm_weight", 1.0))

    start, best = 0, float("inf")
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best = ck["step"], ck["best"]
        gen.manual_seed(tc["seed"] + start)
        print(f"[darklens] resumed from step {start}")

    print("[darklens] simulating validation set ...")
    val = _simulate_validation(sim, tc["n_val"], tc["seed"])

    log_path = out / "train_log.csv"
    new_log = not (resume and log_path.exists())
    logf = open(log_path, "a" if not new_log else "w", newline="")
    writer = csv.writer(logf)
    if new_log:
        writer.writerow(["step", "train_loss", "val_loss", "val_nll_dm", "val_nll_macro", "lr", "elapsed_s"])

    def save(name: str, step: int):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "best": best, "config": cfg}, out / name)

    model.train()
    t0, run_loss, n_run = time.time(), 0.0, 0
    for step in range(start, tc["steps"]):
        batch = sim.simulate(tc["batch_size"], gen)
        x = sim.normalise(batch["image"])
        losses = model.loss(x, batch["theta_dm"].to(dev), batch["theta_macro"].to(dev), dm_weight=dm_w)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        run_loss += losses["loss"].item()
        n_run += 1

        last = step + 1 == tc["steps"]
        timed_out = max_minutes is not None and (time.time() - t0) > 60 * max_minutes
        if (step + 1) % tc["eval_every"] == 0 or last or timed_out:
            v = evaluate_loss(model, *val, dev)
            elapsed = time.time() - t0
            rate = (step + 1 - start) / elapsed
            eta = (tc["steps"] - step - 1) / rate / 60
            writer.writerow([step + 1, run_loss / n_run, v["loss"], v["nll_dm"], v["nll_macro"],
                             sched.get_last_lr()[0], round(elapsed, 1)])
            logf.flush()
            flag = ""
            if v["loss"] < best:
                best = v["loss"]
                save("best.pt", step + 1)
                flag = "  *best*"
            save("last.pt", step + 1)
            print(f"step {step + 1:6d} | train {run_loss / n_run:7.3f} | val {v['loss']:7.3f} "
                  f"(dm {v['nll_dm']:6.3f} [{v['nll_dm'] - base['dm']:+.3f} vs prior], "
                  f"macro {v['nll_macro']:7.3f}) | {rate:5.2f} it/s | "
                  f"ETA {eta:6.1f} min{flag}", flush=True)
            run_loss, n_run = 0.0, 0
            if timed_out:
                print(f"[darklens] stopping early: --max-minutes={max_minutes} reached")
                break
    logf.close()
    return out / "best.pt"
