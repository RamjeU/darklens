from __future__ import annotations

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def get_device(prefer: str | None = None) -> torch.device:
    """Use CUDA, then Apple-silicon MPS, then CPU, unless ``prefer`` says otherwise."""
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)


DEFAULT_CONFIG = {
    "sim": {},
    "model": {"embed_dim": 256, "flow_layers_dm": 6, "flow_layers_macro": 8,
              "flow_hidden": 128, "n_bins": 8, "encoder_widths": [32, 64, 128, 256],
              "tail_bound": 10.0},
    "train": {"steps": 30000, "batch_size": 64, "lr": 5e-4, "weight_decay": 1e-4,
              "warmup": 1000, "grad_clip": 5.0, "eval_every": 500, "n_val": 1024, "seed": 0},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike | None) -> dict:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    with open(path) as f:
        return _merge(DEFAULT_CONFIG, yaml.safe_load(f) or {})


def load_trained(ckpt_path: str | os.PathLike, device: str | torch.device | None = None):
    """Rebuild (model, simulator, config) from a training checkpoint."""
    from .model import DarkLensNPE
    from .simulator import LensSimulator, SimConfig

    if not isinstance(device, torch.device):
        device = get_device(device)
    if not Path(ckpt_path).exists():
        raise SystemExit(f"[darklens] no checkpoint at {ckpt_path}. Train first:\n"
                         f"  python -m darklens train --config configs/default.yaml --run {Path(ckpt_path).parent}")
    ckpt = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    cfg["model"].setdefault("tail_bound", 5.0)  # checkpoints from before v0.2 used 5
    sim = LensSimulator(SimConfig.from_dict(cfg["sim"]), device)
    model = DarkLensNPE(sim.dm_prior, sim.macro_prior, sim.cfg.npix, **cfg["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), sim, cfg
