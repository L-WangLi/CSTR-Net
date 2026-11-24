#!/usr/bin/env python3
"""Train CSTR-Net on C-MAPSS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training import cmapss
from training.factory import MODEL_DEFAULTS

ROOT = Path(__file__).resolve().parent

def training_args(config, seed=None, data_dir=None, out_dir=None,
                  device="auto", epochs=None, batch_size=None):
    cfg = json.loads(Path(config).read_text())
    args = argparse.Namespace(deterministic=False)
    for key, value in {**MODEL_DEFAULTS, **cfg}.items():
        setattr(args, key, value)
    args.seed = args.seed if seed is None else seed
    args.data_dir = str(Path(data_dir).expanduser().resolve() if data_dir else ROOT / cfg["data_dir"])
    args.out_dir = str(Path(out_dir).expanduser().resolve() if out_dir else ROOT / cfg["out_dir"])
    args.device = "" if device == "auto" else device
    args.epochs = args.epochs if epochs is None else epochs
    args.batch_size = args.batch_size if batch_size is None else batch_size
    args.config_path = str(Path(config).resolve())
    return args

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.seed is not None and options.seeds is not None:
        parser.error("Use either --seed or --seeds.")
    config = options.config or ROOT / "configs/cmapss" / f"{options.dataset}.json"
    seeds = options.seeds if options.seeds is not None else [options.seed]
    for seed in seeds:
        args = training_args(config, seed, options.data_dir,
                             options.out_dir, options.device, options.epochs, options.batch_size)
        if options.dry_run:
            print(json.dumps(vars(args), indent=2))
        else:
            cmapss.run(args)

if __name__ == "__main__":
    main()
