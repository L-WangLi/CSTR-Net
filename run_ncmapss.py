#!/usr/bin/env python3
"""Train CSTR-Net on N-CMAPSS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training import ncmapss

ROOT = Path(__file__).resolve().parent
DATASETS = ["DS01", "DS02", "DS03", "DS04", "DS05", "DS06", "DS07", "DS08a", "DS08c"]

def training_args(config, dataset, seed, processed_dir=None, out_dir=None, device="auto",
                  epochs=None, batch_size=None):
    cfg = json.loads(Path(config).read_text())
    protocol = cfg["ds02_protocol"] if dataset == "DS02" else cfg["phase_token_protocol"]
    data = Path(processed_dir).expanduser().resolve() if processed_dir else ROOT / protocol["processed_dir"]
    values = {**cfg["training"], **cfg["model"], "ds": dataset, "seed": seed,
              "processed_dir": str(data), "device": device,
              "out_dir": str(Path(out_dir).expanduser().resolve() if out_dir else ROOT / "runs/ncmapss"),
              "expected_window": protocol["window"], "config_path": str(Path(config).resolve())}
    values.pop("seeds", None)
    if epochs is not None:
        values["epochs"] = epochs
    if batch_size is not None:
        values["batch_size"] = batch_size
    return argparse.Namespace(**values)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ncmapss.json")
    parser.add_argument("--processed-dir", "--data-dir", dest="processed_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    cfg = json.loads(options.config.read_text())
    seeds = cfg["training"]["seeds"] if options.seeds is None else options.seeds
    for ds in options.datasets:
        for seed in seeds:
            args = training_args(options.config, ds, seed, options.processed_dir,
                                 options.out_dir, options.device, options.epochs, options.batch_size)
            meta_path = Path(args.processed_dir) / ds / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if int(meta["window"]) != args.expected_window:
                    raise ValueError(f"{ds}: expected window {args.expected_window}, got {meta['window']}")
            if options.dry_run:
                print(json.dumps(vars(args), indent=2))
            else:
                if not meta_path.is_file():
                    raise FileNotFoundError(f"Preprocess {ds} first; missing {meta_path}. See README.md.")
                ncmapss.run(args)

if __name__ == "__main__":
    main()
