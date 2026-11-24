"""N-CMAPSS training and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.cstr_net import _DEFAULT_BRANCH_SPECS, _LONG_BRANCH_SPECS
from .cmapss import set_seed
from .factory import build_ncmapss_model as build_model

ROOT = Path(__file__).resolve().parents[1]

N_OPS     = 4
N_SENSORS = 14
INPUT_DIM = N_OPS + N_SENSORS   # 18

def resolve_data_dir(ds: str, processed_dir: str = None) -> Path:
    if processed_dir is not None:
        return Path(processed_dir) / ds
    protocol = "downsampled" if ds == "DS02" else "phase_tokens"
    return ROOT / "dataset" / "processed" / "ncmapss" / protocol / ds

def load_bundle(data_dir: Path) -> Dict:
    meta = json.loads((data_dir / "metadata.json").read_text())
    bundle = {
        "X_train":  np.load(data_dir / "X_train.npy"),
        "y_train":  np.load(data_dir / "y_train.npy"),
        "X_val":    np.load(data_dir / "X_val.npy"),
        "y_val":    np.load(data_dir / "y_val.npy"),
        "X_test":   np.load(data_dir / "X_test.npy"),
        "y_test":   np.load(data_dir / "y_test.npy"),
        "uid_test": np.load(data_dir / "uid_test.npy"),
        "metadata": meta,
    }
    return bundle

def n_fc_train(meta: Dict) -> int:
    if "n_fc_train" in meta:
        return int(meta["n_fc_train"])
    return 1   

def make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).float()
    ds = TensorDataset(X_t, y_t)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False,
        generator=g if shuffle else None,
        drop_last=False,
    )

def resolve_branch_specs(window: int) -> Tuple[str, Tuple[Tuple[int, int, int], ...]]:
    if int(window) >= 50:
        return "long_T_ge_50", _LONG_BRANCH_SPECS
    return "default_T_lt_50", _DEFAULT_BRANCH_SPECS

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    model.train()
    total_sq = 0.0
    total_loss = 0.0
    total_n = 0
    for bx, by in loader:
        bx = bx.to(device, non_blocking=True)
        by = by.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(bx)
        loss = nn.functional.mse_loss(pred, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        n = int(by.numel())
        total_loss += float(loss.item()) * n
        total_sq   += float((pred.detach() - by).square().sum().item())
        total_n    += n
    rmse = math.sqrt(total_sq / max(total_n, 1))
    return total_loss / max(total_n, 1), rmse

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for bx, by in loader:
        bx = bx.to(device, non_blocking=True)
        pred = model(bx)
        preds.append(pred.cpu().numpy())
        targets.append(by.numpy())
    pred_np   = np.concatenate(preds)   if preds   else np.empty(0)
    target_np = np.concatenate(targets) if targets else np.empty(0)
    rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2))) if len(pred_np) else float("nan")
    return rmse, pred_np, target_np

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def make_run_dir(out_dir: Path, seed: int) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / f"seed{seed}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def run(args: argparse.Namespace) -> Dict:
    set_seed(args.seed, deterministic=False)
    out_dir = Path(args.out_dir) / args.ds
    run_dir = make_run_dir(out_dir, args.seed)
    device  = resolve_device(args.device)

    data_dir = resolve_data_dir(args.ds, getattr(args, "processed_dir", None))
    print(f"[NCMAPSS] ds={args.ds}  seed={args.seed}  device={device}")
    print(f"          run_dir={run_dir}")

    if not data_dir.exists():
        print(f"[ERROR] Preprocessed data not found: {data_dir}", file=sys.stderr)
        print("        See README.md for preprocessing commands.", file=sys.stderr)
        sys.exit(1)

    bundle = load_bundle(data_dir)
    meta   = bundle["metadata"]

    film_policy = getattr(args, "film_policy", "auto")
    if film_policy == "none":
        film_mode = "none"
    elif film_policy == "auto":
        film_mode = "delta_gamma" if n_fc_train(meta) >= 2 else "none"
    else:
        raise ValueError(f"Unknown FiLM policy: {film_policy}")
    print(f"  FiLM policy={film_policy}, resolved mode={film_mode}")

    window = int(meta.get("window", bundle["X_train"].shape[1]))
    branch_rule, branch_specs = resolve_branch_specs(window)
    branch_rf = [
        1 + (k - 1) * d1 + (k - 1) * d2
        for k, d1, d2 in branch_specs
    ]
    print(
        f"  branch rule: {branch_rule}  window={window}  "
        f"branches={len(branch_specs)}  RF={branch_rf}"
    )

    print(f"  data: train={bundle['X_train'].shape} val={bundle['X_val'].shape} "
          f"test={bundle['X_test'].shape}")

    train_loader = make_loader(bundle["X_train"], bundle["y_train"],
                               args.batch_size, shuffle=True,  seed=args.seed)
    val_loader   = make_loader(bundle["X_val"],   bundle["y_val"],
                               args.batch_size, shuffle=False, seed=args.seed)
    test_loader  = make_loader(bundle["X_test"],  bundle["y_test"],
                               args.batch_size, shuffle=False, seed=args.seed)

    model = build_model(args, film_mode=film_mode, branch_specs=branch_specs).to(device)
    n_params = count_parameters(model)
    print(f"  model: {n_params:,} params")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5,
    )

    best_val_rmse = float("inf")
    best_epoch    = -1
    best_path     = run_dir / "best_checkpoint.pt"
    patience_counter = 0
    train_log: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, optimizer, device, args.grad_clip
        )
        val_rmse, _, _ = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_rmse": round(train_rmse, 4),
            "val_rmse":   round(val_rmse,   4),
            "lr":         round(scheduler.get_last_lr()[0], 8),
            "time_s":     round(elapsed, 2),
        }
        train_log.append(row)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  ep{epoch:4d}  train_rmse={train_rmse:.4f}  "
                  f"val_rmse={val_rmse:.4f}  ({elapsed:.1f}s)")

        if val_rmse < best_val_rmse:
            best_val_rmse   = val_rmse
            best_epoch      = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_rmse, pred_test, target_test = evaluate(model, test_loader, device)
    test_mae  = float(np.mean(np.abs(pred_test - target_test)))

    print(f"\n  === RESULTS ds={args.ds} seed={args.seed} ===")
    print(f"  best_val_rmse={best_val_rmse:.4f}  (epoch {best_epoch})")
    print(f"  test_rmse={test_rmse:.4f}  test_mae={test_mae:.4f}")
    results = {
        "dataset":       args.ds,
        "model":         "CSTR-Net",
        "film_mode":     film_mode,
        "branch_rule":   branch_rule,
        "branch_specs":  [list(spec) for spec in branch_specs],
        "branch_rf":     branch_rf,
        "seed":          args.seed,
        "best_epoch":    best_epoch,
        "best_val_rmse": round(best_val_rmse, 6),
        "test_rmse":     round(test_rmse,     6),
        "test_mae":      round(test_mae,      6),
        "n_params":      n_params,
        "data": {
            "train_shape": list(bundle["X_train"].shape),
            "val_shape":   list(bundle["X_val"].shape),
            "test_shape":  list(bundle["X_test"].shape),
            "window":      window,
        },
        "args": vars(args),
    }
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    (run_dir / "train_log.json").write_text(json.dumps(train_log, indent=2))

    np.save(run_dir / "pred_test.npy",   pred_test.astype(np.float32))
    np.save(run_dir / "target_test.npy", target_test.astype(np.float32))
    np.save(run_dir / "uid_test.npy",    bundle["uid_test"].astype(np.int64))

    config = vars(args).copy()
    config.update({"run_dir": str(run_dir), "n_params": n_params,
                   "input_dim": INPUT_DIM, "n_ops": N_OPS, "n_sensors": N_SENSORS,
                   "film_mode": film_mode, "window": window, "branch_rule": branch_rule,
                   "branch_specs": [list(spec) for spec in branch_specs],
                   "branch_rf": branch_rf})
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  saved → {run_dir}/metrics.json")
    return results

