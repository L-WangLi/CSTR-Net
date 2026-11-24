"""C-MAPSS training and evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .factory import build_model

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TEST_COUNTS = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
LOG_FMT = "[%(asctime)s] %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
CSV_FIELDS = ("epoch", "train_loss", "train_rmse_raw", "val_loss", "val_rmse_all",
              "val_rmse_nonplateau", "ckpt_score", "learning_rate")

class RULDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y_raw: np.ndarray,
        pad_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y_raw).float()
        self.pad_mask = torch.from_numpy(pad_mask).bool() if pad_mask is not None else None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        if self.pad_mask is None:
            return self.x[idx], self.y[idx]
        return self.x[idx], self.y[idx], self.pad_mask[idx]


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cmapss")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FMT, DATE_FMT)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / "console.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def make_run_dir(base_dir: Path, dataset: str, window_size: int, seed: int) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{ts}_{dataset}_w{window_size}_seed{seed}"


def resolve_device(device_arg: str) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_npy_opt(path: Path) -> Optional[np.ndarray]:
    return np.load(path) if path.exists() else None


def load_bundle(
    data_dir: Path,
    dataset: str,
    include_ops: bool,
    window_size: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    base = data_dir / dataset
    if not base.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {base}. Pass --data_dir explicitly if using a custom folder."
        )

    X_train = np.load(base / "X_train.npy")
    y_train = np.load(base / "y_train.npy")
    X_val = np.load(base / "X_val.npy")
    y_val = np.load(base / "y_val.npy")
    X_test = np.load(base / "X_test.npy")
    y_test = np.load(base / "y_test.npy")
    pad_mask_test = _load_npy_opt(base / "pad_mask_test.npy")
    uid_test = _load_npy_opt(base / "uid_test.npy")
    cyc_test = _load_npy_opt(base / "cycle_end_test.npy")

    metadata: Dict[str, Any] = {}
    meta_path = base / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    if X_train.shape[1] != window_size:
        raise ValueError(
            f"Window mismatch: args window_size={window_size}, X_train.shape[1]={X_train.shape[1]} "
            f"from {base}"
        )
    expected_dim = 17 if include_ops else 14
    if X_train.shape[2] != expected_dim:
        raise ValueError(
            f"Input dim mismatch: include_ops={include_ops} expects D={expected_dim}, "
            f"got X_train.shape[2]={X_train.shape[2]} from {base}"
        )
    expected_test = EXPECTED_TEST_COUNTS.get(dataset)
    if expected_test is not None and len(X_test) != expected_test:
        raise ValueError(f"{dataset}: expected {expected_test} test windows, got {len(X_test)}.")

    if pad_mask_test is not None and pad_mask_test.shape != X_test.shape[:2]:
        raise ValueError(
            f"pad_mask_test shape {pad_mask_test.shape} does not match X_test {X_test.shape[:2]}"
        )

    logger.info(
        f"Data loaded from {base}: train={X_train.shape} val={X_val.shape} test={X_test.shape}"
    )
    return {
        "base": base,
        "metadata": metadata,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "pad_mask_test": pad_mask_test,
        "uid_test": uid_test,
        "cyc_test": cyc_test,
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pad_mask: Optional[np.ndarray] = None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        RULDataset(x, y, pad_mask=pad_mask),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def nasa_score_np(pred: np.ndarray, target: np.ndarray) -> float:
    d = pred - target
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def _rmse_np(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _mae_np(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - target)))


def compute_metrics(
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    cap: float = 125.0,
) -> Dict[str, float]:
    pred = np.clip(pred_raw.astype(float), 0.0, cap)
    target = target_raw.astype(float)
    y_norm = target / cap
    plateau = y_norm >= 0.99
    nonplateau = ~plateau

    return {
        "rmse_all": _rmse_np(pred, target),
        "mae_all": _mae_np(pred, target),
        "rmse_nonplateau": _rmse_np(pred[nonplateau], target[nonplateau]),
        "score_all": nasa_score_np(pred, target),
        "n_samples": float(len(target)),
    }


def checkpoint_score(metrics: Dict[str, float], dataset: str, mode: str) -> float:
    if mode == "all":
        return float(metrics["rmse_all"])
    if mode == "nonplateau":
        rn = float(metrics["rmse_nonplateau"])
        return rn if math.isfinite(rn) else float(metrics["rmse_all"])
    if mode == "weighted":
        rn = float(metrics["rmse_nonplateau"])
        ra = float(metrics["rmse_all"])
        if not math.isfinite(rn):
            return ra
        return 0.6 * rn + 0.4 * ra
    if mode == "auto":
        return checkpoint_score(metrics, dataset, "weighted" if dataset in {"FD002", "FD004"} else "nonplateau")
    raise ValueError(f"Unknown checkpoint metric: {mode}")


def save_predictions_csv(
    path: Path,
    pred: np.ndarray,
    target: np.ndarray,
    uid: Optional[np.ndarray],
    cyc: Optional[np.ndarray],
) -> None:
    fields = ["unit_id", "cycle", "target_rul", "predicted_rul", "absolute_error", "squared_error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (p, t) in enumerate(zip(pred, target)):
            writer.writerow({
                "unit_id": int(uid[i]) if uid is not None and i < len(uid) else i,
                "cycle": int(cyc[i]) if cyc is not None and i < len(cyc) else -1,
                "target_rul": float(t),
                "predicted_rul": float(p),
                "absolute_error": float(abs(p - t)),
                "squared_error": float((p - t) ** 2),
            })


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def init_train_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()


def append_train_csv(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow({k: row.get(k, "") for k in CSV_FIELDS})


@torch.no_grad()
def evaluate(model, loader, device, cap):
    model.eval()
    predictions, targets = [], []
    for batch in loader:
        bx, by = batch[:2]
        mask = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None
        pred = model(bx.to(device, non_blocking=True), pad_mask=mask)
        predictions.append(pred.cpu().numpy())
        targets.append(by.cpu().numpy())
    pred = np.concatenate(predictions) if predictions else np.empty(0)
    target = np.concatenate(targets) if targets else np.empty(0)
    return compute_metrics(pred, target, cap), pred, target


def train_one_epoch(model, loader, optimizer, device, args):
    model.train()
    total_loss = total_sq = total_n = 0
    for batch in loader:
        bx, by = batch[:2]
        mask = batch[2].to(device, non_blocking=True) if len(batch) == 3 else None
        bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(bx, pad_mask=mask)
        loss = (prediction - by).square().mean()
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        if not math.isfinite(grad):
            raise RuntimeError(f"Non-finite gradient norm: {grad}")
        optimizer.step()
        n = int(by.numel())
        total_loss += float(loss.item()) * n
        total_sq += float((prediction.detach() - by).square().sum().item())
        total_n += n
    return total_loss / max(total_n, 1), math.sqrt(total_sq / max(total_n, 1))


def run(args):
    dataset = f"FD00{args.fd}"
    run_dir = make_run_dir(Path(args.out_dir), dataset, args.window_size, args.seed)
    logger = setup_logging(run_dir)
    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    bundle = load_bundle(Path(args.data_dir), dataset, args.include_ops, args.window_size, logger)
    train_loader = make_loader(bundle["X_train"], bundle["y_train"], args.batch_size, True, args.seed)
    val_loader = make_loader(bundle["X_val"], bundle["y_val"], args.eval_batch_size, False, args.seed)
    test_loader = make_loader(bundle["X_test"], bundle["y_test"], args.eval_batch_size, False, args.seed,
                              pad_mask=bundle["pad_mask_test"])
    model = build_model(args, bundle["X_train"].shape[2]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    config = {**vars(args), "dataset": dataset, "trainable_params": model.count_parameters()}
    write_json(run_dir / "config.json", config)
    train_csv = run_dir / "train_log.csv"
    init_train_csv(train_csv)
    best_path = run_dir / "best_checkpoint.pt"
    best_score, best_epoch, best_metrics, waiting = float("inf"), -1, {}, 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_rmse = train_one_epoch(model, train_loader, optimizer, device, args)
        val_metrics, _, _ = evaluate(model, val_loader, device, args.max_rul)
        # Keep the separate validation-loss pass and its DataLoader RNG sequence.
        losses = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                losses.append(float((model(bx) - by).square().mean().item()))
        val_loss = float(np.mean(losses)) if losses else float("nan")
        score = checkpoint_score(val_metrics, dataset, args.ckpt_metric)
        if score < best_score - args.min_delta:
            best_score, best_epoch, best_metrics, waiting = score, epoch, val_metrics, 0
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "epoch": epoch, "best_score": best_score, "val_metrics": val_metrics}, best_path)
        else:
            waiting += 1
        scheduler.step()
        append_train_csv(train_csv, {
            "epoch": epoch, "train_loss": train_loss, "train_rmse_raw": train_rmse,
            "val_loss": val_loss, "val_rmse_all": val_metrics["rmse_all"],
            "val_rmse_nonplateau": val_metrics["rmse_nonplateau"],
            "ckpt_score": score, "learning_rate": scheduler.get_last_lr()[0],
        })
        logger.info(f"epoch={epoch} train_rmse={train_rmse:.4f} val_rmse={val_metrics['rmse_all']:.4f}")
        if args.patience > 0 and waiting >= args.patience:
            break
    if not best_path.exists():
        raise RuntimeError("No valid checkpoint was selected; check validation data and training settings.")
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    test_metrics, prediction, target = evaluate(model, test_loader, device, args.max_rul)
    save_predictions_csv(run_dir / "predictions_test.csv", np.clip(prediction, 0, args.max_rul), target,
                         bundle["uid_test"], bundle["cyc_test"])
    metrics = {"dataset": dataset, "model": "CSTR-Net", "seed": args.seed,
               "best_epoch": best_epoch, "best_ckpt_score": best_score,
               "best_val_metrics": best_metrics, "test_metrics": test_metrics,
               "params": model.count_parameters(), "run_dir": str(run_dir)}
    write_json(run_dir / "metrics.json", metrics)
    logger.info(f"test_rmse={test_metrics['rmse_all']:.4f} test_score={test_metrics['score_all']:.2f}")
    return metrics
