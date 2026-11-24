#!/usr/bin/env python3
"""Evaluate a checkpoint on a test split."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from models.checkpoint import load_model

ROOT = Path(__file__).resolve().parent

def prediction_metrics(prediction, target, dataset):
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape or not len(target):
        raise ValueError("Predictions and labels must have the same nonempty shape.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Predictions and labels must be finite.")
    evaluated = np.clip(prediction, 0, 125) if dataset.startswith("FD") else prediction
    error = evaluated - target
    metrics = {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}
    if dataset.startswith("FD"):
        metrics["nasa_score"] = float(np.expm1(np.where(error < 0, -error / 13, error / 10)).sum())
    return evaluated, metrics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="Subset folder containing X_test.npy and y_test.npy.")
    parser.add_argument("--dataset", help="Optional override, for example FD001 or DS02.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke check only; evaluate the first N windows and mark results as partial.")
    options = parser.parse_args()
    if options.batch_size <= 0 or options.threads <= 0 or (options.limit is not None and options.limit <= 0):
        parser.error("Batch size, threads, and limit must be positive.")
    torch.set_num_threads(options.threads)
    if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if options.device == "auto" else torch.device(options.device)
    model, config = load_model(options.checkpoint)
    dataset = options.dataset or config.get("dataset") or config.get("ds") or config.get("fd")
    if isinstance(dataset, int):
        dataset = f"FD{dataset:03d}"
    if not dataset or not str(dataset).startswith(("FD", "DS")):
        parser.error("Cannot determine the dataset; supply --dataset.")
    x = np.load(options.data_dir / "X_test.npy", mmap_mode="r")
    target = np.load(options.data_dir / "y_test.npy", mmap_mode="r")
    if x.ndim != 3 or x.shape[2] != model.input_dim or len(x) != len(target):
        raise ValueError("Test inputs/labels do not match the checkpoint dimensions.")
    expected_window = config.get("window_size", config.get("window"))
    if expected_window is not None and x.shape[1] != int(expected_window):
        raise ValueError(f"Expected window {expected_window}, got {x.shape[1]}.")
    mask_file = options.data_dir / "pad_mask_test.npy"
    mask = np.load(mask_file, mmap_mode="r") if mask_file.exists() else None
    if mask is not None and mask.shape != x.shape[:2]:
        raise ValueError("Padding mask shape must match the input batch and time axes.")
    n = min(len(x), options.limit) if options.limit else len(x)
    model.to(device).eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, n, options.batch_size):
            stop = min(start + options.batch_size, n)
            bx = torch.from_numpy(np.array(x[start:stop], copy=True)).float().to(device)
            pm = torch.from_numpy(np.array(mask[start:stop], copy=True)).bool().to(device) if mask is not None else None
            predictions.append(model(bx, pad_mask=pm).cpu().numpy())
    raw = np.concatenate(predictions)
    prediction, metrics = prediction_metrics(raw, target[:n], str(dataset))
    metrics.update(dataset=dataset, n_evaluated=n, total_test_windows=len(x),
                   partial_evaluation=n < len(x), checkpoint=str(options.checkpoint.resolve()),
                   seed=config.get("seed"), aggregation="one_run", torch_version=str(torch.__version__))
    output = options.out_dir or ROOT / "outputs" / options.checkpoint.parent.name
    output.mkdir(parents=True, exist_ok=True)
    uid_file = options.data_dir / "uid_test.npy"
    uid = np.load(uid_file, mmap_mode="r") if uid_file.exists() else np.arange(n)
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample", "unit", "target", "prediction_raw", "prediction"])
        for i in range(n):
            writer.writerow([i, int(uid[i]), float(target[i]), float(raw[i]), float(prediction[i])])
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
