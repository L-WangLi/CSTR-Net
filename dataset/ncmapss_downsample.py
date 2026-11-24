"""N-CMAPSS downsampled preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "raw" / "ncmapss"
OUT_BASE  = Path(__file__).resolve().parent / "processed" / "ncmapss" / "downsampled"

DATASETS = {"DS02": "N-CMAPSS_DS02-006.h5"}

DOWNSAMPLE = 100   # 1 Hz → 0.01 Hz
N_OPS      = 4
N_SENSORS  = 14
INPUT_DIM  = N_OPS + N_SENSORS   # 18

def load_raw(h5_path: Path) -> Dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "A_dev":    f["A_dev"][:],
            "W_dev":    f["W_dev"][:],
            "X_s_dev":  f["X_s_dev"][:],
            "Y_dev":    f["Y_dev"][:, 0],
            "A_test":   f["A_test"][:],
            "W_test":   f["W_test"][:],
            "X_s_test": f["X_s_test"][:],
            "Y_test":   f["Y_test"][:, 0],
        }

def downsample(A: np.ndarray, W: np.ndarray, Xs: np.ndarray, Y: np.ndarray,
               step: int = DOWNSAMPLE
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(0, len(A), step)
    return A[idx], W[idx], Xs[idx], Y[idx]

def build_unit_data(A: np.ndarray, W: np.ndarray, Xs: np.ndarray, Y: np.ndarray
                    ) -> Dict[int, Dict]:
    result = {}
    for uid in np.unique(A[:, 0]).astype(int):
        m = A[:, 0] == uid
        fc = int(np.unique(A[m, 2])[0])
        result[uid] = {
            "fc":    fc,
            "W":     W[m].astype(np.float32),
            "Xs":    Xs[m].astype(np.float32),
            "Y":     Y[m].astype(np.float32),
            "n":     int(m.sum()),
        }
    return result

def fit_scaler(unit_data: Dict, train_ids: List[int]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xs_all = np.concatenate([unit_data[u]["Xs"] for u in train_ids], axis=0)
    W_all  = np.concatenate([unit_data[u]["W"]  for u in train_ids], axis=0)
    return (Xs_all.mean(0), Xs_all.std(0) + 1e-8,
            W_all.mean(0),  W_all.std(0)  + 1e-8)


def normalize(unit_data: Dict, Xs_mean, Xs_std, W_mean, W_std) -> None:
    for d in unit_data.values():
        d["Xs_norm"] = (d["Xs"] - Xs_mean) / Xs_std
        d["W_norm"]  = (d["W"]  - W_mean)  / W_std


def make_windows(unit_data: Dict, unit_ids: List[int], window: int
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, uid_list = [], [], []
    for uid in unit_ids:
        d = unit_data[uid]
        X_unit = np.concatenate([d["W_norm"], d["Xs_norm"]], axis=1)  # [N, 18]
        Y      = d["Y"]
        n      = len(X_unit)
        for start in range(0, n - window + 1):
            X_list.append(X_unit[start:start + window])
            y_list.append(Y[start + window - 1])
            uid_list.append(uid)
    if not X_list:
        empty = (np.empty((0, window, INPUT_DIM), np.float32),
                 np.empty(0, np.float32),
                 np.empty(0, np.int64))
        return empty
    return (np.stack(X_list).astype(np.float32),
            np.array(y_list, np.float32),
            np.array(uid_list, np.int64))

def process_dataset(ds_name: str, window: int, verbose: bool = True,
                    data_dir: Path = DATA_DIR,
                    out_base: Path = OUT_BASE) -> None:
    h5_file = data_dir / DATASETS[ds_name]
    out_dir  = out_base / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n[{ds_name}] Loading {h5_file.name} ...")
    raw = load_raw(h5_file)

    if verbose:
        print(f"[{ds_name}] Downsampling to 0.01 Hz (every {DOWNSAMPLE}th row) ...")
    A_dev,  W_dev,  Xs_dev,  Y_dev  = downsample(raw["A_dev"],  raw["W_dev"],
                                                   raw["X_s_dev"], raw["Y_dev"])
    A_test, W_test, Xs_test, Y_test = downsample(raw["A_test"], raw["W_test"],
                                                   raw["X_s_test"], raw["Y_test"])

    dev_data  = build_unit_data(A_dev,  W_dev,  Xs_dev,  Y_dev)
    test_data = build_unit_data(A_test, W_test, Xs_test, Y_test)

    dev_ids  = [int(u) for u in sorted(dev_data.keys())]
    test_ids = [int(u) for u in sorted(test_data.keys())]

    val_ids   = [dev_ids[-1]]
    train_ids = [u for u in dev_ids if u not in val_ids]

    if verbose:
        fc_str = lambda uid, data: f"FC{data[uid]['fc']}"
        print(f"[{ds_name}] Split:")
        print(f"  train ({len(train_ids)}): {[f'{u}({fc_str(u, dev_data)})' for u in train_ids]}")
        print(f"  val   ({len(val_ids)}):   {[f'{u}({fc_str(u, dev_data)})' for u in val_ids]}")
        print(f"  test  ({len(test_ids)}):  {[f'{u}({fc_str(u, test_data)})' for u in test_ids]}")

    all_data = {**dev_data, **test_data}
    Xs_mean, Xs_std, W_mean, W_std = fit_scaler(dev_data, train_ids)
    normalize(all_data, Xs_mean, Xs_std, W_mean, W_std)

    if verbose:
        print(f"[{ds_name}] Building windows (w={window}) ...")
    X_train, y_train, uid_train = make_windows(all_data, train_ids, window)
    X_val,   y_val,   uid_val   = make_windows(all_data, val_ids,   window)
    X_test,  y_test,  uid_test  = make_windows(all_data, test_ids,  window)

    if verbose:
        print(f"  train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
        print(f"  y_train: [{y_train.min():.1f}, {y_train.max():.1f}]  "
              f"y_test: [{y_test.min():.1f}, {y_test.max():.1f}]")

    np.save(out_dir / "X_train.npy",  X_train)
    np.save(out_dir / "y_train.npy",  y_train)
    np.save(out_dir / "X_val.npy",    X_val)
    np.save(out_dir / "y_val.npy",    y_val)
    np.save(out_dir / "X_test.npy",   X_test)
    np.save(out_dir / "y_test.npy",   y_test)
    np.save(out_dir / "uid_test.npy", uid_test)

    fc_test  = {str(u): test_data[u]["fc"]  for u in test_ids}
    fc_train = {str(u): dev_data[u]["fc"]   for u in train_ids}
    fc_val   = {str(u): dev_data[u]["fc"]   for u in val_ids}
    n_fc_train = len({v for v in fc_train.values()})   # FC diversity in train

    meta = {
        "dataset":     ds_name,
        "h5_file":     h5_file.name,
        "window":      window,
        "downsample":  DOWNSAMPLE,
        "n_sensors":   N_SENSORS,
        "n_ops":       N_OPS,
        "input_dim":   INPUT_DIM,
        "train_units": train_ids,
        "val_units":   val_ids,
        "test_units":  test_ids,
        "fc_test":      fc_test,
        "fc_train":     fc_train,
        "fc_val":       fc_val,
        "n_fc_train":   n_fc_train,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val":   list(X_val.shape),
            "X_test":  list(X_test.shape),
        },
        "rul_range": {
            "train": [float(y_train.min()), float(y_train.max())],
            "val":   [float(y_val.min()),   float(y_val.max())],
            "test":  [float(y_test.min()),  float(y_test.max())],
        },
        "normalization": {
            "method":    "z-score-global",
            "fit_units": train_ids,
            "Xs_mean": Xs_mean.tolist(),
            "Xs_std":  Xs_std.tolist(),
            "W_mean":  W_mean.tolist(),
            "W_std":   W_std.tolist(),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"[{ds_name}] Saved → {out_dir}/")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess N-CMAPSS DS02 with 0.01 Hz downsampling."
    )
    parser.add_argument("--ds", nargs="+", default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="Datasets to process (default: all).")
    parser.add_argument("--window", type=int, default=50,
                        help="Sliding window length (default: 50).")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="Directory containing h5 files.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for the processed DS02 subset.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_base = Path(args.out_dir) if args.out_dir else OUT_BASE

    print(f"Processing {len(args.ds)} dataset(s): {args.ds}")
    print(f"Downsample: {DOWNSAMPLE}x (0.01 Hz)  Window: {args.window}  "
          f"out: {out_base}")

    for ds in args.ds:
        try:
            process_dataset(ds, args.window, data_dir=data_dir,
                            out_base=out_base)
        except Exception as e:
            print(f"[{ds}] ERROR: {e}", file=sys.stderr)
            raise

    print("\nAll done.")

if __name__ == "__main__":
    main()
