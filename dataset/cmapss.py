"""Final C-MAPSS preprocessing with engine-level validation splits."""
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
CONST_DROP = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
EXPECTED_TEST_COUNTS = {'FD001': 100, 'FD002': 259, 'FD003': 100, 'FD004': 248}


def resolve_data_dir(data_dir: Path, fd: int = 1) -> Path:
    directory = Path(data_dir).expanduser().resolve()
    required = [f'train_FD00{fd}.txt', f'test_FD00{fd}.txt', f'RUL_FD00{fd}.txt']
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {directory}: {missing}")
    return directory


def _read_txt(path: Path) -> pd.DataFrame:
    idx_cols = ['unit', 'cycle']
    op_cols  = [f'op{i}' for i in range(1, 4)]
    sensor_cols = [f's{i}' for i in range(1, 22)]
    cols = idx_cols + op_cols + sensor_cols
    return pd.read_csv(Path(path), sep=r'\s+', header=None, names=cols)


def load_subset(fd: int, data_dir: Path):
    data_dir = resolve_data_dir(data_dir, fd=fd)
    train = _read_txt(data_dir / f'train_FD00{fd}.txt')
    test  = _read_txt(data_dir / f'test_FD00{fd}.txt')
    rul_path = data_dir / f'RUL_FD00{fd}.txt'
    test_rul = np.loadtxt(rul_path) if rul_path.exists() else None
    return train, test, test_rul


def add_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle']).clip(upper=cap)
    return df


def add_test_rul(test_df: pd.DataFrame, test_rul: np.ndarray,
                 cap: int = 125) -> pd.DataFrame:
    df = test_df.copy()
    df['RUL'] = 0.0

    unit_order = list(df['unit'].drop_duplicates())
    if len(unit_order) != len(test_rul):
        raise ValueError(
            f"Engine count mismatch: data {len(unit_order)}, RUL file {len(test_rul)}")

    uid2idx = {u: i for i, u in enumerate(unit_order)}
    for uid, grp in df.groupby('unit'):
        final_rul = min(test_rul[uid2idx[uid]], cap)
        max_cyc   = grp['cycle'].max()
        for idx in grp.index:
            cur_cyc = df.loc[idx, 'cycle']
            df.loc[idx, 'RUL'] = min(final_rul + (max_cyc - cur_cyc), cap)
    return df


def global_minmax_scaling(train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          feat_cols: List[str]
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[feat_cols].values.astype(float))

    train_s = train_df.copy()
    test_s  = test_df.copy()
    train_s[feat_cols] = scaler.transform(train_df[feat_cols].values.astype(float))
    test_s[feat_cols]  = scaler.transform(test_df[feat_cols].values.astype(float))

    info = {
        'method':     'global_minmax',
        'feature_range': [0, 1],
        'data_min':   scaler.data_min_.tolist(),
        'data_max':   scaler.data_max_.tolist(),
        'scale':      scaler.scale_.tolist(),
        'data_range': scaler.data_range_.tolist(),
    }
    return train_s, test_s, info


def condition_wise_minmax_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    op_cols: List[str],
    sensor_cols: List[str],
    n_clusters: int = 6,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if len(op_cols) != 3:
        raise ValueError("condition_wise_minmax_scaling requires op1/op2/op3.")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be positive, got {n_clusters}")

    train_s = train_df.copy()
    test_s = test_df.copy()
    train_s[op_cols + sensor_cols] = train_s[op_cols + sensor_cols].astype(float)
    test_s[op_cols + sensor_cols] = test_s[op_cols + sensor_cols].astype(float)

    op_scaler = MinMaxScaler(feature_range=(0, 1))
    op_scaler.fit(train_df[op_cols].values.astype(float))
    train_ops_scaled = op_scaler.transform(train_df[op_cols].values.astype(float))
    test_ops_scaled = op_scaler.transform(test_df[op_cols].values.astype(float))

    n_unique_ops = np.unique(train_ops_scaled, axis=0).shape[0]
    k = min(int(n_clusters), int(n_unique_ops), len(train_df))
    if k < 1:
        raise ValueError("No training rows available for condition-wise normalization.")

    kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
    train_cluster = kmeans.fit_predict(train_ops_scaled)
    test_cluster = kmeans.predict(test_ops_scaled)

    train_s[op_cols] = train_ops_scaled
    test_s[op_cols] = test_ops_scaled

    cluster_info = []
    for cid in range(k):
        train_mask = train_cluster == cid
        test_mask = test_cluster == cid
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_df.loc[train_mask, sensor_cols].values.astype(float))
        train_s.loc[train_mask, sensor_cols] = scaler.transform(
            train_df.loc[train_mask, sensor_cols].values.astype(float)
        )
        if np.any(test_mask):
            test_s.loc[test_mask, sensor_cols] = scaler.transform(
                test_df.loc[test_mask, sensor_cols].values.astype(float)
            )
        cluster_info.append({
            'cluster': int(cid),
            'train_rows': int(train_mask.sum()),
            'test_rows': int(test_mask.sum()),
            'center': kmeans.cluster_centers_[cid].tolist(),
            'sensor_data_min': scaler.data_min_.tolist(),
            'sensor_data_max': scaler.data_max_.tolist(),
        })

    train_s['condition_cluster'] = train_cluster.astype(int)
    test_s['condition_cluster'] = test_cluster.astype(int)

    info = {
        'method': 'condition_wise_minmax',
        'feature_range': [0, 1],
        'op_cols': op_cols,
        'sensor_cols': sensor_cols,
        'n_requested_clusters': int(n_clusters),
        'n_used_clusters': int(k),
        'cluster_fit_data': 'train_op1_op2_op3_only',
        'test_assignment': 'nearest_training_op_cluster',
        'op_scaler': {
            'data_min': op_scaler.data_min_.tolist(),
            'data_max': op_scaler.data_max_.tolist(),
            'scale': op_scaler.scale_.tolist(),
            'data_range': op_scaler.data_range_.tolist(),
        },
        'clusters': cluster_info,
    }
    return train_s, test_s, info


def _split_train_units(
    train_df: pd.DataFrame,
    valid_split: float,
    random_seed: int,
) -> Tuple[List[int], List[int]]:
    units = np.array(sorted(int(u) for u in train_df['unit'].unique()), dtype=int)
    if valid_split <= 0:
        return units.tolist(), []
    rng = np.random.RandomState(random_seed)
    perm_units = rng.permutation(units)
    n_val_units = int(round(len(perm_units) * valid_split))
    n_val_units = min(max(n_val_units, 1), len(perm_units) - 1)
    val_units = sorted(int(u) for u in perm_units[:n_val_units])
    train_units = sorted(int(u) for u in perm_units[n_val_units:])
    return train_units, val_units


def _make_train_windows(
    df: pd.DataFrame,
    feat_cols: List[str],
    window_size: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_all, y_all = [], []
    uid_all, cyc_all = [], []
    for uid, grp in df.groupby('unit'):
        Xw, yw, end_idx = make_windows(
            grp[feat_cols].values.astype(float),
            grp['RUL'].values.astype(float),
            window_size, stride
        )
        if len(Xw) > 0:
            X_all.append(Xw)
            y_all.append(yw)
            uid_all.append(np.full(len(Xw), int(uid), dtype=np.int32))
            cyc_all.append(grp['cycle'].values.astype(np.int32)[end_idx])

    if not X_all:
        return (
            np.empty((0, window_size, len(feat_cols))),
            np.empty(0),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.concatenate(X_all),
        np.concatenate(y_all),
        np.concatenate(uid_all),
        np.concatenate(cyc_all),
    )


def make_windows(features: np.ndarray, rul: np.ndarray, window_size: int, stride: int = 1):
    if window_size <= 0 or stride <= 0:
        raise ValueError("Window size and stride must be positive.")
    if len(features) < window_size:
        return (np.empty((0, window_size, features.shape[1])), np.empty(0), np.empty(0, dtype=int))
    starts = list(range(0, len(features) - window_size + 1, stride))
    X = np.stack([features[i:i + window_size] for i in starts])
    end_indices = np.array([i + window_size - 1 for i in starts], dtype=int)
    y = np.array([rul[idx] for idx in end_indices], dtype=float)
    return X, y, end_indices

def make_padded_test_window(
    features: np.ndarray,
    rul: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    n = features.shape[0]
    label = np.array([rul[-1]], dtype=float)

    if n >= window_size:
        X = features[-window_size:][np.newaxis].astype(float)  # [1, W, D]
        pad_mask = np.zeros(window_size, dtype=bool)
        return X, label, window_size, pad_mask

    n_pad = window_size - n
    pad_block = np.repeat(features[0:1], n_pad, axis=0)         # [n_pad, D]
    padded = np.concatenate([pad_block, features], axis=0)       # [W, D]
    X = padded[np.newaxis].astype(float)                         # [1, W, D]
    pad_mask = np.array([True] * n_pad + [False] * n, dtype=bool)
    return X, label, n, pad_mask


def preprocess(fd: int, data_dir: Path, out_dir: Path, window_size: int = None,
               stride: int = 1, valid_split: float = 0.1, random_seed: int = 42,
               include_ops: bool = False, normalization: str = 'global_minmax',
               n_condition_clusters: int = 6) -> Dict:
    if not 0 < valid_split < 1:
        raise ValueError("Validation fraction must be between 0 and 1.")
    if normalization not in {'global_minmax', 'condition_minmax'}:
        raise ValueError(f"Unknown normalization: {normalization}")
    if normalization == 'condition_minmax' and not include_ops:
        raise ValueError("Condition-wise scaling requires operating settings.")
    dataset = f'FD00{fd}'
    window_size = window_size or (60 if fd in {2, 4} else 40)
    train_raw, test_raw, test_rul = load_subset(fd, data_dir)
    drop = [name for name in CONST_DROP if name in train_raw.columns]
    train_raw, test_raw = train_raw.drop(columns=drop), test_raw.drop(columns=drop)
    sensor_cols = [name for name in train_raw.columns if name.startswith('s')]
    if len(sensor_cols) != 14:
        raise ValueError("Expected 14 sensor features.")
    op_cols = ['op1', 'op2', 'op3'] if include_ops else []
    feature_cols = op_cols + sensor_cols
    train_raw = add_rul(train_raw, cap=125)
    test_raw = add_test_rul(test_raw, test_rul, cap=125)
    train_ids, val_ids = _split_train_units(train_raw, valid_split, random_seed)
    fit_raw = train_raw[train_raw['unit'].isin(train_ids)].copy()
    val_raw = train_raw[train_raw['unit'].isin(val_ids)].copy()
    evaluation_raw = pd.concat([
        val_raw.assign(_preprocess_split='val'), test_raw.assign(_preprocess_split='test')
    ], axis=0, ignore_index=True)
    if normalization == 'condition_minmax':
        train_scaled, evaluation_scaled, scaler_info = condition_wise_minmax_scaling(
            fit_raw, evaluation_raw, op_cols, sensor_cols,
            n_clusters=n_condition_clusters, random_seed=random_seed,
        )
    else:
        train_scaled, evaluation_scaled, scaler_info = global_minmax_scaling(
            fit_raw, evaluation_raw, feature_cols,
        )
    val_scaled = evaluation_scaled[evaluation_scaled['_preprocess_split'] == 'val'].drop(columns=['_preprocess_split'])
    test_scaled = evaluation_scaled[evaluation_scaled['_preprocess_split'] == 'test'].drop(columns=['_preprocess_split'])
    X_train, y_train, uid_train, cycle_train = _make_train_windows(train_scaled, feature_cols, window_size, stride)
    X_val, y_val, uid_val, cycle_val = _make_train_windows(val_scaled, feature_cols, window_size, stride)
    windows, labels, units, cycles, masks = [], [], [], [], []
    for uid, group in test_scaled.groupby('unit'):
        features = group[feature_cols].values.astype(float)
        rul = group['RUL'].values.astype(float)
        window, target, _, mask = make_padded_test_window(features, rul, window_size)
        windows.append(window)
        labels.append(target)
        units.append(np.full(1, int(uid), dtype=np.int32))
        cycles.append(group['cycle'].values.astype(np.int32)[[-1]])
        masks.append(mask[np.newaxis])
    arrays = {
        'X_train': X_train, 'y_train': y_train, 'X_val': X_val, 'y_val': y_val,
        'X_test': np.concatenate(windows), 'y_test': np.concatenate(labels),
        'uid_train': uid_train, 'uid_val': uid_val, 'uid_test': np.concatenate(units),
        'cycle_end_train': cycle_train, 'cycle_end_val': cycle_val,
        'cycle_end_test': np.concatenate(cycles), 'pad_mask_test': np.concatenate(masks),
    }
    if len(arrays['X_test']) != EXPECTED_TEST_COUNTS[dataset] or len(arrays['X_test']) != len(test_rul):
        raise ValueError(f"{dataset}: expected one test window per official test engine.")
    if not len(X_train) or not len(X_val):
        raise ValueError("Training and validation splits must both contain complete windows.")
    if set(uid_train).intersection(uid_val):
        raise ValueError("Training and validation engine IDs overlap.")
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Preprocessed arrays contain non-finite values.")
    output = Path(out_dir) / dataset
    output.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(output / f'{name}.npy', array)
    scaler_info.update(fit_scope='unit_split_train_engines_only', fit_train_units=train_ids,
                       heldout_validation_units=val_ids, official_test_rows_used_for_fit=False)
    split = {'split_method': 'unit_split', 'valid_split': valid_split, 'random_seed': random_seed,
             'train_units': train_ids, 'val_units': val_ids,
             'test_units': sorted(int(uid) for uid in np.unique(arrays['uid_test']))}
    metadata = {
        'dataset': dataset, 'preprocessing': {'feature_cols': feature_cols, 'sensor_cols': sensor_cols,
            'op_cols': op_cols, 'include_ops': include_ops, 'dropped_sensors': drop,
            'normalization': normalization, 'rul_cap': 125},
        'windowing': {'window_size': window_size, 'stride': stride,
                      'test_keep_last_only': True, 'test_pad_strategy': 'repeat_first_row'},
        'split': split, 'scaler_info': scaler_info,
        'shapes': {name: list(array.shape) for name, array in arrays.items()},
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (output / 'split_engine_ids.json').write_text(json.dumps(split, indent=2))
    (output / 'scaler_metadata.json').write_text(json.dumps(scaler_info, indent=2))
    print(f"{dataset}: train={X_train.shape} val={X_val.shape} test={arrays['X_test'].shape}")
    return {'metadata': metadata, 'save_dir': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', type=Path, default=ROOT / 'dataset/raw/cmapss')
    parser.add_argument('--out_dir', type=Path)
    parser.add_argument('--datasets', nargs='+', type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    args = parser.parse_args()
    for fd in args.datasets:
        config = json.loads((ROOT / f'configs/cmapss/FD00{fd}.json').read_text())
        settings = config['preprocessing']
        output = args.out_dir or ROOT / config['data_dir']
        preprocess(fd=fd, data_dir=args.data_dir, out_dir=output, window_size=config['window_size'],
                   include_ops=config['include_ops'], stride=settings['stride'],
                   valid_split=settings['valid_split'], random_seed=settings['random_seed'],
                   normalization=settings['normalization'],
                   n_condition_clusters=settings['n_condition_clusters'] or 6)


if __name__ == '__main__':
    main()
