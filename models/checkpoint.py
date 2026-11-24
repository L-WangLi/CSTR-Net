"""Load a final CSTR-Net checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from .cstr_net import CSTRNet, _DEFAULT_BRANCH_SPECS, _LONG_BRANCH_SPECS


def load_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in payload:
        state, config = payload["model_state_dict"], payload["config"]
    else:
        state = payload
        config = json.loads((checkpoint_path.parent / "config.json").read_text())
    expected = {"scale_fusion_mode": "fixed", "readout_mode": "context_last",
                "revin_mode": "none", "alpha_mode": "scalar", "scale_context_mode": "none"}
    for key, value in expected.items():
        if config.get(key, value) != value:
            raise ValueError(f"Unsupported checkpoint option: {key}={config[key]}")
    if config.get("scale_uniform_mix", 0.0) != 0.0 or config.get("single_branch", False):
        raise ValueError("Expected a final multi-scale CSTR-Net checkpoint.")
    features = state["temporal_encoder.value_embed.weight"].shape[1]
    sensors = state["temporal_encoder.sensor_id.weight"].shape[0]
    hidden_dim = state["temporal_encoder.value_embed.weight"].shape[0]
    ops_dim = features - 1
    specs = config.get("branch_specs", _LONG_BRANCH_SPECS if config.get("long_branches") else _DEFAULT_BRANCH_SPECS)
    specs = tuple(tuple(branch) for branch in specs)
    positions = state["temporal_encoder.pos_embed.weight"]
    model = CSTRNet(
        input_dim=sensors + ops_dim, hidden_dim=hidden_dim,
        dropout=config.get("dropout", 0.05), branch_specs=specs,
        ops_dim=ops_dim, sensor_dim_without_ops=sensors,
        film_mode="delta_gamma" if "temporal_encoder.film.weight" in state else "none",
        max_seq_len=positions.shape[0], num_heads=config.get("num_heads", 2),
        alpha_init=config.get("alpha_init", 0.1),
    )
    # Fixed-fusion checkpoints can contain unused scorer tensors.
    state = {key: value for key, value in state.items() if not key.startswith("scale_fusion.scorer.")}
    model.load_state_dict(state, strict=True)
    return model.eval(), config
