"""Construct the final CSTR-Net model."""
from __future__ import annotations

from models.cstr_net import CSTRNet, _DEFAULT_BRANCH_SPECS, _LONG_BRANCH_SPECS

MODEL_DEFAULTS = {
    "num_heads": 2, "alpha_init": 0.1, "film_mode": "delta_gamma", "max_seq_len": 64,
}


def build_model(args, input_dim):
    specs = _LONG_BRANCH_SPECS if args.long_branches else _DEFAULT_BRANCH_SPECS
    return CSTRNet(
        input_dim=input_dim, hidden_dim=args.hidden_dim, dropout=args.dropout,
        branch_specs=specs, **{key: getattr(args, key, value) for key, value in MODEL_DEFAULTS.items()},
    )


def build_ncmapss_model(args, film_mode, branch_specs=_LONG_BRANCH_SPECS):
    return CSTRNet(
        input_dim=18, ops_dim=4, sensor_dim_without_ops=14,
        hidden_dim=args.hidden_dim, dropout=args.dropout, branch_specs=branch_specs,
        num_heads=args.num_heads, alpha_init=getattr(args, "alpha_init", 0.1),
        film_mode=film_mode, max_seq_len=args.max_seq_len,
    )
