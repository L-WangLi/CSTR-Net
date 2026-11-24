"""CSTR-Net model and sensor-interaction layers."""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from .layers import (
    _DEFAULT_BRANCH_SPECS,
    _LONG_BRANCH_SPECS,
    _PerSensorGatedTemporalLayer,
    AdaptiveTemporalEvidenceFusion,
    DegradationContextEncoder,
    GraphReadout,
    RegressionHead,
)

FilmMode = Literal["none", "delta_gamma"]

class SensorAttention(nn.Module):

    def __init__(
        self,
        n_sensors: int,
        token_dim: int,
        dropout: float,
        num_heads: int = 2,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={token_dim} must be divisible by num_heads={num_heads}."
            )
        self.n_sensors = n_sensors
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)
        raw_init = math.log(alpha_init / (1.0 - alpha_init))
        self.raw_alpha = nn.Parameter(torch.tensor(raw_init))

    def forward(self, h: Tensor) -> Tensor:
        bsz, steps, n_sensors, channels = h.shape
        if n_sensors != self.n_sensors:
            raise ValueError(f"Expected {self.n_sensors} sensors, got {n_sensors}.")
        flat = h.reshape(bsz * steps, n_sensors, channels)
        ctx, _ = self.attn(flat, flat, flat, need_weights=False)
        alpha = torch.sigmoid(self.raw_alpha)
        out = self.norm(h + self.dropout(alpha * ctx.reshape(bsz, steps, n_sensors, channels)))
        return out

class TemporalBranch(nn.Module):

    def __init__(
        self,
        token_dim: int,
        kernel_size: int,
        dilation_1: int,
        dilation_2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer1 = _PerSensorGatedTemporalLayer(token_dim, kernel_size, dilation_1, dropout)
        self.layer2 = _PerSensorGatedTemporalLayer(token_dim, kernel_size, dilation_2, dropout)

    def forward(
        self,
        x: Tensor,
        sensor_interaction: nn.Module,
    ) -> Tensor:
        h = self.layer1(x)
        h = sensor_interaction(h)
        return self.layer2(h)

class MultiScaleEncoder(nn.Module):

    def __init__(
        self,
        n_sensors: int,
        token_dim: int,
        branch_specs: Tuple[Tuple[int, int, int], ...],
        dropout: float,
        input_features_per_sensor: int,
        num_heads: int = 2,
        alpha_init: float = 0.1,
        ops_dim: int = 0,
        film_mode: FilmMode = "none",
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if film_mode not in {"none", "delta_gamma"}:
            raise ValueError(f"Unknown film_mode: {film_mode}")
        self.input_features_per_sensor = int(input_features_per_sensor)
        self.ops_dim = int(ops_dim)
        self.film_mode = film_mode
        self.max_seq_len = int(max_seq_len)
        self.value_embed = nn.Linear(self.input_features_per_sensor, token_dim)
        self.sensor_id = nn.Embedding(n_sensors, token_dim)
        self.input_norm = nn.LayerNorm(token_dim)
        self.pos_embed = nn.Embedding(self.max_seq_len, token_dim)
        self.film = (
            nn.Linear(self.ops_dim, 2 * token_dim)
            if film_mode == "delta_gamma" and self.ops_dim > 0 else None
        )
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.dropout = nn.Dropout(dropout)
        self.sensor_interaction = SensorAttention(
            n_sensors=n_sensors, token_dim=token_dim, dropout=dropout,
            num_heads=num_heads, alpha_init=alpha_init,
        )
        self.branches = nn.ModuleList([
            TemporalBranch(token_dim, k, d1, d2, dropout)
            for k, d1, d2 in branch_specs
        ])

    def forward(self, sensors: Tensor, ops_context: Optional[Tensor] = None) -> list[Tensor]:
        if sensors.ndim == 3:
            sensors = sensors.unsqueeze(-1)
        if sensors.ndim != 4:
            raise ValueError(f"Expected sensors [B,T,S] or [B,T,S,F], got shape {tuple(sensors.shape)}.")
        _, _, n_sensors, features = sensors.shape
        if features != self.input_features_per_sensor:
            raise ValueError(
                f"Expected {self.input_features_per_sensor} features per sensor, got {features}."
            )
        if n_sensors != self.sensor_id.num_embeddings:
            raise ValueError(f"Expected {self.sensor_id.num_embeddings} sensors, got {n_sensors}.")
        value_tokens = self.value_embed(sensors)
        ids = self.sensor_id.weight.view(1, 1, n_sensors, -1)
        x_normed = self.input_norm(value_tokens + ids)
        steps = sensors.shape[1]
        if steps > self.max_seq_len:
            raise ValueError(f"Expected at most {self.max_seq_len} steps, got {steps}.")
        pos_ids = steps - 1 - torch.arange(steps, device=sensors.device)
        pe = self.pos_embed(pos_ids).view(1, steps, 1, -1)
        x_normed = x_normed + pe
        if self.film is not None and ops_context is not None:
            if ops_context.ndim != 2 or ops_context.shape[1] != self.ops_dim:
                raise ValueError(
                    f"Expected ops_context [B,{self.ops_dim}], got {tuple(ops_context.shape)}."
                )
            delta_gamma, beta = self.film(ops_context).chunk(2, dim=-1)
            x_normed = (
                (1.0 + delta_gamma[:, None, None, :]) * x_normed
                + beta[:, None, None, :]
            )
        x = self.dropout(x_normed)
        return [branch(x, self.sensor_interaction) for branch in self.branches]

class CSTRNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.05,
                 branch_specs: Tuple[Tuple[int, int, int], ...] = _DEFAULT_BRANCH_SPECS,
                 ops_dim: int = 3, sensor_dim_without_ops: int = 14,
                 num_heads: int = 2, alpha_init: float = 0.1,
                 film_mode: FilmMode = "delta_gamma",
                 max_seq_len: int = 64) -> None:
        super().__init__()
        if input_dim not in {sensor_dim_without_ops, sensor_dim_without_ops + ops_dim}:
            raise ValueError(f"Unexpected input dimension: {input_dim}")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ops_dim = ops_dim if input_dim == sensor_dim_without_ops + ops_dim else 0
        self.sensor_dim = input_dim - self.ops_dim
        self.input_features_per_sensor = self.ops_dim + 1
        self.temporal_encoder = MultiScaleEncoder(
            n_sensors=self.sensor_dim, token_dim=hidden_dim, branch_specs=branch_specs,
            dropout=dropout, input_features_per_sensor=self.input_features_per_sensor,
            num_heads=num_heads, alpha_init=alpha_init, ops_dim=self.ops_dim,
            film_mode=film_mode, max_seq_len=max_seq_len,
        )
        self.deg_context = DegradationContextEncoder(hidden_dim, dropout)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.sensor_pool = GraphReadout(hidden_dim, dropout)
        self.temporal_fusion = AdaptiveTemporalEvidenceFusion(hidden_dim, dropout)
        self.head = RegressionHead(hidden_dim, dropout)

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        sensors = x[:, :, self.ops_dim:].unsqueeze(-1)
        ops = x[:, :, :self.ops_dim]
        ops_per_sensor = ops.unsqueeze(2).expand(-1, -1, self.sensor_dim, -1)
        temporal_input = torch.cat([ops_per_sensor, sensors], dim=-1)
        ops_context = ops.mean(dim=1) if self.ops_dim else None
        branches = self.temporal_encoder(temporal_input, ops_context=ops_context)
        h = torch.stack(branches, dim=1).mean(dim=1)
        context = self.context_norm(self.deg_context(h, pad_mask=pad_mask))
        sequence, _ = self.sensor_pool(h, context, pad_mask=pad_mask)
        pooled, _, _ = self.temporal_fusion(sequence, context, pad_mask=pad_mask)
        return self.head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
