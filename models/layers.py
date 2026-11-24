"""Temporal encoding, context aggregation, and prediction layers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import torch.nn.functional as F

from torch import Tensor, nn

_DEFAULT_BRANCH_SPECS: Tuple[Tuple[int, int, int], ...] = (
    (3, 1, 2),    # RF 7
    (3, 2, 8),    # RF 21
    (5, 4, 8),    # RF 49
)

_LONG_BRANCH_SPECS: Tuple[Tuple[int, int, int], ...] = (
    *_DEFAULT_BRANCH_SPECS,
    (5, 8, 16), 
)

class DegradationContextEncoder(nn.Module):

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, h: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        
        if pad_mask is None:
            mean_ts = h.mean(dim=(1, 2))
            max_ts = h.amax(dim=(1, 2))
            last_s = h[:, -1].mean(dim=1)
            std_ts = h.std(dim=(1, 2), unbiased=False)
        else:
            valid = (~pad_mask.to(device=h.device, dtype=torch.bool)).view(h.shape[0], h.shape[1], 1, 1)
            denom = (valid.sum(dim=(1, 2)) * h.shape[2]).clamp_min(1)
            masked = h.masked_fill(~valid, 0.0)
            mean_ts = masked.sum(dim=(1, 2)) / denom
            max_ts = h.masked_fill(~valid, float("-inf")).amax(dim=(1, 2))
            max_ts = torch.nan_to_num(max_ts, neginf=0.0)
            diff = (h - mean_ts[:, None, None, :]).masked_fill(~valid, 0.0)
            std_ts = torch.sqrt((diff.square().sum(dim=(1, 2)) / denom).clamp_min(0.0))
            last_s = _get_last_valid(h, pad_mask=pad_mask).mean(dim=1)
        return self.net(torch.cat([mean_ts, max_ts, last_s, std_ts], dim=-1))

class _PerSensorGatedTemporalLayer(nn.Module):

    def __init__(
        self,
        token_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        conv_kwargs = dict(
            in_channels=token_dim,
            out_channels=token_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=token_dim,
            bias=False,
        )
        self.filter_dw = nn.Conv1d(**conv_kwargs)
        self.gate_dw = nn.Conv1d(**conv_kwargs)
        self.pointwise = nn.Conv1d(token_dim, token_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: Tensor) -> Tensor:
        
        bsz, steps, n_sensors, channels = x.shape
        residual = x
        h = x.permute(0, 2, 3, 1).reshape(bsz * n_sensors, channels, steps)
        h = F.pad(h, (self.left_pad, 0))
        f = torch.tanh(self.filter_dw(h))
        g = torch.sigmoid(self.gate_dw(h))
        y = self.pointwise(f * g)
        y = y.reshape(bsz, n_sensors, channels, steps).permute(0, 3, 1, 2)
        return self.norm(residual + self.dropout(y))

def _get_last_valid(h: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
    if pad_mask is None:
        return h[:, -1]
    valid = ~pad_mask.to(device=h.device, dtype=torch.bool)
    time_idx = torch.arange(h.shape[1], device=h.device).unsqueeze(0)
    last_idx = time_idx.masked_fill(~valid, -1).amax(dim=1).clamp_min(0)
    return h[torch.arange(h.shape[0], device=h.device), last_idx]

class GraphReadout(nn.Module):

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: Tensor,
        context: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        h_last = _get_last_valid(h, pad_mask=pad_mask)
        q = self.query(context).unsqueeze(1)
        scores = (q * h_last).sum(dim=-1) / (h.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        pooled = (h * weights[:, None, :, None]).sum(dim=2)
        return pooled, weights

class TemporalAttentionPooling(nn.Module):

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(token_dim, max(token_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(token_dim // 2, 1), 1),
        )

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        scores = self.scorer(x).squeeze(-1)
        if pad_mask is not None:
            mask = pad_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

class AdaptiveTemporalEvidenceFusion(nn.Module):

    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        self.tap = TemporalAttentionPooling(token_dim, dropout)
        self.gate = nn.Sequential(
            nn.Linear(4 * token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, 3),
        )
        self.out = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        pad_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if pad_mask is None:
            last = x[:, -1]
        else:
            valid = ~pad_mask.to(device=x.device, dtype=torch.bool)
            time_idx = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            last_idx = time_idx.masked_fill(~valid, -1).amax(dim=1).clamp_min(0)
            last = x[torch.arange(x.shape[0], device=x.device), last_idx]

        attn, temporal_weights = self.tap(x, pad_mask=pad_mask)
        mean = x.mean(dim=1) if pad_mask is None else self._masked_mean(x, pad_mask)
        gates = torch.softmax(self.gate(torch.cat([last, attn, mean, context], dim=-1)), dim=-1)
        fused = gates[:, 0:1] * last + gates[:, 1:2] * attn + gates[:, 2:3] * mean
        return self.out(fused), gates, temporal_weights

    @staticmethod
    def _masked_mean(x: Tensor, pad_mask: Tensor) -> Tensor:
        valid = (~pad_mask.to(device=x.device, dtype=torch.bool)).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1)
        return (x * valid).sum(dim=1) / denom

class RegressionHead(nn.Module):
    def __init__(self, token_dim: int, dropout: float) -> None:
        super().__init__()
        mid = max(token_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(token_dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)
