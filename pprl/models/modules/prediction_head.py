from __future__ import annotations

import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, dim: int, group_size: int, point_dim: int):
        super().__init__()
        self.point_dim = point_dim
        self.head = nn.Conv1d(dim, point_dim * group_size, 1)
        self.apply(self._init_weights)

    def forward(self, x):
        B, M, _ = x.shape
        prediction = self.head(x.transpose(1, 2)).transpose(1, 2)
        prediction = prediction.reshape(B, M, -1, self.point_dim)
        return prediction

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
