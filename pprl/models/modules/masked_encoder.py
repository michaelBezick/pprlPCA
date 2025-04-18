from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerEncoder


class MaskedEncoder(nn.Module):
    def __init__(
        self,
        mask_ratio: float,
        transformer_encoder: TransformerEncoder,
        pos_embedder: nn.Module,
        mask_type: str = "rand",  # TODO: check if we need different mask_types
        padding_value: float = 0.0,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.pos_embedder = pos_embedder
        self.transformer_encoder = transformer_encoder
        self.padding_value = padding_value
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._init_weights)

    @property
    def embed_dim(self) -> int:
        return self.transformer_encoder.embed_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _mask_center_rand(self, center, padding_mask, noaug=False):
        B, G, _ = center.shape
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        # count how many center points are paddings in each batch
        num_padding_tokens = torch.count_nonzero(padding_mask, dim=-1)
        # calculate how many center points should be masked in each batch (considering that paddings should NOT be masked)
        # fewer real tokens => fewer masks
        num_non_padding_tokens = G - num_padding_tokens
        num_masks = (num_non_padding_tokens * self.mask_ratio).int()
        max_num_masks = torch.max(num_masks)

        overall_mask = torch.zeros([B, G])
        for i, (n_non_padding_tokens, n_masks) in enumerate(
            zip(num_non_padding_tokens, num_masks)
        ):
            mask = torch.hstack(
                [
                    # we only want a random mask in the range [0, non_padding_tokens]
                    torch.ones(n_masks),  # type: ignore
                    torch.zeros(n_non_padding_tokens - n_masks),
                ]
            )
            rand_idx = torch.randperm(len(mask))
            mask = mask[rand_idx]
            # since we want all masks to have the same number of ones and zeros, we first fill each tensor up with ones
            mask = torch.hstack([mask, torch.ones(max_num_masks - n_masks)])
            # and then fill each tensor with zeros until each tensor has length G
            mask = torch.hstack([mask, torch.zeros(G - len(mask))])
            overall_mask[i, :] = mask

        return overall_mask.bool().to(center.device)

    def forward(self, x, center_points, noaug=False):
        padding_mask = torch.all(center_points == self.padding_value, dim=-1)
        if self.mask_type == "rand":
            ae_mask = self._mask_center_rand(center_points, padding_mask, noaug=noaug)
        else:
            raise NotImplementedError

        batch_size, _, C = x.shape

        x_vis = x[~ae_mask].reshape(batch_size, -1, C)
        center_points_vis = center_points[~ae_mask].reshape(batch_size, -1, 3)
        pos = self.pos_embedder(center_points_vis)

        # recalculate padding mask
        vis_padding_mask = torch.all(center_points_vis == self.padding_value, dim=-1)
        x_vis = self.transformer_encoder(x_vis, pos, vis_padding_mask)
        x_vis = self.norm(x_vis)

        return x_vis, ae_mask, padding_mask
