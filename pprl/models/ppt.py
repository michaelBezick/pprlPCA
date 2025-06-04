from __future__ import annotations

from typing import Callable

import gymnasium.spaces as spaces
import torch
import torch.nn as nn
from parllel import ArrayTree
from torch import Tensor

from pprl.envs import PointCloudSpace
from pprl.models.modules.tokenizer import Tokenizer
from pprl.models.modules.transformer import SequencePooling, TransformerEncoder
from pprl.utils.array_dict import dict_to_batched_data, dict_to_batched_data_pca
import numpy as np

import open3d as o3d

class PointPatchTransformer(nn.Module):
    def __init__(
        self,
        obs_space: spaces.Space,
        tokenizer: Callable,
        pos_embedder: Callable,
        transformer_encoder: Callable,
        embed_dim: int,
        state_embed_dim: int | None = None,
    ) -> None:
        super().__init__()

        if obs_is_dict := isinstance(obs_space, spaces.Dict):
            point_space = obs_space["points"]
        else:
            point_space = obs_space
        assert isinstance(point_space, PointCloudSpace)
        self.obs_is_dict = obs_is_dict

        point_dim = point_space.shape[0]
        self.tokenizer: Tokenizer = tokenizer(point_dim=point_dim, embed_dim=embed_dim)
        self.pos_embedder: nn.Module = pos_embedder(token_dim=embed_dim)
        self.transformer_encoder: TransformerEncoder = transformer_encoder(
            embed_dim=embed_dim
        )
        self.pooling = SequencePooling(embed_dim=embed_dim)

        # initialize weights of attention layers before creating further modules
        self.apply(self._init_weights)

        self.state_encoder = None
        if self.obs_is_dict:
            state_dim = sum(
                space.shape[0] for name, space in obs_space.items() if name != "points"
            )
            # maybe create linear projection layer for state vector
            if state_embed_dim is not None:
                self.state_encoder = nn.Linear(state_dim, state_embed_dim)
                self.state_dim = state_embed_dim
            else:
                self.state_dim = state_dim
        else:
            self.state_dim = 0

    def random_SO3(self, points):
        device, dtype = points.device, points.dtype

        q = torch.randn(4, device=device, dtype=dtype)
        q = q / q.norm()
        w, x, y, z = q

        R = torch.tensor([[1-2*(y*y+z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
                          [2*(x*y + z*w),   1-2*(x*x+z*z),     2*(y*z - x*w)],
                          [2*(x*z - y*w),   2*(y*z + x*w),     1-2*(x*x+y*y)]], device=device, dtype=dtype)
        
        return points @ R.T

    def forward(self, observation: ArrayTree[Tensor]) -> Tensor:


        point_cloud: ArrayTree[Tensor] = (
            observation["points"] if self.obs_is_dict else observation
        )


        points, ptr = point_cloud["pos"], point_cloud["ptr"]


        # preprocessing points to be PCA canonicalized

        PCA = False
        random_rotation = False


        correct_points_list = [
            points[ptr[i] : ptr[i+1] - 3, :3]
            for i in range(len(ptr) - 1)
        ] 


        if PCA:
            pca_basis_list = [
                points[ptr[i+1] - 3 : ptr[i+1], :3].T
                for i in range(len(ptr) - 1)
            ]

            pca_bases = torch.stack(pca_basis_list, dim = 0)
            pca_bases *= (torch.randint(0, 2, (len(pca_basis_list), 3), device=pca_bases.device) * 2 - 1).unsqueeze(1)

            """
            VERY IMPORTANT, PCA COLUMNS ARE THE BASES, AND ORTHOGONALITY HAS BEEN VERIFIED
            """

            pca_basis_list = list(pca_bases)

            rotated_list = [point_cloud @ pca_basis for point_cloud, pca_basis in zip(correct_points_list, pca_basis_list)]
        else:
            rotated_list = correct_points_list


        rotated_flat = torch.cat(rotated_list, dim=0)

        if random_rotation:
            rotated_flat = self.random_SO3(rotated_flat)

        ptr_shifted = ptr.clone()
        for i in range(1, len(ptr)):
            ptr_shifted[i] = ptr[i] - 3 * i

        # pos, batch, color = dict_to_batched_data(point_cloud)  # type: ignore
        pos, batch = dict_to_batched_data_pca(rotated_flat, ptr_shifted)
        color = None

        x, _, center_points = self.tokenizer(pos, batch, color)

        pos = self.pos_embedder(center_points)

        x = self.transformer_encoder(x, pos)

        encoder_out = self.pooling(x)

        if self.obs_is_dict:
            state = [elem for name, elem in observation.items() if name != "points"]
            if self.state_encoder is not None:
                state = torch.concatenate(state, dim=-1)
                state = self.state_encoder(state)
                state = [state]

            encoder_out = torch.concatenate([encoder_out] + state, dim=-1)

        return encoder_out

    @property
    def embed_dim(self) -> int:
        return self.transformer_encoder.embed_dim + self.state_dim

    def _init_weights(self, m):
        # TODO: verify weight init for various versions of point MAE/GPT
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
