from __future__ import annotations

from typing import Any, Sequence, SupportsFloat

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class PointCloudSpace(spaces.Box):
    def __init__(
        self,
        max_expected_num_points: int,
        low: SupportsFloat | NDArray[Any],
        high: SupportsFloat | NDArray[Any],
        feature_shape: Sequence[int] | None = None,
        dtype: type[np.floating[Any]] | type[np.integer[Any]] = np.float32,
        seed: int | None = None,
    ):
        super().__init__(low, high, feature_shape, dtype, seed)
        self.max_num_points = max_expected_num_points

    @property
    def max_expected_num_points(self) -> int:
        return self.max_num_points

    def sample(self) -> np.ndarray:
        n_points = self.np_random.integers(
            self.max_num_points // 2, self.max_num_points
        )

        sample_shape = (n_points,) + self.shape

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(sample_shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        sample[:, unbounded] = self.np_random.normal(
            size=(n_points,) + unbounded[unbounded].shape,
        )

        sample[:, low_bounded] = (
            self.np_random.exponential(
                size=(n_points,) + low_bounded[low_bounded].shape,
            )
            + self.low[low_bounded]
        )

        sample[:, upp_bounded] = (
            -self.np_random.exponential(
                size=(n_points,) + upp_bounded[upp_bounded].shape,
            )
            + self.high[upp_bounded]
        )

        sample[:, bounded] = self.np_random.uniform(
            low=self.low[bounded],
            high=high[bounded],
            size=(n_points,) + bounded[bounded].shape,
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        return sample

    def __repr__(self) -> str:
        return f"PointCloud({self.max_num_points}, {self.low_repr}, {self.high_repr}, {self.shape}, {self.dtype})"
