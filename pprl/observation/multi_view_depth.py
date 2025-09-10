import numpy as np
import open3d as o3d
from typing import List
from pprl.envs.sofaenv.pointcloud_obs import SofaEnvPointCloudObservations


class MultiViewDepthPC(SofaEnvPointCloudObservations):
    """
    Extends SofaEnvPointCloudObservations to merge point clouds from
    *all* Camera objects found in `env.cameras` (fallback to single view).
    """

    def __init__(self, env, num_points: int = 2048, **kwargs):
        super().__init__(env, **kwargs)
        self.num_points = num_points

    # ------------------------------------------------------------------
    # main hook – we override *only* the part that grabs the depth image
    # ------------------------------------------------------------------
    def pointcloud(self, observation) -> np.ndarray:
        pcs: List[np.ndarray] = []

        for cam in getattr(self.env, "cameras", [self.env.camera]):
            # Temporarily redirect the parent class to the current camera
            self.camera_object = (
                cam.sofa_object if hasattr(cam, "sofa_object") else cam
            )

            pc_i = super().pointcloud(observation)   # (Ni, 3) or (Ni,6)
            pcs.append(pc_i)

        merged = np.vstack(pcs)

        # Down‑sample / random sample so batch size is fixed
        if merged.shape[0] > self.num_points:
            idx = self.np_random.choice(
                merged.shape[0], self.num_points, replace=False
            )
            merged = merged[idx]

        return merged.astype(np.float32)

