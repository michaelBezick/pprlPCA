#!/usr/bin/env python3
# multi_view_smoke_test.py

import numpy as np
import open3d as o3d
from pprl.utils.o3d import np_to_o3d, o3d_to_np

def hpr_partial(points: np.ndarray, eye: np.ndarray, min_pts=100) -> np.ndarray:
    """
    Same heuristic as your standalone script:
    radius = cloud diameter * 100
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    diameter = np.linalg.norm(points.max(0) - points.min(0))
    radius   = diameter * 1000
    _, idx = pcd.hidden_point_removal(eye, radius)

    idx = np.asarray(idx)

    if idx.size < min_pts:
        return points


    return points[idx]

from dependencies.sofa_env.sofa_env.scenes.thread_in_hole.thread_in_hole_env import (
    ThreadInHoleEnv,
    ObservationType,                  # <-- import the enum
)
from sofa_env.base import RenderMode

from pprl.envs.sofaenv.pointcloud_obs import (
    SofaEnvPointCloudObservations as PCObs,   # patched wrapper
)
train_cameras = [
    {"position": [   0, 175, 120], "lookAt": [10,  0, 55]},
    # {"position": [-175,    0, 120], "lookAt": [ 10,  0, 55]},
    # {"position": [   0,  175, 120], "lookAt": [10,  0, 55]},
]

base_env = ThreadInHoleEnv(
    create_scene_kwargs=dict(camera_configs=train_cameras),
    observation_type=ObservationType.DEPTH,   # LEGAL value
    render_mode=RenderMode.HEADLESS,          # no window
)

env = PCObs(
    base_env,
    obs_frame="world",
    random_downsample=100_000,   # fixed cloud size
    depth_cutoff=None,        # default 0.99 * max(depth)
    voxel_grid_size=None,
)

pcd, _ = env.reset()

o3d.io.write_point_cloud(
    "pca_frame.ply",
    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3])),
)
