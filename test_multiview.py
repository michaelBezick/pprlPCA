#!/usr/bin/env python3
# multi_view_smoke_test.py

import numpy as np
import open3d as o3d
from pprl.utils.o3d import np_to_o3d, o3d_to_np

from dependencies.sofa_env.sofa_env.scenes.thread_in_hole.thread_in_hole_env import (
    ThreadInHoleEnv,
    ObservationType,                  # <-- import the enum
)
from sofa_env.base import RenderMode

from pprl.envs.sofaenv.pointcloud_obs import (
    SofaEnvPointCloudObservations as PCObs,   # patched wrapper
)

# ------------------------------------------------------------------ #
# 1.  Camera ring (4 fixed views for this quick test)                #
# ------------------------------------------------------------------ #
train_cameras = [
    {"position": [   0, -175, 120], "lookAt": [10,  0, 55]},
    {"position": [-175,    0, 120], "lookAt": [ 10,  0, 55]},
    {"position": [   0,  175, 120], "lookAt": [10,  0, 55]},
    {"position": [ 0,    0, 200], "lookAt": [ 10,  0, 55]},
]

# ------------------------------------------------------------------ #
# 2.  Create the SOFA environment (depth images only)                #
# ------------------------------------------------------------------ #
base_env = ThreadInHoleEnv(
    create_scene_kwargs=dict(camera_configs=train_cameras),
    observation_type=ObservationType.DEPTH,   # LEGAL value
    render_mode=RenderMode.HEADLESS,          # no window
)

# ------------------------------------------------------------------ #
# 3.  Wrap it with the multi‑view point‑cloud wrapper                #
# ------------------------------------------------------------------ #
env = PCObs(
    base_env,
    obs_frame="world",
    random_downsample=None,   # fixed cloud size
    depth_cutoff=None,        # default 0.99 * max(depth)
    voxel_grid_size=None,
)

# ------------------------------------------------------------------ #
# 4.  Reset once and inspect                                         #
# ------------------------------------------------------------------ #
pcd, _ = env.reset()
print("Merged cloud shape:", pcd.shape)        # (2048, 3)  or  (2048, 6)

pcd = np_to_o3d(pcd)

diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound())
)

camera = [diameter, 0, diameter]
radius = diameter * 2

_, pt_map = pcd.hidden_point_removal(camera, radius)

pcd = pcd.select_by_index(pt_map)

pcd = o3d_to_np(pcd)

print("Dropout cloud shape:", pcd.shape)        # (2048, 3)  or  (2048, 6)

# Optional: save to disk for visual inspection
o3d.io.write_point_cloud(
    "multi_view_test2.ply",
    o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd[:, :3])),
)
print("Saved multi_view_test.ply – open it in MeshLab / CloudCompare.")

