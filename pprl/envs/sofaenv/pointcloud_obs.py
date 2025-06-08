from __future__ import annotations

from typing import Literal, Mapping, Sequence, Callable, List, Optional

import gymnasium as gym
import numpy as np
import open3d as o3d
from sofa_env.base import RenderMode, SofaEnv
from sofa_env.utils.camera import get_focal_length
import torch

from pprl.utils.o3d import np_to_o3d, o3d_to_np

from .. import PointCloudSpace

STATE_KEY = "state"

def np_to_o3d(array: np.ndarray):
    assert (shape := array.shape[-1]) in (3, 6)
    pos = array[:, :3]
    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pos))
    if shape == 6:
        color = array[:, 3:]
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def visualize(pos):
    try:
        o3d.visualization.draw_geometries([np_to_o3d(pos)])
    except:
        o3d.visualization.draw_geometries([pos])

def save_point_cloud(pcd, filename):
    pcd = np_to_o3d(pcd)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"pcd saved to {filename}")

class SofaEnvPointCloudObservations(gym.ObservationWrapper):
    def __init__(
        self,
        env: SofaEnv,
        depth_cutoff: float | None = None,
        max_expected_num_points: int | None = None,
        color: bool = False,
        crop: Mapping[str, Sequence[float]] | None = None,
        voxel_grid_size: float | None = None,
        random_downsample: int | None = None,
        n_target_points: int = 0,
        target_points_scale: float = 1,
        obs_frame: Literal["world", "camera"] = "camera",
        center: Sequence[float] | None = None,
        scale: float | None = None,
        normalize: bool = False,
        points_only: bool = True,
        points_key: str = "points",
        post_processing_functions: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None,
    ) -> None:
        super().__init__(env)
        self.depth_cutoff = depth_cutoff
        self.color = color
        self.post_processing_functions = post_processing_functions

        if crop is not None:
            self.crop_min = np.asarray(crop["min_bound"])
            self.crop_max = np.asarray(crop["max_bound"])
        else:
            self.crop_min, self.crop_max = None, None

        self.n_target_points = n_target_points
        self.target_points_scale = target_points_scale

        self.voxel_grid_size = voxel_grid_size
        self.random_downsample = random_downsample

        self.obs_frame = obs_frame
        self.center = np.asarray(center) if center is not None else center
        self.scale = scale
        self.normalize = normalize
        self.points_only = points_only
        self.points_key = points_key

        self.mode = None

        self._initialized = False

        if self.env.render_mode == RenderMode.NONE:
            raise ValueError(
                "RenderMode of environment cannot be RenderMode.NONE, if point clouds are to be created from OpenGL depth images."
            )

        if max_expected_num_points is None:
            max_expected_num_points = int(np.prod(self.env.observation_space.shape[:2]))  # type: ignore

        self.observation_space = PointCloudSpace(
            max_expected_num_points=max_expected_num_points,
            low=-np.float32("inf"),
            high=np.float32("inf"),
            feature_shape=(6,) if self.color else (3,),
        )

        self.max_expected_num_points = max_expected_num_points

    def reset(self, **kwargs):
        """Reads the data for the point clouds from the sofa_env after it is resetted."""

        # First reset calls _init_sim to setup the scene
        observation, reset_info = self.env.reset(**kwargs)

        if self.post_processing_functions:
            for fn in self.post_processing_functions:
                if hasattr(fn, "new_episode"):
                    fn.new_episode()

        if not self._initialized:
            env = self.env.unwrapped
            scene_creation_result = env.scene_creation_result
            if (
                not isinstance(scene_creation_result, dict)
                and "camera" in scene_creation_result
                and isinstance(
                    scene_creation_result["camera"],
                    (env.sofa_core.Object, env.camera_templates.Camera),
                )
            ):
                raise AttributeError(
                    "No camera was found to create a raycasting scene. Please make sure createScene() returns a dictionary with key 'camera' or specify the cameras for point cloud creation in camera_configs."
                )

            if isinstance(
                scene_creation_result["camera"],
                env.camera_templates.Camera,
            ):
                self.camera_object = scene_creation_result["camera"].sofa_object
            else:
                self.camera_object = scene_creation_result["camera"]

            # Read camera parameters from SOFA camera
            width = int(self.camera_object.widthViewport.array())
            height = int(self.camera_object.heightViewport.array())
            fx, fy = get_focal_length(self.camera_object, width, height)
            self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=width / 2,
                cy=height / 2,
            )

            self._initialized = True

        return self.observation(observation), reset_info

    def line_distance_sums(self, centered_points: torch.Tensor,
                           direction: torch.Tensor):
        """
        points      : (N,3) centered cloud (float32/64, CPU or CUDA)
        direction   : (3,)  line direction (need not be unit length)

        Returns
        -------
        pos_sum, neg_sum  (each scalar tensor)
        """

        v_hat = direction / direction.norm()          # (3,)
        dots  = centered_points @ v_hat                        # (N,)  u·v  (torch.mv is fine too)

        # squared distance ‖u‖² - (u·v̂)²
        r2 = centered_points.pow(2).sum(dim=1) - dots.pow(2)   # (N,)

        dists = r2.clamp_min_(0.)                 # avoid tiny neg. due to FP error

        # masks for the two half‑spaces
        pos_mask = dots > 0
        neg_mask = dots < 0

        pos_sum = dists[pos_mask].sum()
        neg_sum = dists[neg_mask].sum()
        return pos_sum, neg_sum

    def disambiguate(self, centered_pcd, eig_vecs):
        """
        Function takes in centered_pcd and returns PCA normalized cloud
        eig_vecs are row vecs
        """

        #if cos(theta) is +, then point is on one side of centroid, and vice versa

        differences = []
        for i in range(3):
            pos_sum, neg_sum = self.line_distance_sums(centered_pcd, eig_vecs[i])

            if neg_sum > pos_sum:
            # invert direction
                eig_vecs[i] *= -1

            differences.append(abs(pos_sum - neg_sum))

        stability_order = np.argsort(differences)[::-1]
        i1, i2 = stability_order[:2]

        v1 = eig_vecs[i1]
        v2 = eig_vecs[i2]

        v3 = torch.cross(v1, v2)
        v3 /= v3.norm()

        eig_vecs = torch.stack([v1, v2, v3], dim=0)


        return centered_pcd @ eig_vecs.T

    def observation(self, observation) -> np.ndarray | dict:
        """Replaces the observation of a step in a sofa_env scene with a point cloud."""

        """ maybe just add PCA translation here instead of in forward """

        pcd = self.pointcloud(observation)


        if self.voxel_grid_size is not None:
            pcd = np_to_o3d(pcd)
            pcd = pcd.voxel_down_sample(self.voxel_grid_size)
            pcd = o3d_to_np(pcd)


        centered = pcd[:, :3] - pcd[:, :3].mean(axis=0, keepdims=True)

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh.astype(np.float32)  # shape (3, 3)

        new_points = self.disambiguate(torch.tensor(centered, dtype=torch.float32), torch.tensor(components, dtype=torch.float32)).numpy()



        if self.points_only:
            return new_points
        else:
            return {
                STATE_KEY: observation,
                self.points_key: new_points,
            }

    def pointcloud(self, observation) -> np.ndarray:
        pcs = []
        cam_list = getattr(self.env.unwrapped, "cameras", [self.env.unwrapped.camera])


        for cam in cam_list:

            self.env.unwrapped._camera_object = (
                cam.sofa_object if hasattr(cam, "sofa_object") else cam
            )

            self.camera_object = self.env.unwrapped._camera_object

            self.env.unwrapped._update_sofa_visuals()

            # (re‑compute intrinsics if resolutions can differ)
            w = int(self.camera_object.widthViewport.array())
            h = int(self.camera_object.heightViewport.array())
            fx, fy = get_focal_length(self.camera_object, w, h)
            self.intrinsic.set_intrinsics(w, h, fx, fy, w / 2, h / 2)

            pc = self.pointcloud_old(observation)

            pcs.append(pc)


        merged = np.vstack(pcs)

        if self.post_processing_functions is not None:
            for fn in self.post_processing_functions:
                merged = fn(merged)

        if self.random_downsample is None:
            self.random_downsample = 900

        if (
            self.random_downsample is not None
            and (num_points := len(merged)) > self.random_downsample
        ):
            choice = self.np_random.choice(
                num_points, self.random_downsample, replace=False
            )
            merged = merged[choice]

        return merged.astype(np.float32)

    def pointcloud_old(self, observation) -> np.ndarray:
        """Returns a point cloud calculated from the depth image of the sofa scene"""
        # Get the depth image from the SOFA scene
        depth = self.env.unwrapped.get_depth_from_open_gl()

        if (depth_cutoff := self.depth_cutoff) is None:
            depth_cutoff = 0.99 * depth.max()

        if self.obs_frame == "world" or self.n_target_points > 0:
            cam_rotation = self.camera_object.orientation.array()
            # sofa and open3d use different quaternion conventions
            # in sofa, the real part is last
            # in open3d, the real part must come first
            cam_rotation = o3d.geometry.get_rotation_matrix_from_quaternion(
                cam_rotation[[-1, 0, 1, 2]]
            )
            cam_position = self.camera_object.position.array()

        if self.obs_frame == "world":
            extrinsic = compute_camera_extrinics(cam_rotation, cam_position)
        elif self.obs_frame == "camera":
            extrinsic = compute_camera_extrinics(np.identity(3), np.zeros(3))

        if self.color:
            rgb = observation

            # Calculate point cloud
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color=o3d.geometry.Image(np.ascontiguousarray(rgb)),
                depth=o3d.geometry.Image(np.ascontiguousarray(depth)),
                # depth is in meters, no need to rescale
                depth_scale=1.0,
                depth_trunc=depth_cutoff,
                convert_rgb_to_intensity=False,
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd,
                self.intrinsic,
                extrinsic=extrinsic,
            )
        else:
            # create_from_depth_image is supposed to do this cutoff, but it doesn't
            depth = np.where(depth > depth_cutoff, 0, depth)

            depth = o3d.geometry.Image(np.ascontiguousarray(depth))

            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth,
                self.intrinsic,
                extrinsic=extrinsic,
                depth_scale=1.0,
                # depth_trunc=depth_cutoff,
            )

        if self.crop_min is not None:
            pcd = pcd.crop(
                o3d.geometry.AxisAlignedBoundingBox(
                    min_bound=self.crop_min, max_bound=self.crop_max
                )
            )

        if self.voxel_grid_size is not None:
            pcd = pcd.voxel_down_sample(self.voxel_grid_size)

        pcd = o3d_to_np(pcd)

        if (
            self.random_downsample is not None
            and (num_points := len(pcd)) > self.random_downsample
        ):
            choice = self.np_random.choice(
                num_points, self.random_downsample, replace=False
            )
            pcd = pcd[choice]

        if self.n_target_points > 0:
            # goal position in homogenous coordinates
            goal_pos = np.ones((4,))
            goal_pos[:3] = self.env.unwrapped.target_position

            if self.obs_frame == "camera":
                world2cam = np.identity(4)
                inv_cam_rotation = cam_rotation.T
                world2cam[:3, :3] = inv_cam_rotation
                world2cam[:3, 3] = -inv_cam_rotation @ cam_position
                goal_pos = world2cam @ goal_pos

            goal_points = self.np_random.uniform(
                low=-self.target_points_scale,
                high=self.target_points_scale,
                size=(self.n_target_points, 3),
            ).astype(np.float32)
            goal_points = goal_points + goal_pos[:3]

            if self.color:
                goal_points_rgb = np.zeros_like(goal_points)
                # TODO: don't hardcode color
                # goal_points_rgb[...] = 1  # white
                goal_points = np.hstack((goal_points, goal_points_rgb))

            pcd = np.concatenate((pcd, goal_points))

        pos = pcd[:, :3]
        if self.center is not None:
            pos[...] -= self.center

        if self.scale is not None:
            pos[...] /= self.scale

        if self.normalize:
            center = pos.mean(axis=0)
            pos[...] -= center
            scale = 0.999999 / np.abs(pos).max()
            pos[...] *= scale

        return pcd


def compute_camera_extrinics(
    cam_rotation: np.ndarray,
    cam_position: np.ndarray,
) -> np.ndarray:
    # Flip around the z axis.
    # This is necessary because the camera looks in the negative z direction in SOFA,
    # but we invert z in get_depth_from_open_gl().
    rotate_about_x_axis = o3d.geometry.get_rotation_matrix_from_quaternion([0, 1, 0, 0])
    cam_rotation = np.matmul(cam_rotation, rotate_about_x_axis)

    # the extrinsic matrix is that it describes how the world is transformed relative to the camera
    # this is described with an affine transform, which is a rotation followed by a translation
    # https://ksimek.github.io/2012/08/22/extrinsic/
    assert cam_rotation.shape == (3, 3)
    assert cam_position.shape == (3,)
    inverse_rotation = cam_rotation.T
    extrinsic_matrix = np.identity(4)
    extrinsic_matrix[:3, :3] = inverse_rotation
    extrinsic_matrix[:3, 3] = -inverse_rotation @ cam_position
    return extrinsic_matrix
