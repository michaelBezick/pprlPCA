from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Literal, Mapping, Sequence

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from mani_skill2.envs.sapien_env import BaseEnv as SapienBaseEnv
from sapien.core import Pose

from pprl.utils.o3d import np_to_o3d, o3d_to_np
import torch

import open3d as o3d
from .. import PointCloudSpace
from pathlib import Path

STATE_KEY = "state"

# These functions and wrappers are adapted/inspired by the ones found in https://github.com/haosulab/ManiSkill2-Learn


def apply_pose_to_points(x, pose):
    return to_normal(to_generalized(x) @ pose.to_transformation_matrix().T)

def save_point_cloud(pcd, filename):
    pcd = np_to_o3d(pcd)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"pcd saved to {filename}")

def to_generalized(x):
    if x.shape[-1] == 4:
        return x
    assert x.shape[-1] == 3
    output_shape = list(x.shape)
    output_shape[-1] = 4
    ret = np.ones(output_shape).astype(np.float32)
    ret[..., :3] = x
    return ret


def to_normal(x):
    if x.shape[-1] == 3:
        return x
    assert x.shape[-1] == 4
    return x[..., :3] / x[..., 3:]


def merge_dicts(ds, asarray=False):
    """Merge multiple dicts with the same keys to a single one."""
    # NOTE(jigu): To be compatible with generator, we only iterate once.
    ret = defaultdict(list)
    for d in ds:
        for k in d:
            ret[k].append(d[k])
    ret = dict(ret)
    # Sanity check (length)
    assert len(set(len(v) for v in ret.values())) == 1, "Keys are not same."
    if asarray:
        ret = {k: np.concatenate(v) for k, v in ret.items()}
    return ret

def farthest_point_sampling(points, num_samples, init_idx=None):
    """
    Fast Farthest Point Sampling (FPS) from a point cloud.

    Parameters:
        points (np.ndarray): (N, 3) array of points
        num_samples (int): Number of points to sample
        init_idx (int or None): Optional starting index

    Returns:
        np.ndarray: (num_samples,) indices of sampled points
    """

    if points.shape[0] <= num_samples:
        return points

    points = np_to_o3d(points)
    points = points.farthest_point_down_sample(num_samples)
    points = o3d_to_np(points)
    return points


class PointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: SapienBaseEnv,
        max_expected_num_points: int | None = None,
        color: bool = False,
        crop: Mapping[str, Sequence[float]] | None = None,
        n_target_points: int = 0,
        target_points_scale: float = 1,
        voxel_grid_size: float | None = None,
        exclude_handle_points: bool = False,
        handle_voxel_grid_size: float | None = None,
        random_downsample: int | None = None,
        obs_frame: Literal["world", "base", "ee"] = "world",
        normalize: bool = False,
        points_only: bool = True,
        points_key: str = "points",
        our_method: bool = True,
        use_fps: bool = True,
        fps_n: int = 0,
    ) -> None:
        super().__init__(env)
        self.color = color

        self.our_method = our_method
        self.use_fps = use_fps
        self.fps_n = fps_n

        if crop is not None:
            self.crop_min = np.asarray(crop["min_bound"])
            self.crop_max = np.asarray(crop["max_bound"])
        else:
            self.crop_min, self.crop_max = None, None

        self.n_target_points = n_target_points
        self.target_points_scale = target_points_scale

        if not (voxel_grid_size is None or random_downsample is None):
            raise ValueError(
                "Cannot do both voxel downsampling and random downsampling."
            )
        self.voxel_grid_size = voxel_grid_size
        self.exclude_handle_points = exclude_handle_points
        self.handle_voxel_grid_size = handle_voxel_grid_size
        self.random_downsample = random_downsample

        self.obs_frame = obs_frame
        self.normalize = normalize
        self.points_only = points_only
        self.points_key = points_key

                # ---- PLY dump config (tweak as you like) ----
        self.dump_ply_enable      = True        # turn on/off dumping
        self.dump_ply_dir         = Path("pcd_dumps")  # output folder
        self.dump_ply_prefix      = "overhead"  # file name prefix
        self.dump_ply_max         = 50          # number of frames to save
        self.dump_ply_exit_on_done = False      # True => raise SystemExit after saving 50
        self._dump_ply_count      = 0           # internal counter

        self.dump_ply_dir.mkdir(parents=True, exist_ok=True)

        self.pcd_idx = 0


        wrapped_space = self.env.observation_space

        if max_expected_num_points is None:
            max_expected_num_points = wrapped_space["pointcloud"]["xyzw"].shape[0]

        point_cloud_space = PointCloudSpace(
            max_expected_num_points=max_expected_num_points,
            low=-np.float32("inf"),
            high=np.float32("inf"),
            feature_shape=(6,) if self.color else (3,),
        )

        if not self.points_only:
            self.state_space = spaces.Dict(
                {"agent": wrapped_space["agent"], "extra": wrapped_space["extra"]}
            )
            self.observation_space = spaces.Dict(
                {
                    self.points_key: point_cloud_space,
                    STATE_KEY: spaces.flatten_space(self.state_space),
                }
            )
        else:
            self.observation_space = point_cloud_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Modifies the :attr:`env` after calling :meth:`reset`, returning a modified observation using :meth:`self.observation`."""

        obs, info = self.env.reset(seed=seed, options=options)

        if self.exclude_handle_points:
            cabinet_links = self.env.unwrapped.cabinet.get_links()
            self.handle_mesh_ids = [
                mesh.visual_id
                for link in cabinet_links
                for mesh in link.get_visual_bodies()
                if "handle" in mesh.name
            ]

        return self.observation(obs), info

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

    def score_r4(self, centered_points: torch.Tensor,
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

            r2 = centered_points.pow(2).sum(dim=1)
            
            r4 = r2 * r2

            r4 = r4.clamp_min_(0.)                 # avoid tiny neg. due to FP error

            # masks for the two half‑spaces
            pos_mask = dots > 0
            neg_mask = dots < 0

            pos_sum = r4[pos_mask].sum()
            neg_sum = r4[neg_mask].sum()
            return pos_sum, neg_sum

    def disambiguate(self, centered_pcd, eig_vecs):
        """
        Function takes in centered_pcd and returns PCA normalized cloud
        eig_vecs are row vecs
        """

        #if cos(theta) is +, then point is on one side of centroid, and vice versa
        for i in range(2):
            pos_sum, neg_sum = self.score_r4(centered_pcd, eig_vecs[i])

            if neg_sum > pos_sum:
            # invert direction
                eig_vecs[i] *= -1

        v1 = eig_vecs[0]
        v2 = eig_vecs[1]

        v3 = torch.cross(v1, v2)
        v3 /= v3.norm()

        eig_vecs = torch.stack([v1, v2, v3], dim=0)

        return centered_pcd @ eig_vecs.T

    def _save_ply(self, arr: np.ndarray, path: Path) -> None:
        """arr: (N,3) or (N,6) xyz[+rgb]. Saves binary .ply via Open3D."""
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
        if arr.shape[1] >= 6:
            cols = arr[:, 3:6].astype(np.float64)
            if cols.max() > 1.0:
                cols = cols / 255.0
            pc.colors = o3d.utility.Vector3dVector(cols)
        o3d.io.write_point_cloud(str(path), pc, write_ascii=False, compressed=False)


    def observation(self, observation: dict) -> np.ndarray | dict:
        """Replaces the observation of a step in a maniskill2 scene with a point cloud."""

        """Furthermore, removes goal / target proprioception"""

        pcd = self.pointcloud(observation)

        if (self.our_method):

            if (self.use_fps):
            # FPS
                pcd = farthest_point_sampling(pcd, self.fps_n)

            # PCA
            centered = pcd[:, :3] - pcd[:, :3].mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            components = vh.astype(np.float32)  # shape (3, 3)

            new_points = self.disambiguate(torch.tensor(centered, dtype=torch.float32), torch.tensor(components, dtype=torch.float32)).numpy()

            if (self.color):
                new_points = np.concatenate([new_points, pcd[:, 3:]], axis=1)
        else:
            new_points = pcd


        MIN_POINTS = 150

        n, d = new_points.shape
        if n < MIN_POINTS:
            pad_rows = MIN_POINTS - n
            pad = np.zeros((pad_rows, d), dtype=new_points.dtype)
            new_points = np.vstack([new_points, pad])

        if self.points_only:
            return new_points
        else:

            """
            IMPORTANT: remove all target information through zeroing
            """
            for key in list(observation['extra'].keys()):
                if "target" in key:
                    #just zero out values
                    data = observation['extra'][key]
                    observation['extra'][key] = np.zeros_like(data)


            state = spaces.flatten(
                self.state_space,
                {"agent": observation["agent"], "extra": observation["extra"]},
            )
            return {
                STATE_KEY: state,
                self.points_key: pcd,
            }

    def pointcloud(self, observation) -> np.ndarray:

        point_cloud = observation["pointcloud"]["xyzw"]
        if self.exclude_handle_points:
            mesh_segmentation = observation["pointcloud"]["Segmentation"][..., 0]

        # filter out points beyond z_far of camera
        mask = point_cloud[..., -1] == 1
        point_cloud = point_cloud[mask][..., :3]
        if self.exclude_handle_points:
            mesh_segmentation = mesh_segmentation[mask]

        if self.color:
            point_cloud_rgb = observation["pointcloud"]["rgb"]
            point_cloud_rgb = point_cloud_rgb[mask]
            point_cloud_rgb = point_cloud_rgb.astype(np.float32) / 255.0
            point_cloud = np.hstack((point_cloud, point_cloud_rgb))

        if self.crop_min is not None:
            assert self.crop_max is not None
            pos = point_cloud[:, :3]
            mask = np.all((pos >= self.crop_min) & (pos <= self.crop_max), axis=-1)
            point_cloud = point_cloud[mask]
            if self.exclude_handle_points:
                mesh_segmentation = mesh_segmentation[mask]

        if self.n_target_points > 0:
            # TODO: don't hardcode target dict key
            goal_pos = observation["extra"]["target_link_pos"]
            goal_points = self.np_random.uniform(
                low=-self.target_points_scale,
                high=self.target_points_scale,
                size=(self.n_target_points, 3),
            ).astype(np.float32)
            goal_points = goal_points + goal_pos

            if self.color:
                goal_points_rgb = np.zeros_like(goal_points)
                # TODO: don't hardcode color
                goal_points_rgb[:, 0] = 1  # red
                goal_points = np.hstack((goal_points, goal_points_rgb))

            point_cloud = np.concatenate((point_cloud, goal_points))

            if self.exclude_handle_points:
                mesh_segmentation = np.concatenate(
                    (
                        mesh_segmentation,
                        np.zeros(shape=goal_points.shape[:1], dtype=np.bool_),
                    )
                )

        if self.voxel_grid_size is not None:
            if self.exclude_handle_points:
                # if mesh_segmentation available, use it to find handle points
                handle_mask = np.isin(mesh_segmentation, self.handle_mesh_ids)
                if handle_found := bool(handle_mask.sum()):
                    # only split handle points if they exist
                    handle_points = point_cloud[handle_mask]
                    point_cloud = point_cloud[~handle_mask]

            point_cloud = np_to_o3d(point_cloud)
            point_cloud = point_cloud.voxel_down_sample(self.voxel_grid_size)
            point_cloud = o3d_to_np(point_cloud)

            if self.exclude_handle_points and handle_found:
                # only concatenate if handle points exist
                if self.handle_voxel_grid_size is not None:
                    # additionally downsample handle points first if requested
                    handle_points = np_to_o3d(handle_points)
                    handle_points = handle_points.voxel_down_sample(
                        self.handle_voxel_grid_size
                    )
                    handle_points = o3d_to_np(handle_points)

                point_cloud = np.concatenate((point_cloud, handle_points))
        elif (
            self.random_downsample is not None
            and (num_points := len(point_cloud)) > self.random_downsample
        ):
            choice = self.env.np_random.choice(
                num_points, self.random_downsample, replace=False
            )
            point_cloud = point_cloud[choice]

        if self.obs_frame == "base":
            # TODO: not sure if this is always valid
            base_pose = observation["agent"]["base_pose"]
            p, q = base_pose[:3], base_pose[3:]
            to_origin = Pose(p=p, q=q).inv()
            point_cloud[..., :3] = apply_pose_to_points(point_cloud[..., :3], to_origin)
        elif self.obs_frame == "ee":
            tcp_poses = observation["extra"]["tcp_pose"]
            tcp_pose = tcp_poses if tcp_poses.ndim == 1 else tcp_poses[0]
            p, q = tcp_pose[:3], tcp_pose[3:]
            to_origin = Pose(p=p, q=q).inv()
            point_cloud[..., :3] = apply_pose_to_points(point_cloud[..., :3], to_origin)

        if self.normalize:
            pos = point_cloud[:, :3]
            center = pos.mean(axis=0)
            pos[...] -= center
            scale = 0.999999 / np.abs(pos).max()
            pos[...] *= scale

        pca = False
        # print(point_cloud)
        # exit()
        if pca:
            # Append PCA components as fake points
            centered = point_cloud[:, :3] - point_cloud[:, :3].mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            components = vh.astype(np.float32)  # shape (3, 3)


            # Concatenate to point cloud
            point_cloud = np.concatenate([point_cloud, components], axis=0)
            dummy = np.array([[-100, -100, -100]])
            point_cloud = np.concatenate(([point_cloud, dummy]), axis=0)

        return point_cloud


class SafePointCloudWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: SapienBaseEnv,
        min_num_points: int | None = None,
        points_key: str = "points",
    ) -> None:
        super().__init__(env)
        self.min_num_points = min_num_points

        wrapped_space = self.env.observation_space

        if isinstance(wrapped_space, spaces.Box):
            assert len(wrapped_space.shape) == 1
            self.dict_observation = False
            self.point_dim = wrapped_space.shape[0]

        elif isinstance(wrapped_space, spaces.Dict):
            self.dict_observation = True
            self.points_key = points_key
            self.point_dim = wrapped_space[points_key].shape[0]

    def observation(self, observation: Any) -> Any:
        pcd = observation[self.points_key] if self.dict_observation else observation

        if self.min_num_points is not None and len(pcd) < self.min_num_points:
            pcd = np.concatenate(
                (
                    pcd,
                    np.zeros(
                        (self.min_num_points - len(pcd), self.point_dim),
                        dtype=np.float32,
                    ),
                ),
                axis=0,
            )

        if self.dict_observation:
            observation[self.points_key] = pcd
        else:
            observation = pcd

        return observation


class FrameStackWrapper(gym.ObservationWrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = deque(maxlen=self.num_frames)

        # TODO: we currently hardcode 3 cameras
        if isinstance(self.env.observation_space, gym.spaces.Box):
            point_cloud_shape = self.env.observation_space.shape
            self.observation_space = gym.spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=(point_cloud_shape[0] * 3, point_cloud_shape[1] + 1),  # type: ignore
            )
        elif isinstance(self.env.observation_space, gym.spaces.Dict):
            point_cloud_shape = self.env.observation_space["point_cloud"].shape
            point_cloud_space = gym.spaces.Box(
                low=-np.float32("inf"),
                high=np.float32("inf"),
                shape=(point_cloud_shape[0] * 3, point_cloud_shape[1] + 1),
            )
            state_space = self.env.observation_space["state"]
            self.observation_space = gym.spaces.Dict(
                {"point_cloud": point_cloud_space, "state": state_space}
            )

    def observation(self, observation):
        for i, frame in enumerate(self.frames):
            frame[..., -1] = i

        return np.concatenate(self.frames)

    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(observation, dict):
            point_cloud = observation["point_cloud"]
            point_cloud = self._add_column(point_cloud)
            self.frames.append(point_cloud)
            return (
                {
                    "point_cloud": self.observation(observation),
                    "state": observation["state"],
                },
                reward,
                terminated,
                truncated,
                info,
            )

        observation = self._add_column(observation)
        self.frames.append(observation)
        return self.observation(observation), reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if isinstance(observation, dict):
            point_cloud = observation["point_cloud"]
            point_cloud = self._add_column(point_cloud)
            self.frames.append(point_cloud)
            return {
                "point_cloud": self.observation(observation),
                "state": observation["state"],
            }, info

        observation = self._add_column(observation)
        for _ in range(self.num_frames):
            self.frames.append(observation)
        return self.observation(observation), info

    def _add_column(self, observation):
        return np.hstack([observation, np.zeros_like(observation[..., None, 0])])


# class ManiFrameStack(gym.ObservationWrapper):
#     def __init__(
#         self,
#         env,
#         image_shape,
#         filter_points_below_z,
#         voxel_grid_size: float | None = None,
#         normalize: bool = True,
#         num_frames: int = 4,
#         num_classes: int = 75,
#         convert_to_ee_frame: bool = True,
#         convert_to_base_frame: bool = False,
#         n_goal_points: None | int = None,
#         add_state: bool = False,
#         add_seg: bool = False,
#     ):
#         super().__init__(env)
#         self.num_frames = num_frames
#         self.num_classes = num_classes
#         self.frames = deque(maxlen=self.num_frames)
#         self.filter_points_below_z = filter_points_below_z
#         self.convert_to_ee_frame = convert_to_ee_frame
#         self.voxel_grid_size = voxel_grid_size
#         self.normalize = normalize
#         self.n_goal_points = n_goal_points
#         self.add_state = add_state
#         self.convert_to_base_frame = convert_to_base_frame
#         self.add_seg = add_seg
#         assert not (convert_to_base_frame and convert_to_ee_frame)
#
#         # shape is hardcoded for 3 cameras and two segmentation masks
#         #
#         point_dim = 6 if add_seg else 4
#         point_cloud_space = gym.spaces.Box(
#             low=-float("inf"),
#             high=float("inf"),
#             shape=(image_shape[0] * image_shape[1] * num_frames * 3, point_dim),
#         )
#         if not self.add_state:
#             self.observation_space = point_cloud_space
#         else:
#             agent_space = self.env.observation_space["agent"]
#             state_size = (
#                 agent_space["qpos"].shape[0]
#                 + agent_space["qvel"].shape[0]
#                 + agent_space["base_pose"].shape[0]
#                 + agent_space["base_vel"].shape[0]
#                 + agent_space["base_ang_vel"].shape[0]
#             )
#             self.observation_space = gym.spaces.Dict(
#                 {
#                     "point_cloud": point_cloud_space,
#                     "state": gym.spaces.Box(
#                         low=-float("inf"), high=float("inf"), shape=(state_size,)
#                     ),
#                 }
#             )
#
#     def _point_cloud(self, observation):
#         image_obs = observation["image"]
#         camera_params = observation["camera_param"]
#         pointcloud_obs = OrderedDict()
#
#         for cam_uid, images in image_obs.items():
#             cam_pcd = {}
#
#             position = images["Position"]
#             position[..., 3] = position[..., 2] < 0
#
#             # Convert to world space
#             cam2world = camera_params[cam_uid]["cam2world_gl"]
#             xyzw = position.reshape(-1, 4) @ cam2world.T
#             cam_pcd["xyzw"] = xyzw
#
#             if self.add_seg:
#                 seg = images["Segmentation"].reshape(-1, 4)[..., :3]
#                 cam_pcd["segmentation"] = seg
#             else:
#                 seg = None
#
#             pointcloud_obs[cam_uid] = cam_pcd
#
#         pointcloud_obs = merge_dicts(pointcloud_obs.values())
#
#         if self.add_seg:
#             seg = np.asarray(pointcloud_obs["segmentation"])[..., :3]
#
#         point_cloud = np.asarray(pointcloud_obs["xyzw"])[..., :3]
#
#         if self.n_goal_points is not None:
#             goal_pos = observation["extra"]["target_link_pos"]
#             goal_pts_xyz = (
#                 np.random.uniform(low=-1.0, high=1.0, size=(3, self.n_goal_points, 3))
#                 * 0.1
#             ).astype(np.float32)
#             goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
#
#             if self.add_seg:
#                 goal_pts_seg = np.zeros_like(goal_pts_xyz)
#                 goal_pts_seg[..., 1:] = self.num_classes
#                 seg = np.concatenate([seg, goal_pts_seg], axis=-2)
#
#             point_cloud = np.concatenate([point_cloud, goal_pts_xyz], axis=-2)
#
#         if self.add_seg:
#             out = np.dstack(
#                 (
#                     point_cloud,
#                     seg / self.num_classes,
#                 ),
#             )
#         else:
#             out = point_cloud
#
#         out = out[out[..., 2] > self.filter_points_below_z]
#
#         if self.convert_to_ee_frame:
#             tcp_poses = observation["extra"]["tcp_pose"]
#             assert tcp_poses.ndim <= 2
#             if tcp_poses.ndim == 2:
#                 tcp_pose = tcp_poses[
#                     0
#                 ]  # use the first robot hand's tcp pose as the end-effector frame
#             else:
#                 tcp_pose = tcp_poses  # only one robot hand
#             p, q = tcp_pose[:3], tcp_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#             out[..., :3] = apply_pose_to_points(out[..., :3], to_origin)
#
#         if self.convert_to_base_frame:
#             base_pose = observation["agent"]["base_pose"]
#             p, q = base_pose[:3], base_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#             out[..., :3] = apply_pose_to_points(out[..., :3], to_origin)
#
#         if self.voxel_grid_size is not None:
#             out = voxel_grid_sample(out, self.voxel_grid_size)
#
#         if not self.add_seg:
#             frames = np.zeros(out.shape[:-1] + (1,), dtype=out.dtype)
#             out = np.hstack([out, frames])
#
#         return out.reshape(-1, out.shape[-1])
#
#     def observation(self, observation):
#         for i, frame in enumerate(self.frames):
#             frame[..., -1] = i / self.num_frames
#
#         out = np.concatenate(self.frames)
#
#         # normalize full point cloud
#         if self.normalize:
#             out = normalize(out)
#
#         if self.add_state:
#             agent_obs = observation["agent"]
#             qpos = agent_obs["qpos"]
#             qvel = agent_obs["qvel"]
#             base_pose = agent_obs["base_pose"]
#             base_vel = agent_obs["base_vel"]
#             base_ang_vel = agent_obs["base_ang_vel"].reshape(
#                 -1,
#             )
#             state = np.concatenate([qpos, qvel, base_pose, base_vel, base_ang_vel])
#             out = {"point_cloud": out, "state": state}
#
#         return out
#
#     def step(self, action):
#         observation, reward, terminated, truncated, info = self.env.step(action)
#         self.frames.append(self._point_cloud(observation))
#         return self.observation(observation), reward, terminated, truncated, info
#
#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         [self.frames.append(self._point_cloud(obs)) for _ in range(self.num_frames)]
#         return self.observation(obs), info
#
#
# class ManiSkillPointCloudWrapper(gym.ObservationWrapper):
#     def __init__(
#         self,
#         env,
#         n_goal_points: int | None = None,
#         obs_frame: str = "ee",
#         use_color: bool = False,
#         filter_points_below_z: float | None = None,
#         voxel_grid_size: float | None = None,
#         normalize: bool = True,
#     ):
#         super().__init__(env)
#         num_points = env.observation_space["pointcloud"]["xyzw"].shape[0]
#         point_dim = 6 if use_color else 3
#         self.observation_space = gym.spaces.Box(
#             high=np.inf, low=-np.inf, shape=(num_points, point_dim)
#         )
#         self.env = env
#         self.n_goal_points = n_goal_points
#         self.obs_frame = obs_frame
#         self.use_color = use_color
#         self.filter_points_below_z = filter_points_below_z
#         self.voxel_grid_size = voxel_grid_size
#         self.normalize = normalize
#
#     def observation(self, observation):
#         if self.obs_frame in ["base", "world"]:
#             base_pose = observation["agent"]["base_pose"]
#             p, q = base_pose[:3], base_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#         elif self.obs_frame == "ee":
#             tcp_poses = observation["extra"]["tcp_pose"]
#             assert tcp_poses.ndim <= 2
#             if tcp_poses.ndim == 2:
#                 tcp_pose = tcp_poses[
#                     0
#                 ]  # use the first robot hand's tcp pose as the end-effector frame
#             else:
#                 tcp_pose = tcp_poses  # only one robot hand
#             p, q = tcp_pose[:3], tcp_pose[3:]
#             to_origin = Pose(p=p, q=q).inv()
#         else:
#             raise ValueError(f"Invalid obs_frame: {self.obs_frame}")
#
#         point_cloud = observation["pointcloud"]["xyzw"]
#         mask = point_cloud[..., -1] == 1
#         point_cloud = point_cloud[mask][..., :-1]
#
#         rgb = observation["pointcloud"]["rgb"].astype(np.float32)
#         rgb = rgb[mask]
#         rgb *= 1 / 255.0
#
#         segmentation = observation["pointcloud"].get("Segmentation")
#
#         if self.filter_points_below_z is not None:
#             filter_mask = point_cloud[..., 2] > self.filter_points_below_z
#             point_cloud = point_cloud[filter_mask]
#             rgb = rgb[filter_mask]
#             if segmentation is not None:
#                 segmentation = segmentation[filter_mask]
#
#         if self.n_goal_points is not None:
#             assert (goal_pos := observation["extra"]["goal_pos"]) is not None
#             goal_pts_xyz = (
#                 np.random.uniform(low=-1.0, high=1.0, size=(self.n_goal_points, 3))
#                 * 0.01
#             ).astype(np.float32)
#             goal_pts_xyz = goal_pts_xyz + goal_pos[None, :]
#             goal_pts_rgb = np.zeros_like(goal_pts_xyz)
#             goal_pts_rgb[:, 1] = 1
#             point_cloud = np.concatenate([point_cloud, goal_pts_xyz])
#             rgb = np.concatenate([rgb, goal_pts_rgb])  # type: ignore
#
#         point_cloud = apply_pose_to_points(point_cloud, to_origin)
#
#         if self.use_color:
#             point_cloud = np.hstack([point_cloud, rgb])  # type: ignore
#
#         if segmentation is not None:
#             out = np.zeros((point_cloud.shape[0], 6), dtype=np.float32)
#             out[..., :3] = point_cloud
#             out[..., 3] = segmentation[..., 0] / 100.0
#             point_cloud = out
#
#         if self.voxel_grid_size is not None:
#             point_cloud = voxel_grid_sample(point_cloud, self.voxel_grid_size)
#
#         if self.normalize:
#             point_cloud = normalize(point_cloud)
#
#         return point_cloud

