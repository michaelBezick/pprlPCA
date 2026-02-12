from __future__ import annotations

import numpy as np
from typing import Sequence

from mani_skill2 import format_path
from mani_skill2.envs.ms1.push_chair import PushChairEnv
from mani_skill2.sensors.camera import CameraConfig
from mani_skill2.utils.io_utils import load_json
from mani_skill2.utils.registration import register_env
from scipy.spatial import distance as sdist


TRIPOD_UIDS_DEFAULT = ("overhead_camera_0",)


def _as_cfg_list(x):
    """Normalize possible return types to list[CameraConfig]."""
    if x is None:
        return []
    if isinstance(x, dict):
        return list(x.values())
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


@register_env("PushChair-v2", max_episode_steps=200)
class ConfigurablePushChairEnv(PushChairEnv):
    """PushChair with modified rewards + configurable camera filtering."""

    def __init__(
        self,
        *args,
        n_models: int | None = None,
        tripod_uids: Sequence[str] | None = None,
        prune_render_cameras: bool = True,
        **kwargs,
    ):
        if n_models is not None:
            model_json = format_path(self.DEFAULT_MODEL_JSON)
            model_db: dict[str, dict] = load_json(model_json)
            kwargs["model_ids"] = list(model_db.keys())[:n_models]

        self.tripod_uids = tuple(tripod_uids) if tripod_uids is not None else TRIPOD_UIDS_DEFAULT
        self.prune_render_cameras = prune_render_cameras
        super().__init__(*args, **kwargs)

    def _register_cameras(self) -> list[CameraConfig]:
        # 1) start with whatever the env registers as observation cameras (can be empty in PushChair)
        cams = _as_cfg_list(super()._register_cameras())

        # 2) ALSO pull from render cameras if the overhead cams live there in this env
        if hasattr(self, "_register_render_cameras"):
            cams += _as_cfg_list(self._register_render_cameras())

        # 3) keep only tripod uids
        kept = [c for c in cams if getattr(c, "uid", None) in self.tripod_uids]
        if kept:
            return kept

        # 4) fallback: if nothing matched but there *are* cameras, keep the first
        if cams:
            return [cams[0]]

        # 5) last resort: allow no cameras (prevents IndexError)
        # If your obs_mode later requires images/pointcloud, it will fail elsewhere with a clearer error.
        return []

    def _configure_cameras(self) -> None:
        super()._configure_cameras()

        # Prune final camera registry to only the one you want
        if hasattr(self, "_camera_cfgs") and isinstance(self._camera_cfgs, dict):
            pruned = {uid: cfg for uid, cfg in self._camera_cfgs.items() if uid in self.tripod_uids}
            self._camera_cfgs = pruned or dict(list(self._camera_cfgs.items())[:1])

        if hasattr(self, "_cameras") and isinstance(self._cameras, dict):
            pruned = {uid: cam for uid, cam in self._cameras.items() if uid in self.tripod_uids}
            self._cameras = pruned or dict(list(self._cameras.items())[:1])

        # Optional: also prune render cameras so overhead_1/2 aren't created/kept
        if self.prune_render_cameras:
            if hasattr(self, "_render_camera_cfgs") and isinstance(self._render_camera_cfgs, dict):
                pruned = {uid: cfg for uid, cfg in self._render_camera_cfgs.items() if uid in self.tripod_uids}
                self._render_camera_cfgs = pruned or dict(list(self._render_camera_cfgs.items())[:1])
            if hasattr(self, "_render_cameras") and isinstance(self._render_cameras, dict):
                pruned = {uid: cam for uid, cam in self._render_cameras.items() if uid in self.tripod_uids}
                self._render_cameras = pruned or dict(list(self._render_cameras.items())[:1])

    # --------- your existing logic below ----------

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)
        self.prev_chair_pos = self.chair.pose.p[:2].astype(np.float32)
        self.prev_dist_chair_to_target = np.linalg.norm(self.prev_chair_pos - self.target_xy)
        return obs, info

    def _get_obs_extra(self):
        obs = super()._get_obs_extra()
        obs["target_link_pos"] = self.target_p[:3]
        return obs

    def evaluate(self, **kwargs):
        disp_chair_to_target = self.chair.pose.p[:2] - self.target_xy
        dist_chair_to_target = np.linalg.norm(disp_chair_to_target)

        z_axis_chair = self.root_link.pose.to_transformation_matrix()[:3, 2]
        chair_tilt = np.arccos(z_axis_chair[2])

        vel_norm = np.linalg.norm(self.root_link.velocity)
        ang_vel_norm = np.linalg.norm(self.root_link.angular_velocity)

        flags = dict(
            chair_close_to_target=dist_chair_to_target <= 0.2,
            chair_standing=chair_tilt <= 0.05 * np.pi,
            chair_static=self.check_actor_static(self.root_link, max_v=0.1, max_ang_v=0.2),
        )
        return dict(
            success=all(flags.values()),
            **flags,
            dist_chair_to_target=dist_chair_to_target,
            chair_tilt=chair_tilt,
            chair_vel_norm=vel_norm,
            chair_ang_vel_norm=ang_vel_norm,
        )

    def compute_dense_reward(self, action: np.ndarray, info: dict, **kwargs):
        reward = 0
        ee_coords = np.array(self.agent.get_ee_coords())  # [M, 3]
        chair_pcd = self._get_chair_pcd()  # [N, 3]

        dist_ees_to_chair = sdist.cdist(ee_coords, chair_pcd)  # [M, N]
        dist_ees_to_chair = dist_ees_to_chair.min(1)  # [M]
        dist_ee_to_chair = dist_ees_to_chair.mean()
        log_dist_ee_to_chair = np.log(dist_ee_to_chair + 1e-5)
        reward += -dist_ee_to_chair - np.clip(log_dist_ee_to_chair, -10, 0)

        chair_tilt = info["chair_tilt"]
        reward += -chair_tilt * 0.2

        action_norm = np.linalg.norm(action)
        reward -= action_norm * 1e-6

        chair_vel = self.root_link.velocity[:2]
        chair_vel_norm = np.linalg.norm(chair_vel)
        disp_chair_to_target = self.root_link.get_pose().p[:2] - self.target_xy
        cos_chair_vel_to_target = sdist.cosine(disp_chair_to_target, chair_vel)

        stage_reward = -10

        chair_pos = self.chair.pose.p[:2].astype(np.float32)
        delta_chair_pos = chair_pos - self.prev_chair_pos
        vec_prev_chair_pos_to_target = self.target_xy - self.prev_chair_pos
        dist_chair_to_target = info["dist_chair_to_target"]

        abs_cos_similarity = np.abs(np.dot(delta_chair_pos, vec_prev_chair_pos_to_target)) / (
            np.linalg.norm(delta_chair_pos) * np.linalg.norm(vec_prev_chair_pos_to_target) + 1e-8
        )

        delta_reward = (self.prev_dist_chair_to_target - dist_chair_to_target) * abs_cos_similarity * 100.0
        reward += delta_reward

        self.prev_chair_pos = chair_pos
        self.prev_dist_chair_to_target = dist_chair_to_target

        if chair_tilt < 0.2 * np.pi:
            if dist_ee_to_chair < 0.1:
                stage_reward += 2
                if dist_chair_to_target <= 0.2:
                    stage_reward += 2
                    reward += np.exp(-chair_vel_norm * 10) * 2

        reward = reward + stage_reward

        info.update(
            dist_ee_to_chair=dist_ee_to_chair,
            action_norm=action_norm,
            chair_vel_norm=chair_vel_norm,
            cos_chair_vel_to_target=cos_chair_vel_to_target,
            stage_reward=stage_reward,
        )
        return reward

