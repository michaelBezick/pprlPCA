import copy
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator

import gymnasium as gym
import gymnasium.spaces as spaces
import hydra
import numpy as np
import parllel.logger as logger
import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from parllel import Array, ArrayDict, dict_map
from parllel.callbacks.recording_schedule import RecordingSchedule
from parllel.patterns import build_cages, build_sample_tree
from parllel.replays.replay import ReplayBuffer
from parllel.runners import RLRunner
from parllel.samplers import BasicSampler
from parllel.samplers.eval import EvalSampler, MultiEvalSampler
from parllel.torch.agents.sac_agent import SacAgent
from parllel.torch.algos.sac import build_replay_buffer_tree
from parllel.torch.distributions.squashed_gaussian import SquashedGaussian
from parllel.transforms.vectorized_video import RecordVectorizedVideo
from parllel.types import BatchSpec
from sapien.core import Pose
from scipy.spatial.transform import Rotation as R

from pprl.utils.array_dict import build_obs_array

# =========================
# Base camera constants (Z-up world; eval uses direct quaternion)
# =========================
WORLD_UP = (0.0, 0.0, 1.0)

BASE_VFOV_DEG = 90.0
RECORD_EVERY = False
RECORD_NONE = True


def orbit_eye_and_lookat_wxyz_around(
    eye0: np.ndarray,
    target: np.ndarray,
    *,
    yaw_deg: float = 0.0,  # orbit around WORLD_UP
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (eye, quat_wxyz) after orbiting eye0 about target and re-orienting to look at target.

    yaw_deg:   rotate the camera position around WORLD_UP (azimuth).
    pitch_deg: rotate the camera position around a right-axis (elevation).
    radius_delta: move closer/farther along the radial vector (target->eye).
    """
    eye0 = np.asarray(eye0, dtype=float)
    target = np.asarray(target, dtype=float)

    rad = np.deg2rad(yaw_deg)

    c, s = np.cos(rad), np.sin(rad)

    rot_mat = np.array([[c, -s], [s, c]], dtype=float)

    rel_xy = eye0[:2] - target[:2]
    new_xy = (rot_mat @ rel_xy.T).T + target[:2]

    new_pos = np.array((new_xy[0], new_xy[1], eye0[2]))

    new_orientation = look_at_wxyz(new_pos, target)

    return (new_pos, new_orientation)


def _rot_axis_angle(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix for rotating around `axis` by `deg`."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    x, y, z = axis

    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
    I = np.eye(3)
    return I + s * K + (1 - c) * (K @ K)


def orbit_up_down_eye_and_lookat_wxyz(
    eye0: np.ndarray,
    target: np.ndarray,
    *,
    pitch_deg: float = 0.0,  # +pitch moves camera upward (usually)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray]:
    eye0 = np.asarray(eye0, dtype=float)
    target = np.asarray(target, dtype=float)
    up_v = np.asarray(up, dtype=float)

    rel = eye0 - target  # target->eye
    right = np.cross(up_v, rel)
    n = np.linalg.norm(right)

    # Handle near-pole case (camera nearly above/below target)
    if n < 1e-8:
        # pick any axis perpendicular to rel
        tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(tmp, rel) / (np.linalg.norm(rel) + 1e-12)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(tmp, rel)
        n = np.linalg.norm(right)

    right = right / (n + 1e-12)

    R_right = _rot_axis_angle(right, pitch_deg)
    new_pos = target + R_right @ rel

    new_orientation = look_at_wxyz(new_pos, target)  # or include up_v if supported
    return new_pos, new_orientation


# =========================
# Quaternion / axes helpers (SAPIEN wxyz)
# =========================
def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


import numpy as np
from scipy.spatial.transform import Rotation as R


def apply_world_yaw_z_wxyz(q_wxyz: np.ndarray, deg: float) -> np.ndarray:
    """World yaw about +Z by 'deg' degrees: q_new = Rz(deg) * q."""
    r_base = R.from_quat(_wxyz_to_xyzw(q_wxyz))
    r_z = R.from_rotvec(np.deg2rad(deg) * np.array([0.0, 0.0, 1.0], dtype=float))
    q_new = (r_z * r_base).as_quat()
    return _xyzw_to_wxyz(q_new)


def _apply_local_axis_wxyz(
    q_base_wxyz: np.ndarray, axis_local: np.ndarray, deg: float
) -> np.ndarray:
    """
    Apply an INTRINSIC (local-frame) rotation about `axis_local` (given in camera coords).
    If q maps camera->world (SAPIEN Pose convention), then:
        R_new = R_base * R_local
    """
    if abs(deg) < 1e-12:
        return q_base_wxyz.copy()

    r_base = R.from_quat(_wxyz_to_xyzw(q_base_wxyz))

    axis_local = np.asarray(axis_local, dtype=float)
    axis_local = axis_local / (np.linalg.norm(axis_local) + 1e-12)

    r_local = R.from_rotvec(np.deg2rad(deg) * axis_local)

    q_new_xyzw = (r_base * r_local).as_quat()
    return _xyzw_to_wxyz(q_new_xyzw)


def roll_local_wxyz(q_base_wxyz: np.ndarray, roll_deg: float) -> np.ndarray:
    """
    Pure camera roll with ZERO look-direction drift (SAPIEN/ManiSkill2):
    roll about camera forward axis (+X).
    """
    if abs(roll_deg) < 1e-12:
        return q_base_wxyz.copy()

    r_base = R.from_quat(_wxyz_to_xyzw(q_base_wxyz))
    r_local = R.from_rotvec(
        np.deg2rad(roll_deg) * np.array([1.0, 0.0, 0.0])
    )  # +X forward
    q_new_xyzw = (r_base * r_local).as_quat()  # right-multiply = local/intrinsic
    return _xyzw_to_wxyz(q_new_xyzw)


def _basis_from_quat_wxyz(q_wxyz: np.ndarray):
    """
    Return (right, up, view_dir) unit vectors in WORLD frame from camera orientation.
    Assumes OpenGL-like camera convention where optical axis aligns with -Z of the camera frame.
    If R is the rotation (camera->world), then:
      camera x+ -> world right = R[:,0]
      camera y+ -> world up    = R[:,1]
      camera z+ -> world back  = R[:,2]
    So the viewing direction (world) is -R[:,2].
    """
    R_cam_world = R.from_quat(_wxyz_to_xyzw(q_wxyz)).as_matrix()
    right = R_cam_world[:, 0]
    up = R_cam_world[:, 1]
    view_dir = -R_cam_world[:, 2]  # looking direction
    # Normalize just in case of tiny numeric drift
    right /= np.linalg.norm(right) + 1e-12
    up /= np.linalg.norm(up) + 1e-12
    view_dir /= np.linalg.norm(view_dir) + 1e-12
    return right, up, view_dir


# Tiny util used in reset() debug print
def pick_img(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return None


# =========================
# Eval camera set (built from BASE_POS / BASE_QUAT_WXYZ)
# =========================


def _build_eval_cameras(
    base_pos,
    base_quat_wxyz,
    target: np.ndarray | None = None,
    *,
    orbit_deg: float = 20.0,
) -> Dict[str, dict]:
    """
    Build eval configs keyed by name.

    Perturbations are fixed-angle orbits around `target`:
      - left/right: yaw about WORLD_UP
      - up:         pitch about camera-right axis
    and we always re-orient with look_at so the target stays centered.
    """
    pos0 = np.asarray(base_pos, dtype=float).copy()
    q0 = np.asarray(base_quat_wxyz, dtype=float).copy()
    vfov0 = BASE_VFOV_DEG

    # If caller didn’t provide a target, pick one along the current view ray.
    if target is None:
        _, _, view = _basis_from_quat_wxyz(q0)
        focus_dist = 1.0  # meters; adjust per-env if needed
        target = pos0 + focus_dist * view
    else:
        target = np.asarray(target, dtype=float)

    perturb_string = f"{orbit_deg:.0f}deg"

    # NOTE on signs:
    # With WORLD_UP=(0,0,1) and right-hand rule, +yaw usually moves the camera "left"
    # around the target (as seen from the camera’s nominal view). If it feels flipped,
    # just swap the +/- on yaw below.
    eye_left, q_left = orbit_eye_and_lookat_wxyz_around(
        pos0,
        target,
        yaw_deg=+orbit_deg,
    )
    eye_right, q_right = orbit_eye_and_lookat_wxyz_around(
        pos0,
        target,
        yaw_deg=-orbit_deg,
    )
    eye_up, q_up = orbit_up_down_eye_and_lookat_wxyz(pos0, target, pitch_deg=-orbit_deg)

    cams = {
        "nominal": {
            "position": pos0.tolist(),
            "quat_wxyz": q0.tolist(),
            "vertical_field_of_view": vfov0,
        },
        f"orbit_up+{perturb_string}": {  # orbit UP
            "position": eye_up.tolist(),
            "quat_wxyz": q_up.tolist(),
            "vertical_field_of_view": vfov0,
        },
        f"orbit_left+{perturb_string}": {  # orbit LEFT
            "position": eye_left.tolist(),
            "quat_wxyz": q_left.tolist(),
            "vertical_field_of_view": vfov0,
        },
        f"orbit_right+{perturb_string}": {  # orbit RIGHT
            "position": eye_right.tolist(),
            "quat_wxyz": q_right.tolist(),
            "vertical_field_of_view": vfov0,
        },
        "roll+45deg": {
            "position": pos0.tolist(),
            "quat_wxyz": roll_local_wxyz(q0, +45.0).tolist(),
            "vertical_field_of_view": vfov0,
        },
    }

    return cams


def _clear_pointcloud_buffers(env):
    """Walk the wrapper chain and clear any MS2 pointcloud _buffer dicts."""
    seen = set()
    cur = env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if hasattr(cur, "_buffer") and isinstance(cur._buffer, dict):
            cur._buffer.clear()
        cur = getattr(cur, "env", None)
    u = getattr(env, "unwrapped", None)
    if hasattr(u, "_buffer") and isinstance(u._buffer, dict):
        u._buffer.clear()


from gymnasium.wrappers import TimeLimit


# =========================
# Camera wrapper factories
# =========================
class TrainCamWrapperFactory:
    def __init__(
        self,
        base_factory,
        camera_name,
        train_pos,
        train_quat_wxyz,
        max_episode_steps: int | None = None,
        domain_randomization: bool = False,
        pos_jitter_max=(0.0, 0.0, 0.0),
        rot_jitter_max_deg=(0.0, 0.0, 0.0),
    ):
        import numpy as _np

        self.base_factory = base_factory
        self.camera_name = camera_name
        self.train_pos = list(_np.asarray(train_pos, dtype=float))
        self.train_quat = list(_np.asarray(train_quat_wxyz, dtype=float))
        self.max_episode_steps = max_episode_steps

        self.domain_randomization = bool(domain_randomization)
        self.pos_jitter_max = _np.asarray(pos_jitter_max, dtype=float)
        self.rot_jitter_max_deg = _np.asarray(rot_jitter_max_deg, dtype=float)

    def __call__(self, *args, **kwargs):
        env = self.base_factory(*args, **kwargs)
        horizon = self.max_episode_steps or getattr(
            getattr(env, "spec", None), "max_episode_steps", 200
        )
        env = TimeLimit(env, max_episode_steps=int(horizon))

        pose = Pose(p=self.train_pos, q=self.train_quat)
        return FixedOrConfiguredCamera(
            env,
            camera_name=self.camera_name,
            mode="train",
            train_pose=pose,
            eval_cam_cfg=None,
            domain_randomization=self.domain_randomization,
            pos_jitter_max=self.pos_jitter_max,
            rot_jitter_max_deg=self.rot_jitter_max_deg,
        )


class EvalCamWrapperFactory:
    def __init__(
        self,
        base_factory,
        camera_name,
        cam_cfg: dict,
        max_episode_steps: int | None = None,
    ):
        import copy as _copy

        self.base_factory = base_factory
        self.camera_name = camera_name
        self.cam_cfg = _copy.deepcopy(cam_cfg)
        self.max_episode_steps = max_episode_steps

    def __call__(self, *args, **kwargs):
        import copy as _copy

        env = self.base_factory(*args, **kwargs)
        horizon = self.max_episode_steps or getattr(
            getattr(env, "spec", None), "max_episode_steps", 200
        )
        env = TimeLimit(env, max_episode_steps=int(horizon))
        return FixedOrConfiguredCamera(
            env,
            camera_name=self.camera_name,
            mode="eval",
            train_pose=None,
            eval_cam_cfg=_copy.deepcopy(self.cam_cfg),
        )


# =========================
# Camera object wrapper
# =========================
def _get_named_camera(env, name: str | None):
    u = env.unwrapped
    found_maps = []
    for attr in ("sensors", "_sensors", "cameras", "_cameras"):
        d = getattr(u, attr, None)
        if isinstance(d, dict) and len(d) > 0:
            found_maps.append((attr, list(d.keys()), d))
    if name is not None:
        for _, _, d in found_maps:
            if name in d:
                return d[name]
    if len(found_maps) == 1 and len(found_maps[0][1]) == 1:
        _, keys, d = found_maps[0]
        return d[keys[0]]
    details = (
        " ; ".join([f"{attr}={keys}" for attr, keys, _ in found_maps])
        or "no camera dicts found"
    )
    raise RuntimeError(f"Camera '{name}' not found. Available: {details}")


class FixedOrConfiguredCamera(gym.Wrapper):
    """
    Train: use a fixed Pose.
    Eval:  apply the given cam cfg (pose + vfov if supported).
    """

    def __init__(
        self,
        env,
        camera_name: str | None = None,
        mode: str = "train",
        train_pose: Pose | None = None,
        eval_cam_cfg: dict | None = None,
        domain_randomization: bool = False,
        pos_jitter_max: np.ndarray | tuple = (0.0, 0.0, 0.0),
        rot_jitter_max_deg: np.ndarray | tuple = (0.0, 0.0, 0.0),
    ):
        super().__init__(env)
        self.camera_name = camera_name
        self.mode = mode
        self.train_pose = train_pose
        self.eval_cam_cfg = copy.deepcopy(eval_cam_cfg) if eval_cam_cfg else None

        self.domain_randomization = bool(domain_randomization)
        self.pos_jitter_max = np.asarray(pos_jitter_max, dtype=float)
        self.rot_jitter_max_deg = np.asarray(rot_jitter_max_deg, dtype=float)
        self._rng = np.random.default_rng()

    def _ensure_fov(self, cam, vfov_deg=None):
        if vfov_deg is not None:
            vfov_rad = float(np.deg2rad(vfov_deg))
            targets = [getattr(cam, "camera", None), cam]
            for obj in targets:
                if obj is None:
                    continue
                if hasattr(obj, "set_fovy"):
                    # Try radians (SAPIEN convention)
                    try:
                        obj.set_fovy(vfov_rad)
                        return True
                    except Exception:
                        pass
                    # Fallback: try degrees just in case
                    try:
                        obj.set_fovy(float(vfov_deg))
                        return True
                    except Exception:
                        pass
            return False
        else:
            return True

    def _apply_pose(self, cam, pose: Pose):

        ent = getattr(cam, "camera", None) or getattr(cam, "_camera", None) or cam

        # 1) Detach from any parent (don't swallow failures)
        if hasattr(ent, "set_parent"):
            try:
                # SAPIEN signature: set_parent(parent, keep_pose)
                ent.set_parent(None, True)
            except TypeError:
                # some versions might allow set_parent(None)
                ent.set_parent(None)

        # 2) Use set_local_pose; in SAPIEN: if no parent, this is the GLOBAL pose
        if hasattr(ent, "set_local_pose"):
            ent.set_local_pose(pose)
            return

        # Fallbacks
        if hasattr(ent, "set_world_pose"):
            ent.set_world_pose(pose)
            return
        if hasattr(ent, "set_pose"):
            ent.set_pose(pose)
            return

        raise RuntimeError("Camera object lacks usable pose setters.")

    def reset(self, *, seed=None, options=None):

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, info = self.env.reset(seed=seed, options=options)
        cam = _get_named_camera(self.env, self.camera_name)

        u = self.env.unwrapped
        u._pc_cam_wrap = cam
        u._pc_cam_uid = getattr(cam, "uid", None) or (
            self.camera_name or "render_camera"
        )
        u._pc_cam_entity = getattr(cam, "camera", None) or getattr(cam, "_camera", None)

        for dict_name in ("_cameras", "cameras", "_sensors", "sensors"):
            d = getattr(u, dict_name, None)
            if isinstance(d, dict):
                d.clear()
                d[u._pc_cam_uid] = cam

        # self._ensure_depth_fov(cam, vfov_deg=BASE_VFOV_DEG, near=0.02, far=3.0)
        self._ensure_fov(cam, vfov_deg=BASE_VFOV_DEG)

        if self.mode == "train":
            if self.train_pose is not None:
                if self.domain_randomization:
                    # ---- position jitter (world frame) ----
                    shift = self._rng.uniform(
                        low=-self.pos_jitter_max, high=self.pos_jitter_max, size=3
                    )
                    new_pos = (
                        np.asarray(self.train_pose.p, dtype=float) + shift
                    ).tolist()

                    # ---- orientation jitter (extrinsic world Z/Y/X) ----
                    yaw_max, pitch_max, roll_max = self.rot_jitter_max_deg.tolist()
                    yaw = float(self._rng.uniform(-yaw_max, yaw_max))
                    pitch = float(self._rng.uniform(-pitch_max, pitch_max))
                    roll = float(self._rng.uniform(-roll_max, roll_max))

                    base_q = np.asarray(self.train_pose.q, dtype=float)
                    new_q = apply_world_euler_wxyz(base_q, yaw, pitch, roll).tolist()

                    pose = Pose(p=new_pos, q=new_q)
                    self._apply_pose(cam, pose)
                else:
                    self._apply_pose(cam, self.train_pose)
        else:
            if self.eval_cam_cfg is None:
                raise ValueError("eval_cam_cfg must be provided in eval mode.")
            pos = self.eval_cam_cfg["position"]
            vfov = float(self.eval_cam_cfg.get("vertical_field_of_view", BASE_VFOV_DEG))

            if "quat_wxyz" in self.eval_cam_cfg:
                quat = self.eval_cam_cfg["quat_wxyz"]
                pose = Pose(p=pos, q=quat)
            else:
                # Fallback (shouldn’t happen in this config)
                raise ValueError("no quat_wxyz in config")
                # pose = Pose(p=pos, q=BASE_QUAT_WXYZ.tolist())

            self._apply_pose(cam, pose)
            # self._ensure_depth_fov(cam, vfov_deg=vfov, near=0.02, far=3.0)
            self._ensure_fov(cam, vfov_deg=vfov)

        # Force a render/update once after pose change
        for obj in (cam, getattr(cam, "camera", None), getattr(cam, "_camera", None)):
            if obj is not None and hasattr(obj, "take_picture"):
                try:
                    obj.take_picture()
                    break
                except Exception:
                    pass
        scene = getattr(self.env.unwrapped, "scene", None) or getattr(
            self.env.unwrapped, "_scene", None
        )
        if scene is not None and hasattr(scene, "update_render"):
            try:
                scene.update_render()
            except Exception:
                pass

        # Quick sanity: count valid depth if available
        for attr in ("get_images", "get_image", "get_obs", "get_observation"):
            if hasattr(cam, attr):
                try:
                    imgs = getattr(cam, attr)()
                    depth = pick_img(imgs, ("Depth", "depth", "DEPTH"))
                    if depth is not None:
                        depth = np.asarray(depth)
                        valid = np.isfinite(depth) & (depth > 0)
                        print(
                            "Depth stats:",
                            int(valid.sum()),
                            float(np.nanmin(depth)),
                            float(np.nanmax(depth)),
                        )
                except Exception:
                    pass
                break

        _clear_pointcloud_buffers(self.env)

        return obs, info


import numpy as np
from scipy.spatial.transform import Rotation as R


def look_at_wxyz(eye, target, world_up=(0, 0, 1), *, eps=1e-6, left_hint_world=None):
    """
    SAPIEN camera convention:
      camera +X forward, +Y left, +Z up  (robotics)
    Returns quaternion in wxyz.
    """
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(world_up, float)

    f = target - eye
    fn = np.linalg.norm(f)
    if fn < eps:
        # degenerate
        q = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity-ish
        return q
    f = f / fn  # +X forward in world

    # left = up x forward  (matches x forward, y left, z up)
    left = np.cross(up, f)
    ln = np.linalg.norm(left)

    if ln < eps:
        # forward ~ parallel to up => singular; pick a stable left direction
        if left_hint_world is None:
            # choose something not parallel to f
            a = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, f)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            left = np.cross(a, f)
        else:
            left = np.asarray(left_hint_world, float)
            left = left - np.dot(left, f) * f  # project to plane ⟂ f
        left = left / (np.linalg.norm(left) + 1e-12)
    else:
        left = left / ln

    cam_up = np.cross(f, left)  # ensures right-handed basis
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-12)

    # Columns are camera axes in world: [forward(+X), left(+Y), up(+Z)]
    R_cam_world = np.stack([f, left, cam_up], axis=1)

    q_xyzw = R.from_matrix(R_cam_world).as_quat()  # xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # wxyz


# =========================
# RL Build
# =========================
@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel = config.parallel
    discount = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    storage = "shared" if parallel else "local"

    with open_dict(config.env):
        env_name = config.env.pop("name")
        traj_info = config.env.pop("traj_info")
        cam_name = config.env.pop("camera_name", None)
        dr_enabled = bool(config.env.pop("domain_randomization", False))
        pos_jit = tuple(config.env.pop("pos_jitter_max", (0.0, 0.0, 0.0)))
        rot_jit = tuple(config.env.pop("rot_jitter_max_deg", (0.0, 0.0, 0.0)))

    if env_name == "OpenCabinetDrawer":

        base_pos = np.array([-0.433352, 0.948292, 0.885752], dtype=float)

        base_pos += np.array([-1, 1, 0.3])

        target = [-0.09954548, 0.21631327, 0.6722093]

        base_quat_wxyz = look_at_wxyz(base_pos, target)

    elif env_name == "OpenCabinetDoor":

        base_pos = np.array([-0.433352, 0.948292, 0.885752], dtype=float)

        base_pos += np.array([-1, 1, 0.3])

        target = [-0.09954548, 0.21631327, 0.6722093]

        base_quat_wxyz = look_at_wxyz(base_pos, target)

    elif env_name == "TurnFaucet":

        base_pos = np.array([-0.433352, 0.948292, 0.885752], dtype=float)

        target = [-0.01467387, 0.0249525, 0.25527865]

        base_quat_wxyz = look_at_wxyz(base_pos, target)

    elif env_name == "PushChair":

        base_pos = np.array([2, 2, 2], dtype=float)

        target = np.array([0.0, 0.0, 0.5], dtype=float)

        base_quat_wxyz = look_at_wxyz(base_pos, target)

    else:
        raise ValueError("Invalid Env Name")

    EVAL_CAMERAS: Dict[str, dict] = _build_eval_cameras(
        base_pos, base_quat_wxyz, target=target
    )

    TrajInfoClass = get_class(traj_info)
    TrajInfoClass.set_discount(discount)

    # Base env factory (Hydra partial)
    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    # TRAIN cages: fixed camera pose (use base pos + base quat)
    train_factory = TrainCamWrapperFactory(
        base_factory=env_factory,
        camera_name=cam_name,
        train_pos=base_pos,
        train_quat_wxyz=base_quat_wxyz,
        max_episode_steps=config.env.max_episode_steps,
        domain_randomization=dr_enabled,
        pos_jitter_max=pos_jit,
        rot_jitter_max_deg=rot_jit,
    )

    cages, metadata = build_cages(
        EnvClass=train_factory,
        n_envs=batch_spec.B,
        env_kwargs={"add_rendering_to_info": True},
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )

    # Build sample tree & spaces
    replay_length = int(config.algo.replay_length) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T
    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
        keys_to_skip=("obs", "next_obs"),
    )
    obs_space, action_space = metadata.obs_space, metadata.action_space
    sample_tree["observation"] = build_obs_array(
        metadata.example_obs,
        obs_space,
        batch_shape=tuple(batch_spec),
        storage=storage,
        padding=1,
        full_size=replay_length,
    )
    sample_tree["next_observation"] = sample_tree["observation"].new_array(
        padding=0, inherit_full_size=True
    )

    assert isinstance(action_space, spaces.Box)
    n_actions = action_space.shape[0]

    # ===== Per-perturbation Eval Samplers (record ALL) =====
    callbacks = []
    eval_samplers: dict[str, EvalSampler] = {}

    eval_tree_keys = [
        "action",
        "agent_info",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "done",
    ]
    eval_tree_example = ArrayDict({k: sample_tree[k] for k in eval_tree_keys})

    for name, cam_cfg in EVAL_CAMERAS.items():
        eval_factory = EvalCamWrapperFactory(
            base_factory=env_factory,
            camera_name=cam_name,
            cam_cfg=cam_cfg,
            max_episode_steps=config.env.max_episode_steps,
        )
        cages_i, meta_i = build_cages(
            EnvClass=eval_factory,
            n_envs=config.eval.n_eval_envs,
            env_kwargs={"add_rendering_to_info": True},
            TrajInfoClass=TrajInfoClass,
            parallel=parallel,
        )
        eval_sample_tree = eval_tree_example.new_array(
            batch_shape=(1, config.eval.n_eval_envs)
        )
        eval_sample_tree["env_info"] = dict_map(
            Array.from_numpy,
            meta_i.example_info,
            batch_shape=(1, config.eval.n_eval_envs),
            storage=storage,
        )

        step_transforms = []

        # Record EVERY eval mismatch

        condition = True if RECORD_EVERY else (name == "nominal")
        if RECORD_NONE:
            condition = False

        if condition:
            recorder = RecordVectorizedVideo(
                sample_tree=eval_sample_tree,
                buffer_key_to_record="env_info.rendering",
                env_fps=24,
                output_dir=Path(config.video_path) / name,
                video_length=config.env.max_episode_steps,
                use_wandb=True,
            )
            step_transforms.append(recorder)
            callbacks.append(RecordingSchedule(recorder, trigger="on_eval"))

        eval_samplers[name] = EvalSampler(
            max_traj_length=config.env.max_episode_steps,
            max_trajectories=config.eval.max_trajectories,
            envs=cages_i,
            agent=None,  # set after agent exists
            sample_tree=eval_sample_tree,
            step_transforms=step_transforms,
        )
    multi_eval_sampler = MultiEvalSampler(eval_samplers)

    # =========================
    # Model / Agent / Algo
    # =========================
    model = torch.nn.ModuleDict()
    with open_dict(config.model):
        encoder_name = config.model.pop("name")

    if encoder_name != "Passthru":
        encoder = instantiate(config.model, _convert_="partial", obs_space=obs_space)
        model["encoder"] = encoder
        embedding_size = encoder.embed_dim
    else:
        embedding_size = spaces.flatdim(obs_space)

    model["pi"] = instantiate(
        config.pi_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        action_space=action_space,
        _convert_="partial",
    )
    model["q1"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )
    model["q2"] = instantiate(
        config.q_mlp_head,
        input_size=embedding_size,
        action_size=n_actions,
        _convert_="partial",
    )

    distribution = SquashedGaussian(dim=n_actions, scale=action_space.high[0])
    device_str = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device_str}, allow_val_change=True)
    device = torch.device(device_str)

    agent = SacAgent(
        model=model,
        distribution=distribution,
        device=device,
        learning_starts=config.algo.learning_starts,
    )

    # attach agent to each eval sampler
    for s in eval_samplers.values():
        s.agent = agent  # type: ignore[attr-defined]

    sampler = BasicSampler(
        batch_spec=batch_spec, envs=cages, agent=agent, sample_tree=sample_tree
    )

    # Replay buffer
    replay_buffer_tree = build_replay_buffer_tree(sample_tree)

    def batch_transform(tree: ArrayDict[Array]) -> ArrayDict[torch.Tensor]:
        tree = tree.to_ndarray()  # type: ignore
        tree = tree.apply(torch.from_numpy)
        return tree.to(device=device)

    replay_buffer = ReplayBuffer(
        tree=replay_buffer_tree,
        sampler_batch_spec=batch_spec,
        size_T=replay_length,
        replay_batch_size=config.algo.batch_size,
        newest_n_samples_invalid=0,
        oldest_n_samples_invalid=1,
        batch_transform=batch_transform,
    )

    # Optimizers
    with open_dict(config.optimizer):
        q_optim_conf = config.optimizer.pop("q", {}) or {}
        pi_optim_conf = config.optimizer.pop("pi", {}) or {}
        encoder_optim_conf = config.optimizer.pop("encoder", {}) or {}
    pi_optimizer = instantiate(
        config.optimizer, [{"params": agent.model["pi"].parameters(), **pi_optim_conf}]
    )
    q_optimizer = instantiate(
        config.optimizer,
        [
            {"params": agent.model["q1"].parameters(), **q_optim_conf},
            {"params": agent.model["q2"].parameters(), **q_optim_conf},
        ],
    )
    if "encoder" in agent.model:
        q_optimizer.add_param_group(
            {"params": agent.model["encoder"].parameters(), **encoder_optim_conf}
        )

    gamma = config.get("lr_scheduler_gamma")
    if gamma is not None:
        pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=gamma)
        q_scheduler = torch.optim.lr_scheduler.ExponentialLR(q_optimizer, gamma=gamma)
        lr_schedulers = [pi_scheduler, q_scheduler]
    else:
        lr_schedulers = None

    algorithm = instantiate(
        config.algo,
        _convert_="partial",
        batch_spec=batch_spec,
        agent=agent,
        replay_buffer=replay_buffer,
        q_optimizer=q_optimizer,
        pi_optimizer=pi_optimizer,
        learning_rate_schedulers=lr_schedulers,
        action_space=action_space,
    )

    # ---- TRAIN VIDEO RECORDER (optional) ----
    train_video_recorder = RecordVectorizedVideo(
        sample_tree=sample_tree,
        buffer_key_to_record="env_info.rendering",
        env_fps=50,
        output_dir=Path(config.video_path) / "train",
        video_length=config.env.max_episode_steps,
        use_wandb=True,
    )
    # callbacks.append(RecordingSchedule(train_video_recorder, trigger="on_log"))

    # Runner
    runner = RLRunner(
        sampler=sampler,
        agent=agent,
        algorithm=algorithm,
        batch_spec=batch_spec,
        eval_sampler=multi_eval_sampler,
        n_steps=config.runner.n_steps,
        log_interval_steps=config.runner.log_interval_steps,
        eval_interval_steps=config.runner.eval_interval_steps,
        callbacks=callbacks,
    )

    try:
        yield runner
    finally:
        multi_eval_sampler.close()
        sampler.close()
        sample_tree.close()
        agent.close()
        for cage in cages:
            cage.close()


import argparse
import sys


def _preparse_conf_dir():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--conf-dir",
        type=str,
        default=None,
        help="Directory that contains train_sac.yaml etc.",
    )
    p.add_argument("--config-name", "-cn", type=str, default="train_sac")
    args, rest = p.parse_known_args()
    injected = []
    if args.conf_dir:
        injected.append(f"-cp={args.conf_dir}")  # replace config_path
    if args.config_name:
        injected.append(f"-cn={args.config_name}")
    sys.argv = [sys.argv[0], *injected, *rest]


_preparse_conf_dir()


@hydra.main(version_base=None, config_path="../conf", config_name="train_sac")
def main(config: DictConfig) -> None:
    with open_dict(config):
        wandb_config = config.pop("wandb", {})
        notes = wandb_config.pop("notes", None)
        tags = wandb_config.pop("tags", None)
        group_name = wandb_config.pop("group_name", None)
        print("Group name is: ", group_name)

    run = wandb.init(
        project="pprl",
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,
        save_code=True,
        reinit=True,
        tags=tags,
        notes=notes,
        group=group_name,
    )

    logger.init(
        wandb_run=run,
        log_dir=Path(f"log_data/sac/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"),
        tensorboard=True,
        output_files={"txt": "log.txt"},  # type: ignore
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),  # type: ignore
        model_save_path=Path("model.pt"),
    )

    video_path = Path(config.video_path) / f"{datetime.now().strftime('%Y-%m-%d')}/{run.id}"  # type: ignore
    config.update({"video_path": video_path})

    with build(config) as runner:
        runner.run()

    logger.close()
    run.finish()  # type: ignore


if __name__ == "__main__":
    main()
