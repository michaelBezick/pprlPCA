from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import copy
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
import hydra
import torch
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose

import parllel.logger as logger
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

from pprl.utils.array_dict import build_obs_array

# ------------------------
# Camera math
# ------------------------

#WORLD_UP = (0.0, 1.0, 0.0)

'''
def _lookat_quat_zero_roll(position, look_at, world_up=(0.0, 1.0, 0.0)):
    pos = np.asarray(position, dtype=float)
    tgt = np.asarray(look_at, dtype=float)
    upw = np.asarray(world_up, dtype=float)

    fwd = tgt - pos
    fwd /= np.linalg.norm(fwd)

    # project world_up into plane orthogonal to forward (kills roll)
    up_proj = upw - fwd * np.dot(upw, fwd)
    up_proj /= np.linalg.norm(up_proj)

    right = np.cross(fwd, up_proj)
    right /= np.linalg.norm(right)

    # OpenGL camera: +x right, +y up, -z forward
    Rm = np.column_stack((right, up_proj, -fwd))

    q_xyzw = R.from_matrix(Rm).as_quat()      # (x,y,z,w)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

    
    #return Pose(p=pos.tolist(), q=q_wxyz.tolist())
    #return Pose(p=pos.tolist(), q=[0.364705, 0.279848, 0.115917, -0.880476])
    #return Pose(p=pos.tolist(), q=[0.702640218095606, 0.204639780540436, 0.223325355061539, -0.64385126195074])
    return Pose(p=pos.tolist(), q=[0.717801992001589, 0.142621934300541, 0.166360199707428, -0.660865300709779])

    #return Pose(p=pos.tolist(), q=[0.364705,0.279848,0.115917,-0.880476])

def _lookat_quat(position, look_at, world_up=(0.0, 1.0, 0.0), roll_deg=0.0):
    pos = np.asarray(position, dtype=float)
    target = np.asarray(look_at, dtype=float)
    up_world = np.asarray(world_up, dtype=float)

    forward = target - pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up_cam = np.cross(right, forward)

    theta = np.deg2rad(roll_deg)
    up_rot = up_cam * np.cos(theta) + right * np.sin(theta)
    right_rot = np.cross(forward, up_rot)

    rot_mtx = np.column_stack((right_rot, up_rot, -forward))
    q_xyzw = R.from_matrix(rot_mtx).as_quat()   # (x, y, z, w)
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
    return Pose(p=pos.tolist(), q=q_wxyz.tolist())
'''

def get_pointcloud_camera(env):
    u = env.unwrapped
    if hasattr(u, "_pc_cam_wrap") and u._pc_cam_wrap is not None:
        return u._pc_cam_wrap
    # fallback to your old heuristic
    return _get_named_camera(env, getattr(u, "_pc_cam_uid", None) or "render_camera")
def _hwc_u8(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)   # CHW -> HWC
    if img.dtype != np.uint8:
        mx = np.nanmax(img)
        if not np.isfinite(mx) or mx == 0: 
            return None
        if mx <= 1.0 + 1e-6:
            img = np.clip(img, 0, 1) * 255.0
        img = img.astype(np.uint8)
    return img

import os, math
import numpy as np

'''
def spherical_forward(yaw_deg: float, pitch_deg: float):
    # yaw: rotate around +Z; pitch: tilt up(+)/down(-)
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    # OpenGL-style: +x right, +y forward? We want +z up world.
    # Forward vector pointing FROM camera TO target:
    fwd = np.array([cp*cy, cp*sy, sp], dtype=float)  # z-up convention
    fwd /= np.linalg.norm(fwd)
    return fwd

def pose_from_spherical(center, radius, yaw_deg, pitch_deg):
    center = np.asarray(center, dtype=float)
    fwd = spherical_forward(yaw_deg, pitch_deg)
    cam_pos = center - radius * fwd          # place camera behind along forward
    #pose = _lookat_quat(cam_pos, center, world_up=(0,0,1), roll_deg=0.0)  # uses wxyz
    pose = _lookat_quat_zero_roll(cam_pos, center, world_up=(0,1,0))  # uses wxyz
    return cam_pos, pose
'''

# ===== Training camera (fixed) =====

BASE_VFOV_DEG = 90.0

CAM_CANON = {
    "position": [-0.433352, 0.948292, 0.885752],
    "orientation": [0.717801992001589, 0.142621934300541, 0.166360199707428, -0.660865300709779],
    "roll_deg": 0.0,
    "vertical_field_of_view": BASE_VFOV_DEG,  # keep your 62.0 unless you want to change it
}

# Train reads from CAM_CANON:
TRAIN_POS  = np.array(CAM_CANON["position"])
TRAIN_LOOK = np.array(CAM_CANON["lookAt"])

'''
# Eval dictionary only contains "nominal" and matches train exactly
EVAL_CAMERAS = {"nominal": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in CAM_CANON.items()}}
'''

# ===== Eval perturbations (SOFA-style) =====
BASE_POS  = TRAIN_POS.copy()
BASE_LOOK = TRAIN_LOOK.copy()
FORWARD   = (BASE_LOOK - BASE_POS) / np.linalg.norm(BASE_LOOK - BASE_POS)
delta_m  = 50.0 * 1e-3


translations = {
 "shift+x+50": {"position": (BASE_POS + np.array([+delta_m, 0, 0])).tolist(), "lookAt": BASE_LOOK.tolist()},
 "shift+y+50": {"position": (BASE_POS + np.array([0, +delta_m, 0])).tolist(), "lookAt": BASE_LOOK.tolist()},
 "shift+z+50": {"position": (BASE_POS + np.array([0, 0, +delta_m])).tolist(), "lookAt": (BASE_LOOK + np.array([0, 0, +delta_mm])).tolist()},
}
fwd_back = {
 "along_view+50": {
     "position": (BASE_POS + delta_mm * FORWARD).tolist(),
     "lookAt":   (BASE_LOOK + delta_mm * FORWARD).tolist(),
 }
}
rolls = {
 "roll+15deg": {"position": BASE_POS.tolist(), "lookAt": BASE_LOOK.tolist(), "roll_deg": +15.0},
 "roll+30deg": {"position": BASE_POS.tolist(), "lookAt": BASE_LOOK.tolist(), "roll_deg": +30.0},
}
handcrafted = {"nominal": {"position": BASE_POS.tolist(), "lookAt": BASE_LOOK.tolist()}}
fovs = {
 "fov30": {"position": BASE_POS.tolist(), "lookAt": BASE_LOOK.tolist(), "vertical_field_of_view": 30.0},
 "fov70": {"position": BASE_POS.tolist(), "lookAt": BASE_LOOK.tolist(), "vertical_field_of_view": 70.0},
}
EVAL_CAMERAS: dict[str, dict] = {**handcrafted, **translations, **fwd_back, **rolls, **fovs}

#--- Force the env to register only a single tripod camera BEFORE sensors/obs are built
def _force_single_camera_registration(env, keep_uid_candidates: tuple[str, ...]) -> None:
    u = env.unwrapped
    orig = getattr(u, "_register_cameras", None)
    if orig is None or getattr(u, "_register_cameras_wrapped", False):
        return  # nothing to do or already wrapped

    def choose_one(cams):
        try:
            seq = list(cams) if isinstance(cams, (list, tuple)) else [cams]
        except TypeError:
            return cams
        # prefer an exact UID match in the candidate list
        uid_pairs = [(getattr(c, "uid", None), c) for c in seq]
        for cand in keep_uid_candidates:
            for uid, cfg in uid_pairs:
                if uid == cand:
                    return [cfg]
        # prefer common tripod names
        for common in ("render_camera", "render", "tripod", "scene", "viewer"):
            for uid, cfg in uid_pairs:
                if uid == common:
                    return [cfg]
        # fallback: keep the first
        return [uid_pairs[0][1]] if uid_pairs else seq

    def wrapped(*a, **kw):
        cams = orig(*a, **kw)
        return choose_one(cams)

    import types
    u._register_cameras = types.MethodType(wrapped, u)
    u._register_cameras_wrapped = True
from gymnasium.wrappers import TimeLimit

class TrainCamWrapperFactory:
    def __init__(self, base_factory, camera_name, train_pos, train_look, roll_deg=0.0, max_episode_steps: int | None = None):
        import numpy as _np
        self.base_factory = base_factory
        self.camera_name  = camera_name
        self.train_pos    = list(_np.asarray(train_pos, dtype=float))
        self.train_look   = list(_np.asarray(train_look, dtype=float))
        self.roll_deg     = float(roll_deg)
        self.max_episode_steps = max_episode_steps

    def __call__(self, *args, **kwargs):
        env  = self.base_factory(*args, **kwargs)
        # 1) ensure a single tripod is registered
        keep = tuple(x for x in (self.camera_name, "render_camera", "render", "tripod", "scene") if x)
        _force_single_camera_registration(env, keep_uid_candidates=keep)
        # 2) enforce episode horizon (so completed_trajs is non-empty)
        horizon = self.max_episode_steps or getattr(getattr(env, "spec", None), "max_episode_steps", 200)
        env = TimeLimit(env, max_episode_steps=int(horizon))
        # 3) pose the tripod
        #pose = _lookat_quat(self.train_pos, self.train_look, roll_deg=self.roll_deg)
        pose = _lookat_quat_zero_roll(self.train_pos, self.train_look)
        return FixedOrConfiguredCamera(env, camera_name=self.camera_name, mode="train", train_pose=pose, eval_cam_cfg=None)


class EvalCamWrapperFactory:
    def __init__(self, base_factory, camera_name, cam_cfg: dict, max_episode_steps: int | None = None):
        import copy as _copy
        self.base_factory = base_factory
        self.camera_name  = camera_name
        self.cam_cfg      = _copy.deepcopy(cam_cfg)
        self.max_episode_steps = max_episode_steps

    def __call__(self, *args, **kwargs):
        import copy as _copy
        env = self.base_factory(*args, **kwargs)
        # keep only the tripod for EVAL too
        keep = tuple(x for x in (self.camera_name, "render_camera", "render", "tripod", "scene") if x)
        _force_single_camera_registration(env, keep_uid_candidates=keep)
        horizon = self.max_episode_steps or getattr(getattr(env, "spec", None), "max_episode_steps", 200)
        env = TimeLimit(env, max_episode_steps=int(horizon))
        return FixedOrConfiguredCamera(env, camera_name=self.camera_name, mode="eval", train_pose=None, eval_cam_cfg=_copy.deepcopy(self.cam_cfg))

# =========================
# Camera wrapper
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
    details = " ; ".join([f"{attr}={keys}" for attr, keys, _ in found_maps]) or "no camera dicts found"
    raise RuntimeError(f"Camera '{name}' not found. Available: {details}")

class FixedOrConfiguredCamera(gym.Wrapper):
    """
    Train: use a fixed Pose.
    Eval:  apply the given cam cfg (pose + vfov if supported).
    """
    def __init__(self, env, camera_name: str | None = None, mode: str = "train",
                 train_pose: Pose | None = None, eval_cam_cfg: dict | None = None):
        super().__init__(env)
        self.camera_name = camera_name
        self.mode = mode
        self.train_pose = train_pose
        self.eval_cam_cfg = copy.deepcopy(eval_cam_cfg) if eval_cam_cfg else None

    def _ensure_depth_fov(self, cam, vfov_deg=None, near=0.02, far=3.0):
        # FOV first (if provided)
        if vfov_deg is not None:
            for key in ("set_fov_y", "set_fov", "set_vertical_field_of_view",
                        "set_vertical_fov", "set_parameters"):
                for obj in (cam, getattr(cam, "camera", None)):
                    if obj is not None and hasattr(obj, key):
                        try: getattr(obj, key)(float(vfov_deg)); break
                        except Exception: pass
        # Then depth range
        for obj in (cam, getattr(cam, "camera", None)):
            if obj is None: continue
            for name, args in (("set_depth_range", (near, far)),
                               ("set_near", (near,)), ("set_far", (far,))):
                if hasattr(obj, name):
                    try: getattr(obj, name)(*args)
                    except Exception: pass


    def _ensure_depth_range(self, cam, near=0.02, far=3.0):
        # try common APIs on wrapper and underlying component
        targets = [cam, getattr(cam, "camera", None)]
        for t in targets:
            if t is None: 
                continue
            for name, args in (
                ("set_near", (near,)),
                ("set_far",  (far,)),
                ("set_depth_range", (near, far)),
                ("set_parameters", (near, far)),   # some wrappers overload this
            ):
                if hasattr(t, name):
                    try:
                        getattr(t, name)(*args)
                    except Exception:
                        pass
    def _apply_pose(self, cam, pose: Pose):
        # 1) prefer SAPIEN entity
        ent = getattr(cam, "camera", None) or getattr(cam, "_camera", None)

        # (a) detach from parent so nothing re-tilts the camera
        for obj in (ent, cam):
            if obj is not None and hasattr(obj, "set_parent"):
                try:
                    obj.set_parent(None)
                except Exception:
                    pass

        # (b) set WORLD pose first (avoid local pose unless last resort)
        def try_world(obj) -> bool:
            if obj is None:
                return False
            for name in ("set_pose", "set_world_pose", "set_model_matrix"):
                if hasattr(obj, name):
                    try:
                        getattr(obj, name)(pose)
                        return True
                    except Exception:
                        pass
            return False

        if try_world(ent) or try_world(cam):
            return

        # (c) fallbacks: matrices or local pose
        M = pose.to_transformation_matrix()
        def try_matrix(obj) -> bool:
            if obj is None:
                return False
            for name in ("set_extrinsic_cv", "set_extrinsic", "set_cam2world"):
                if hasattr(obj, name):
                    try:
                        getattr(obj, name)(M)
                        return True
                    except Exception:
                        pass
            return False

        if try_matrix(ent) or try_matrix(cam):
            return

        for obj in (ent, cam):
            if obj is not None and hasattr(obj, "set_local_pose"):
                try:
                    obj.set_local_pose(pose)
                    return
                except Exception:
                    pass

        raise RuntimeError("Camera object lacks usable pose setters.")

    def _apply_vfov_if_supported(self, cam, vfov_deg: float):
        for key in ("set_fov_y", "set_fov", "set_vertical_field_of_view", "set_vertical_fov", "set_parameters"):
            if hasattr(cam, key):
                try:
                    getattr(cam, key)(vfov_deg); return
                except Exception:
                    pass
        sub = getattr(cam, "camera", None)
        if sub is not None:
            for key in ("set_fov_y", "set_fov", "set_vertical_field_of_view", "set_vertical_fov", "set_parameters"):
                if hasattr(sub, key):
                    try:
                        getattr(sub, key)(vfov_deg); return
                    except Exception:
                        pass

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        cam = _get_named_camera(self.env, self.camera_name)


        u = self.env.unwrapped
        # keep references so *everyone* can find the same camera
        u._pc_cam_wrap  = cam
        u._pc_cam_uid   = getattr(cam, "uid", None) or (self.camera_name or "render_camera")
        u._pc_cam_entity = getattr(cam, "camera", None) or getattr(cam, "_camera", None)

        # prune camera dicts to just this one (prevents “viewer” camera confusion)
        for dict_name in ("_cameras", "cameras", "_sensors", "sensors"):
            d = getattr(u, dict_name, None)
            if isinstance(d, dict):
                d.clear()
                d[u._pc_cam_uid] = cam

        self._ensure_depth_range(cam, near=0.02, far=3.0)

        if self.mode == "train":
            if self.train_pose is not None:
                self._apply_pose(cam, self.train_pose)

            self._ensure_depth_fov(cam, vfov_deg=BASE_VFOV_DEG, near=0.02, far=3.0)
        else:
            if self.eval_cam_cfg is None:
                raise ValueError("eval_cam_cfg must be provided in eval mode.")
            pos  = self.eval_cam_cfg["position"]
            look = self.eval_cam_cfg["lookAt"]
            roll = float(self.eval_cam_cfg.get("roll_deg", 0.0))
            #pose = _lookat_quat(pos, look, roll_deg=roll)
            pose = _lookat_quat_zero_roll(pos, look)
            self._apply_pose(cam, pose)
            self._ensure_depth_fov(
                cam,
                vfov_deg=self.eval_cam_cfg.get("vertical_field_of_view", BASE_VFOV_DEG),
                near=0.02, far=3.0
            )
            vfov = self.eval_cam_cfg.get("vertical_field_of_view", None)
            if vfov is not None:
                self._apply_vfov_if_supported(cam, float(vfov))

                # force a render/update once after pose change
        for obj in (cam, getattr(cam, "camera", None), getattr(cam, "_camera", None)):
            if obj is not None and hasattr(obj, "take_picture"):
                try:
                    obj.take_picture()
                    break
                except Exception:
                    pass
        scene = getattr(self.env.unwrapped, "scene", None) or getattr(self.env.unwrapped, "_scene", None)
        if scene is not None and hasattr(scene, "update_render"):
            try:
                scene.update_render()
            except Exception:
                pass


                # Quick sanity: count valid depth
        for attr in ("get_images", "get_image", "get_obs", "get_observation"):
            if hasattr(cam, attr):
                try:
                    imgs = getattr(cam, attr)()
                    depth = pick_img(imgs, ("Depth", "depth", "DEPTH"))
                    if depth is not None:
                        depth = np.asarray(depth)
                        valid = np.isfinite(depth) & (depth > 0)
                        print("Depth stats:", int(valid.sum()), float(np.nanmin(depth)), float(np.nanmax(depth)))

                except Exception:
                    pass
                break

        print(self.env.unwrapped._cameras['render_camera'].camera.get_pose())
        return obs, info

# =========================
# RL Build
# =========================
@contextmanager
def build(config: DictConfig) -> Iterator[RLRunner]:
    parallel   = config.parallel
    discount   = config.algo.discount
    batch_spec = BatchSpec(config.batch_T, config.batch_B)
    storage    = "shared" if parallel else "local"

    with open_dict(config.env):
        config.env.pop("name")
        traj_info = config.env.pop("traj_info")
        cam_name  = config.env.pop("camera_name", None)  # don't pass into env ctor

    TrajInfoClass = get_class(traj_info)
    TrajInfoClass.set_discount(discount)

    # Base env factory (Hydra partial)
    env_factory = instantiate(config.env, _convert_="partial", _partial_=True)

    # TRAIN cages: fixed camera pose
    train_factory = TrainCamWrapperFactory(
        base_factory=env_factory,
        camera_name=cam_name,     # ex: "overhead_camera_0" or None to auto-pick if unique
        train_pos=TRAIN_POS,
        train_look=TRAIN_LOOK,
        roll_deg=0.0,
        max_episode_steps=config.env.max_episode_steps
    )

    cages, metadata = build_cages(
        EnvClass=train_factory,
        n_envs=batch_spec.B,
        env_kwargs={"add_rendering_to_info": True},   # <- so train videos record
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
        metadata.example_obs, obs_space,
        batch_shape=tuple(batch_spec), storage=storage, padding=1, full_size=replay_length,
    )
    sample_tree["next_observation"] = sample_tree["observation"].new_array(padding=0, inherit_full_size=True)

    # sample_tree["env_info"] = dict_map(
    #     Array.from_numpy, metadata.example_info,
    #     batch_shape=tuple(batch_spec), storage=storage,
    # )

    example_info = dict(metadata.example_info)
    import numpy as np
    example_info.setdefault("success", np.array(False, dtype=np.bool_))
    sample_tree["env_info"] = dict_map(
        Array.from_numpy, example_info,
        batch_shape=tuple(batch_spec), storage=storage,
    )

    assert isinstance(action_space, spaces.Box)
    n_actions = action_space.shape[0]

    # ===== Per-perturbation Eval Samplers =====
    callbacks = []
    eval_samplers: dict[str, EvalSampler] = {}

    eval_tree_keys = ["action","agent_info","observation","reward","terminated","truncated","done"]
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
        eval_sample_tree = eval_tree_example.new_array(batch_shape=(1, config.eval.n_eval_envs))
        eval_sample_tree["env_info"] = dict_map(
            Array.from_numpy, meta_i.example_info,
            batch_shape=(1, config.eval.n_eval_envs), storage=storage,
        )

        step_transforms = []
        #if name == "nominal":  # record only one
        recorder = RecordVectorizedVideo(
            sample_tree=eval_sample_tree,
            buffer_key_to_record="env_info.rendering",
            env_fps=50,
            output_dir=Path(config.video_path) / name,
            video_length=config.env.max_episode_steps,
            use_wandb=True,
        )
        step_transforms.append(recorder)
        callbacks.append(RecordingSchedule(recorder, trigger="on_eval"))

        eval_samplers[name] = EvalSampler(
            max_traj_length   = config.env.max_episode_steps,
            max_trajectories  = config.eval.max_trajectories,
            envs              = cages_i,
            agent             = None,  # set after agent exists
            sample_tree       = eval_sample_tree,
            step_transforms   = step_transforms,
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

    model["pi"] = instantiate(config.pi_mlp_head, input_size=embedding_size, action_size=n_actions, action_space=action_space, _convert_="partial")
    model["q1"] = instantiate(config.q_mlp_head,  input_size=embedding_size, action_size=n_actions, _convert_="partial")
    model["q2"] = instantiate(config.q_mlp_head,  input_size=embedding_size, action_size=n_actions, _convert_="partial")

    distribution = SquashedGaussian(dim=n_actions, scale=action_space.high[0])
    device_str = config.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.config.update({"device": device_str}, allow_val_change=True)
    device = torch.device(device_str)

    agent = SacAgent(model=model, distribution=distribution, device=device, learning_starts=config.algo.learning_starts)

    # attach agent to each eval sampler
    for s in eval_samplers.values():
        s.agent = agent  # type: ignore[attr-defined]

    sampler = BasicSampler(batch_spec=batch_spec, envs=cages, agent=agent, sample_tree=sample_tree)

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
    pi_optimizer = instantiate(config.optimizer, [{"params": agent.model["pi"].parameters(), **pi_optim_conf}])
    q_optimizer  = instantiate(
        config.optimizer,
        [
            {"params": agent.model["q1"].parameters(), **q_optim_conf},
            {"params": agent.model["q2"].parameters(), **q_optim_conf},
        ],
    )
    if "encoder" in agent.model:
        q_optimizer.add_param_group({"params": agent.model["encoder"].parameters(), **encoder_optim_conf})

    # LR schedulers (optional) – fixed walrus bug
    gamma = config.get("lr_scheduler_gamma")
    if gamma is not None:
        pi_scheduler = torch.optim.lr_scheduler.ExponentialLR(pi_optimizer, gamma=gamma)
        q_scheduler  = torch.optim.lr_scheduler.ExponentialLR(q_optimizer,  gamma=gamma)
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

    # ---- TRAIN VIDEO RECORDER (rollout) ----
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

