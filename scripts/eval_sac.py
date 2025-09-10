#!/usr/bin/env python3
# eval_camera_sweep.py
#
#   python eval_camera_sweep.py \
#          --ckpt  checkpoints/ppt_thread_seed0.pt \
#          --episodes 25
#
# Prints a table like
# ┌──────────────────────┬────────┬───────────┐
# │ camera_id            │ meanR  │ success % │
# ├──────────────────────┼────────┼───────────┤
# │ baseline_shift_z+50  │ 68.3   │  80.0     │
# │ shift_left_x-50      │ 66.0   │  72.0     │
# │ …                    │ …      │   …       │
# └──────────────────────┴────────┴───────────┘

from __future__ import annotations
import argparse, copy, pprint
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, open_dict, OmegaConf
import hydra
from hydra.utils import instantiate, get_class
import gymnasium as gym

# ─────────────────────────────────────────────────────────────────────────────
#  Import utilities you already copied from train_sac.py
#   - build_cages, PCObs, build_obs_array, etc.
#   - build_agent_and_algo()  (the final version with PPT + MLP args)
#   - load_checkpoint()       (the version that accepts --cfg fallback)
#   - make_camera_setups()    (shift / roll perturbations)
#   - EvalRunner              (the previous answer)
# ─────────────────────────────────────────────────────────────────────────────
from parllel.patterns import build_cages, build_sample_tree
from parllel.samplers.eval import EvalSampler
from parllel.types import BatchSpec
from parllel.runners import EvalRunner
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from pprl.models.ppt import PointPatchTransformer
from pprl.models.sac.q_and_pi_heads import PiMlpHead, QMlpHead
from pprl.envs import PointCloudSpace

def get_action_space(config):

    with open_dict(config.env.value):
        config.env.value.pop("name")
        traj_info = config.env.value.pop("traj_info")

    base_train_factory = instantiate(config.env.value, _convert_="partial", _partial_=True)
    parallel = config.parallel.value
    batch_spec = BatchSpec(config.batch_T.value, config.batch_B.value)


    TrajInfoClass = get_class(traj_info)

    cages, metadata = build_cages(
        # EnvClass=functools.partial(build_train_env, base_train_factory),
        EnvClass=base_train_factory,
        n_envs=batch_spec.B,
        TrajInfoClass=TrajInfoClass,
        parallel=parallel,
    )


    replay_length = int(config.algo.value.replay_length) // batch_spec.B
    replay_length = (replay_length // batch_spec.T) * batch_spec.T

    sample_tree, metadata = build_sample_tree(
        env_metadata=metadata,
        batch_spec=batch_spec,
        parallel=parallel,
        full_size=replay_length,
        keys_to_skip=("obs", "next_obs"),
    )

    obs_space, action_space = metadata.obs_space, metadata.action_space

    return action_space




def build_base_env_factory(cfg_env):
    cfg_env = copy.deepcopy(cfg_env.value)
    with open_dict(cfg_env):
        cfg_env.pop("name", None)
        cfg_env.pop("traj_info", None)
    return instantiate(cfg_env, _convert_="partial", _partial_=True)

def _infer_action_dim_from_state_dict(cfg):
    # Clean out invalid env keys
    base_env_factory = build_base_env_factory(cfg.env)

    # Supply the minimum valid arguments
    env = base_env_factory(
        mode=cfg.env.value.mode,
        render_mode=cfg.env.value.render_mode,
        create_scene_kwargs=cfg.env.value.create_scene_kwargs,
    )

    action_space = env.action_space
    if isinstance(action_space, gym.spaces.Box):
        return action_space.shape[0]
    else:
        raise RuntimeError("Unsupported action space type.")



def _shift(pos: np.ndarray, dx=0, dy=0, dz=0):
    return (pos + np.array([dx, dy, dz], dtype=float))

def _roll_around_optical_axis(pos: np.ndarray, lookat: np.ndarray, degrees: float):
    v = lookat - pos
    v = v / np.linalg.norm(v)
    up = np.array([0., 0., 1.])
    theta = np.radians(degrees)
    up_rot = (up * np.cos(theta) +
              np.cross(v, up) * np.sin(theta) +
              v * np.dot(v, up) * (1 - np.cos(theta)))
    return {"position": pos.tolist(), "lookAt": lookat.tolist(), "up": up_rot.tolist()}

import copy
def _pylist_of_dicts(seq):
    return [copy.deepcopy(d) for d in seq]




CM_PER_SHIFT = 50       # 50 cm  (⇒ 0.5 m)   – adjust if you use mm
BASE_POSITION = np.array([0., 175., 120.])
BASE_LOOKAT   = np.array([10.,   0.,  55.])

def make_camera_setups() -> dict[str, list[dict]]:
    """
    Returns a dict {camera_id : [ {position, lookAt}, … ]} that the Hydra
    config will accept as `env.cameras`.
    Each value is a **list** because RLBench supports multi-view inputs.  We
    perturb only the *first* camera; copy it for extra views if needed.
    """
    cams: dict[str, list[dict]] = {}

    # Baseline mismatch supplied by the user  (+50 cm on Z + lookAt)
    cams["baseline_shift_z+50"] = [
        {"position": [0, 175, 170], "lookAt": [10, 0, 105]}
    ]

    # L/R, U/D translations
    cams["shift_left_x-50"]  = [
        {"position": _shift(BASE_POSITION, dx=-CM_PER_SHIFT).tolist(),
         "lookAt":   BASE_LOOKAT.tolist()}
    ]
    cams["shift_right_x+50"] = [
        {"position": _shift(BASE_POSITION, dx=+CM_PER_SHIFT).tolist(),
         "lookAt":   BASE_LOOKAT.tolist()}
    ]
    cams["shift_up_z+50"] = [
        {"position": _shift(BASE_POSITION, dz=+CM_PER_SHIFT).tolist(),
         "lookAt":   BASE_LOOKAT.tolist()}
    ]
    cams["shift_down_z-50"] = [
        {"position": _shift(BASE_POSITION, dz=-CM_PER_SHIFT).tolist(),
         "lookAt":   BASE_LOOKAT.tolist()}
    ]

    # Roll (optical-axis rotation)
    cams["roll_90deg"]  = [_roll_around_optical_axis(BASE_POSITION,
                                                    BASE_LOOKAT,  90)]
    cams["roll_180deg"] = [_roll_around_optical_axis(BASE_POSITION,
                                                    BASE_LOOKAT, 180)]
    return cams

def build_agent_and_algo(cfg, device, action_dim):
    model_cfg = cfg["model"].value  # unwrap OmegaConf
    obs_space = PointCloudSpace(
        max_expected_num_points=cfg.env.value.max_expected_num_points,
        low=-1000,
        high=1000,
        feature_shape=(6,),  # match observation_type = pointcloud
    )
        # Instantiate PointGPT directly
    from pprl.models.pointgpt_rl import PointGPT
    model = PointGPT(
        obs_space=obs_space,
        embed_dim=model_cfg["embed_dim"],
        state_embed_dim=model_cfg["state_embed_dim"],
        tokenizer=hydra.utils.instantiate(model_cfg["tokenizer"]),
        pos_embedder=hydra.utils.instantiate(model_cfg["pos_embedder"]),
        transformer_encoder=hydra.utils.instantiate(model_cfg["transformer_encoder"]),
        gpt_encoder=hydra.utils.instantiate(model_cfg["gpt_encoder"]),
        gpt_decoder=hydra.utils.instantiate(model_cfg["gpt_decoder"]),
    ).to(device)

    # Build heads
    pi_head = hydra.utils.instantiate(cfg.pi_mlp_head.value,
                                      input_size=model_cfg["embed_dim"],
                                      action_size=action_dim,
                                      action_space=get_action_space(cfg)).to(device)

    q1_head = hydra.utils.instantiate(cfg.q_mlp_head.value,
                                      input_size=model_cfg["embed_dim"],
                                      action_size=action_dim).to(device)
    q2_head = hydra.utils.instantiate(cfg.q_mlp_head.value,
                                      input_size=model_cfg["embed_dim"],
                                      action_size=action_dim).to(device)

    agent = torch.nn.ModuleDict({
        "encoder": model,
        "pi": pi_head,
        "q1": q1_head,
        "q2": q2_head,
    }).to(device)

    class EvalSAC:
        def __init__(self, mod): self.mod = mod
        @torch.no_grad()
        def act(self, obs, deterministic=True):
            z = self.mod["encoder"](obs["pc"])
            mu, _ = self.mod["pi"](z)
            return mu.clamp(-1, 1).cpu().numpy()

    return agent, EvalSAC(agent)

    model_cfg["embed_dim"] = 384
    model_cfg["state_embed_dim"] = 384
    model_cfg["action_dim"] = action_dim

    encoder = hydra.utils.instantiate(model_cfg).to(device)

def build_agent_and_algo_old(cfg, device, action_dim):
    """Rebuild encoder + heads from Hydra cfg (plus sensible fallbacks)."""

    # ─── 0. Short-hand dicts ──────────────────────────────────────────────
    m      = OmegaConf.to_container(cfg["model"], resolve=True)
    sac_hp = OmegaConf.to_container(cfg.get("algo", {}), resolve=True)  # may be missing

    # ─── 1. Encoder hyper-parameters ─────────────────────────────────────
    embed_dim   = m.get("embed_dim", 128)
    point_dim   = m.get("point_dim", 6)
    max_points  = m.get("max_expected_num_points", 1024)
    group_size  = m.get("group_size", 32)
    samp_ratio  = m.get("sampling_ratio", 0.1)

    # ─── 2. Observation space ────────────────────────────────────────────
    obs_space = PointCloudSpace(
        max_expected_num_points=max_points,
        low=-1.0,
        high=1.0,
        feature_shape=(point_dim,),
    )
    def make_tokenizer(*, point_dim: int, embed_dim: int):
        # factories must accept **named** arguments:
        def mlp_1(*, input_size: int) -> nn.Module:
            return MLP([input_size, 64, 64])

        def mlp_2(*, output_size: int) -> nn.Module:
            return MLP([128, 128, embed_dim])

        from pprl.models.modules.tokenizer import Tokenizer
        return Tokenizer(
            mlp_1=mlp_1,
            mlp_2=mlp_2,
            group_size=group_size,
            sampling_ratio=samp_ratio,
            point_dim=point_dim,
            embed_dim=embed_dim,
        )

    def make_pos_embedder(*, token_dim: int):
        # return nn.Sequential(nn.Linear(3, embed_dim), nn.GELU())
        return nn.Sequential(nn.Linear(3, token_dim), nn.GELU())

    def make_transformer_encoder(*, embed_dim: int):
        """Return a depth-N encoder whose sub-factories accept keyword args."""

        from pprl.models.modules.transformer import TransformerEncoder, TransformerBlock

        # ── attention, MLP, block factories – all keyword-aware ────────────
        def attn_factory(*, embed_dim: int):
            return nn.MultiheadAttention(embed_dim=embed_dim,
                                         num_heads=8,
                                         batch_first=True)

        def mlp_factory(*, embed_dim: int):
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            )

        def block_factory(*, embed_dim: int):
            return TransformerBlock(
                attention=attn_factory,
                mlp=mlp_factory,
                embed_dim=embed_dim,
            )

        depth = m.get("transformer_depth", 4)
        return TransformerEncoder(
            block_factory=block_factory,
            embed_dim=embed_dim,
            depth=depth,
        )

    ppt = PointPatchTransformer(
        obs_space=obs_space,
        tokenizer=make_tokenizer,
        pos_embedder=make_pos_embedder,
        transformer_encoder=make_transformer_encoder,
        embed_dim=embed_dim,
    ).to(device)

    action_space = get_action_space(cfg)

    # ─── 4. Head hyper-parameters (with cfg overrides) ───────────────────
    pi_hidden_sizes = cfg.pi_mlp_head.value.hidden_sizes
    pi_hidden_nonlinearity = cfg.pi_mlp_head.value.hidden_nonlinearity

    q_hidden_sizes = cfg.q_mlp_head.value.hidden_sizes
    q_hidden_nonlinearity = cfg.q_mlp_head.value.hidden_nonlinearity

    pi_head = PiMlpHead(
        input_size=embed_dim,
        action_size=action_dim,
        action_space=action_space,
        hidden_sizes=pi_hidden_sizes,
        hidden_nonlinearity=pi_hidden_nonlinearity,
    ).to(device)

    q1_head = QMlpHead(
        input_size=embed_dim,
        action_size=action_dim,
        hidden_sizes=q_hidden_sizes,
        hidden_nonlinearity=q_hidden_nonlinearity,
    ).to(device)

    q2_head = QMlpHead(
        input_size=embed_dim,
        action_size=action_dim,
        hidden_sizes=q_hidden_sizes,
        hidden_nonlinearity=q_hidden_nonlinearity,
    ).to(device)

    agent = torch.nn.ModuleDict(
        {"encoder": ppt, "pi": pi_head, "q1": q1_head, "q2": q2_head}
    ).to(device)

    # ─── 5. Minimal deterministic policy wrapper ────────────────────────
    class EvalSAC:
        def __init__(self, mod): self.mod = mod
        @torch.no_grad()
        def act(self, obs, deterministic=True):
            z = self.mod["encoder"](obs["pc"])
            mu, _ = self.mod["pi"](z)
            return mu.clamp(-1, 1).cpu().numpy()

    return agent, EvalSAC(agent)


def load_checkpoint(ckpt_path: Path, device: str,
                    cfg_path: Path | None = None):
    """
    Loads the checkpoint and rebuilds the agent (PPT + heads).
    Accepts both 'rich' checkpoints (with cfg inside) and plain state-dicts
    when --cfg is provided.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    # obtain Hydra cfg ----------------------------------------------------
    if "cfg" in ckpt:                # rich checkpoint
        cfg = ckpt["cfg"]
        state_dict = ckpt["model"]   # nested state-dict under 'model'
    else:                            # plain state-dict
        if cfg_path is None:
            raise ValueError("Checkpoint lacks 'cfg'.  Pass --cfg <yaml>.")
        cfg = OmegaConf.load(cfg_path)
        state_dict = ckpt            # the file *is* the state-dict

    action_dim = _infer_action_dim_from_state_dict(cfg)
    print("action_dim ", action_dim)

    # ── build networks with the correct action_dim ──────────────────────
    agent, eval_policy = build_agent_and_algo(cfg, device, action_dim)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()


    return cfg, eval_policy

# ─────────────────────────────────────────────────────────────────────────────
#  Per-camera evaluation routine
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_one_camera(cfg_base: DictConfig,
                        cam_cfgs,
                        policy,
                        n_episodes: int,
                        device: str = "cuda:0"):

    # 1) patch the camera configuration
    cfg = copy.deepcopy(cfg_base)
    with open_dict(cfg.env.create_scene_kwargs):
        cfg.env.create_scene_kwargs["camera_configs"] = _pylist_of_dicts(cam_cfgs)
    cfg.env.mode = "eval"

    # 2) build cages (vectorised envs)
    TrajInfoClass = get_class(cfg.env.pop("traj_info"))
    base_eval_factory = build_base_env_factory(cfg.env)

    eval_cages, metadata = build_cages(
        EnvClass=base_eval_factory,
        n_envs=cfg.eval.n_eval_envs,
        env_kwargs={"add_rendering_to_info": False},
        TrajInfoClass=TrajInfoClass,
        parallel=False,
    )

    # 3) sample-tree & sampler (tiny: only obs/action/reward/terminated)
    batch_spec = BatchSpec(T=cfg.env.max_episode_steps, B=cfg.eval.n_eval_envs)
    eval_tree_example = metadata.example_timestep  # already in Array format
    eval_sample_tree = eval_tree_example.new_array(batch_shape=(1, cfg.eval.n_eval_envs))

    eval_sampler = EvalSampler(
        max_traj_length=cfg.env.max_episode_steps,
        max_trajectories=n_episodes,
        envs=eval_cages,
        agent=policy,
        sample_tree=eval_sample_tree,
    )

    # 4) run the evaluation
    runner = EvalRunner(
        eval_sampler       = eval_sampler,
        n_eval_steps       = n_episodes * cfg.env.max_episode_steps,
        log_interval_steps = 9999999,      # don't spam logger
    )
    runner.run()

    # 5) pull trajectory metrics from TrajInfo objects
    returns   = [ti.Return for ti in eval_sampler.completed_traj_infos]
    successes = [ti.Success for ti in eval_sampler.completed_traj_infos] \
                if hasattr(eval_sampler.completed_traj_infos[0], "Success") else None

    for cage in eval_cages: cage.close()
    eval_sampler.close()
    eval_sample_tree.close()

    result = {
        "meanR": np.mean(returns),
        "success": 100 * np.mean(successes) if successes is not None else np.nan,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--cfg",  type=Path, default=None, help="Hydra YAML (only if ckpt has no cfg)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device",   default="cuda:0")
    args = ap.parse_args()

    # 0) load checkpoint & rebuild agent
    cfg_base, eval_policy = load_checkpoint(args.ckpt, args.device, args.cfg)

    # 1) camera perturbations
    camera_sets = make_camera_setups()

    # 2) loop and collect stats
    rows = []
    for cam_id, cam_list in camera_sets.items():
        stats = evaluate_one_camera(cfg_base, cam_list, eval_policy, args.episodes,
                                    device=args.device)
        rows.append((cam_id, stats["meanR"], stats["success"]))
        print(f"{cam_id:<22}  R̄={stats['meanR']:.2f}   "
              f"succ={stats['success']:.1f}%")

    # 3) pretty table
    df = pd.DataFrame(rows, columns=["camera_id", "meanR", "success %"])
    print("\n" + df.to_markdown(index=False))

if __name__ == "__main__":
    main()

