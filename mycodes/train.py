import argparse
import os
import sys
import time
import random
from typing import Tuple
import csv

#ppo RL
train_parser = argparse.ArgumentParser(
    description="PPO training for IsaacLab bin packing",
    add_help=True,
)

train_parser.add_argument("--train_steps", type=int, default=1000, help="Total vectorized env transitions")
train_parser.add_argument("--rollout_steps", type=int, default=20, help="Number of RL steps per PPO rollout")
train_parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO optimization epochs per rollout")
train_parser.add_argument("--mini_batch_size", type=int, default=64, help="PPO minibatch size")
train_parser.add_argument("--learning_rate", type=float, default=3.0e-4, help="Optimizer learning rate")
train_parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
train_parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
train_parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO clipping coefficient")
train_parser.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
train_parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy bonus coefficient")
train_parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Gradient clipping norm")
train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
train_parser.add_argument("--save_policy", action="store_true", default=False, help="Save trained policy weights")
train_parser.add_argument("--policy_out", type=str, default="ppo_bin_packing_policy.pt", help="Policy output path")
train_parser.add_argument("--log_dir", type=str, default="training_logs", help="Directory for training logs")
train_parser.add_argument("--run_name", type=str, default="ppo_bin_packing", help="Training run name")

# Parse known args:
#   train_args = args for this train.py
#   env_args   = args that should be parsed by bin_packing_env.py
train_args, env_args = train_parser.parse_known_args()

# Pass only environment/IsaacLab args to bin_packing_env.py.
# bin_packing_env.py will parse them and launch the SimulationApp.
sys.argv = [sys.argv[0]] + env_args


import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import bin_packing_env as bpe  # type: ignore  # noqa: E402

import numpy as np  
import torch  
import torch.nn as nn 
import torch.optim as optim  


# Utility functions

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_vector_observation(obs) -> torch.Tensor:
    """Return vector observation tensor.

    This PPO test uses vector observations.
    If obs is multimodal, use the vector branch.
    """
    if isinstance(obs, dict):
        if "vector" not in obs:
            raise ValueError("Multimodal observation dictionary has no 'vector' key.")
        obs = obs["vector"]

    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    return obs.float()


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters()])


# Actor-Critic policy network

class ActorCritic(nn.Module):
    """Small MLP actor-critic for continuous drop-position actions.

    Input:
        vector observation [num_envs, obs_dim]

    Output:
        Gaussian policy over action [a_x, a_y].
        Environment clamps actions to [-1, 1].
    """

    def __init__(self, obs_dim: int, action_dim: int = 2):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(256, action_dim)
        self.critic = nn.Linear(256, 1)

        # Trainable log standard deviation for Gaussian policy.
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Conservative initialization helps prevent very large early actions.
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        features = self.backbone(obs)

        # Mean is bounded to [-1, 1].
        mean = torch.tanh(self.actor_mean(features))

        value = self.critic(features).squeeze(-1)

        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)

        return dist, value

    def act(self, obs: torch.Tensor):
        dist, value = self.forward(obs)

        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        # The simulator expects normalized action in [-1, 1].
        env_action = torch.clamp(raw_action, -1.0, 1.0)

        return raw_action, env_action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, raw_actions: torch.Tensor):
        dist, value = self.forward(obs)
        log_prob = dist.log_prob(raw_actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# Build IsaacLab environment from bin_packing_env.py

def make_env():
    sim_cfg = bpe.sim_utils.SimulationCfg(device=bpe.args_cli.device)
    sim = bpe.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=[0.0, -3.0, 5.0],
        target=[0.0, 0.0, 0.75],
    )

    scene_cfg = bpe.CubeDropSceneCfg(
        num_envs=bpe.args_cli.num_envs,
        env_spacing=bpe.args_cli.env_spacing,
    )
    scene = bpe.InteractiveScene(scene_cfg)

    camera = bpe.make_debug_camera(sim)

    sim.reset()

    camera.reset()
    camera.update(dt=sim.get_physics_dt())

    cam_pos = torch.tensor([[0.0, -2.6, 4.8]], device=sim.device)
    cam_target = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    camera.set_world_poses_from_view(cam_pos, cam_target)
    camera.update(dt=sim.get_physics_dt())

    sim.play()

    env = bpe.BinPackingRLEnv(sim=sim, scene=scene, camera=camera)
    return env


# PPO training

def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute generalized advantage estimation.

    Shapes:
        rewards: [T, N]
        dones:   [T, N]
        values:  [T, N]
        next_value: [N]
        next_done:  [N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        if t == T - 1:
            next_nonterminal = 1.0 - next_done.float()
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1].float()
            next_values = values[t + 1]

        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_nonterminal * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values
    return advantages, returns


def train():
    set_seed(train_args.seed)

    if bpe.args_cli.obs_type != "vector":
        print(
            "[WARN] This PPO smoke-test trainer uses vector observations. "
            f"You passed --obs_type {bpe.args_cli.obs_type}. "
            "For training, please use --obs_type vector.",
            flush=True,
        )
        raise ValueError("train.py currently supports --obs_type vector for stable PPO testing.")

    env = make_env()
    device = env.device

    obs = env.reset()
    obs_t = get_vector_observation(obs).to(device)

    num_envs = env.num_envs
    obs_dim = obs_t.shape[1]
    action_dim = 2

    policy = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=train_args.learning_rate, eps=1.0e-5)

    print("[INFO] PPO training started", flush=True)
    print(f"[INFO] num_envs          = {num_envs}", flush=True)
    print(f"[INFO] num_cubes         = {bpe.args_cli.num_cubes}", flush=True)
    print(f"[INFO] cube_size         = {bpe.args_cli.cube_size}", flush=True)
    print(f"[INFO] obs_dim           = {obs_dim}", flush=True)
    print(f"[INFO] action_dim        = {action_dim}", flush=True)
    print(f"[INFO] reward_type       = {bpe.args_cli.reward_type}", flush=True)
    print(f"[INFO] train_steps       = {train_args.train_steps}", flush=True)
    print(f"[INFO] rollout_steps     = {train_args.rollout_steps}", flush=True)
    print(f"[INFO] ppo_epochs        = {train_args.ppo_epochs}", flush=True)
    print(f"[INFO] mini_batch_size   = {train_args.mini_batch_size}", flush=True)
    
    os.makedirs(train_args.log_dir, exist_ok=True)
    log_path = os.path.join(train_args.log_dir, f"{train_args.run_name}.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "update",
            "global_transitions",
            "rollout_reward_mean",
            "pg_loss",
            "v_loss",
            "entropy",
            "approx_kl",
            "param_delta",
            "elapsed_s",
        ])

    print(f"[INFO] Logging training metrics to: {log_path}", flush=True)

    global_transitions = 0
    update_idx = 0
    episode_idx = 0
    train_t0 = time.perf_counter()

    current_done = torch.zeros(num_envs, dtype=torch.bool, device=device)

    while global_transitions < train_args.train_steps and bpe.simulation_app.is_running():
        update_idx += 1

        obs_buf = []
        raw_action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        rollout_reward_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)

        for rollout_step in range(train_args.rollout_steps):
            with torch.no_grad():
                raw_action, env_action, log_prob, value = policy.act(obs_t)

            next_obs, reward, done, info = env.step(env_action)

            next_obs_t = get_vector_observation(next_obs).to(device)

            obs_buf.append(obs_t.detach())
            raw_action_buf.append(raw_action.detach())
            logprob_buf.append(log_prob.detach())
            reward_buf.append(reward.detach())
            done_buf.append(done.detach())
            value_buf.append(value.detach())

            rollout_reward_sum += reward.detach()

            global_transitions += num_envs
            obs_t = next_obs_t
            current_done = done

            if torch.all(done):
                episode_idx += 1

                print(
                    f"[EPISODE] idx={episode_idx:04d} | "
                    f"global_transitions={global_transitions} | "
                    f"sim_steps={env.episode_step} | "
                    f"success_envs={int(info['success_envs'].sum().item())}/{num_envs} | "
                    f"reward_sum_mean={env.episode_reward_sum.mean().item():.4f}",
                    flush=True,
                )

                obs = env.reset()
                obs_t = get_vector_observation(obs).to(device)
                current_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
                break

            if global_transitions >= train_args.train_steps:
                break

        # Stack rollout buffers.
        obs_b = torch.stack(obs_buf, dim=0)                  # [T, N, obs_dim]
        raw_action_b = torch.stack(raw_action_buf, dim=0)    # [T, N, action_dim]
        old_logprob_b = torch.stack(logprob_buf, dim=0)      # [T, N]
        reward_b = torch.stack(reward_buf, dim=0)            # [T, N]
        done_b = torch.stack(done_buf, dim=0)                # [T, N]
        value_b = torch.stack(value_buf, dim=0)              # [T, N]

        with torch.no_grad():
            _, next_value = policy.forward(obs_t)
            advantages, returns = compute_gae(
                rewards=reward_b,
                dones=done_b,
                values=value_b,
                next_value=next_value.detach(),
                next_done=current_done.detach(),
                gamma=train_args.gamma,
                gae_lambda=train_args.gae_lambda,
            )

        # Flatten [T, N] -> [T*N]
        b_obs = obs_b.reshape(-1, obs_dim)
        b_raw_actions = raw_action_b.reshape(-1, action_dim)
        b_old_logprob = old_logprob_b.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_b.reshape(-1)

        # Normalize advantages.
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1.0e-8)

        batch_size = b_obs.shape[0]
        mini_batch_size = min(train_args.mini_batch_size, batch_size)

        old_params = flatten_params(policy)

        last_pg_loss = None
        last_v_loss = None
        last_entropy = None
        last_approx_kl = None

        inds = np.arange(batch_size)

        for _ in range(train_args.ppo_epochs):
            np.random.shuffle(inds)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = inds[start:end]
                mb_inds_t = torch.tensor(mb_inds, dtype=torch.long, device=device)

                new_logprob, entropy, new_value = policy.evaluate_actions(
                    b_obs[mb_inds_t],
                    b_raw_actions[mb_inds_t],
                )

                logratio = new_logprob - b_old_logprob[mb_inds_t]
                ratio = torch.exp(logratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()

                mb_adv = b_advantages[mb_inds_t]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio,
                    1.0 - train_args.clip_coef,
                    1.0 + train_args.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_value - b_returns[mb_inds_t]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    + train_args.value_coef * v_loss
                    - train_args.entropy_coef * entropy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), train_args.max_grad_norm)
                optimizer.step()

                last_pg_loss = pg_loss.detach()
                last_v_loss = v_loss.detach()
                last_entropy = entropy_loss.detach()
                last_approx_kl = approx_kl.detach()

        new_params = flatten_params(policy)
        param_delta = torch.linalg.norm(new_params - old_params).item()

        elapsed = time.perf_counter() - train_t0

        print(
            f"[UPDATE] update={update_idx:04d} | "
            f"global_transitions={global_transitions:06d}/{train_args.train_steps} | "
            f"rollout_reward_mean={rollout_reward_sum.mean().item(): .4f} | "
            f"pg_loss={last_pg_loss.item(): .5f} | "
            f"v_loss={last_v_loss.item(): .5f} | "
            f"entropy={last_entropy.item(): .5f} | "
            f"approx_kl={last_approx_kl.item(): .6f} | "
            f"param_delta={param_delta:.6e} | "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                update_idx,
                global_transitions,
                rollout_reward_sum.mean().item(),
                last_pg_loss.item(),
                last_v_loss.item(),
                last_entropy.item(),
                last_approx_kl.item(),
                param_delta,
                elapsed,
            ])

    if train_args.save_policy:
        os.makedirs(os.path.dirname(train_args.policy_out) or ".", exist_ok=True)
        torch.save(
            {
                "model_state_dict": policy.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "train_args": vars(train_args),
                "env_args": vars(bpe.args_cli),
            },
            train_args.policy_out,
        )
        print(f"[INFO] Saved policy to {train_args.policy_out}", flush=True)

    print("[INFO] PPO training completed successfully.", flush=True)

    bpe.simulation_app.close()


if __name__ == "__main__":
    train()