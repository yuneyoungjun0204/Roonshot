"""MAPPO Trainer: 롤아웃 수집 + PPO 업데이트 (멀티 환경 병렬 지원)"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .config import MAPPOConfig
from .network import Actor, CentralizedCritic
from .buffer import RolloutBuffer


class RunningMeanStd:
    """관측 정규화용 Running Mean/Std (ML-Agents normalize=true 대응)"""

    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor):
        """배치 단위 업데이트 (Welford's algorithm)"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.var = d["var"]
        self.count = d["count"]


class MAPPOTrainer:
    """
    MAPPO 학습기 (멀티 환경 병렬 지원)
    - N개 Unity 환경에서 병렬 롤아웃 수집
    - 환경별 독립 버퍼/LSTM hidden state/에피소드 추적
    - 모든 버퍼 청크를 합쳐 PPO 업데이트
    """

    def __init__(self, cfg: MAPPOConfig, vec_env):
        """
        Args:
            cfg: MAPPO configuration
            vec_env: VecEnvWrapper 또는 SingleEnvAdapter
        """
        self.vec_env = vec_env
        self.num_envs = vec_env.num_envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MAPPO] Device: {self.device}, Envs: {self.num_envs}")

        # Unity에서 보내주는 실제 관측 차원으로 config 업데이트
        unity_obs = vec_env.unity_obs_dim
        cfg.stacked_obs_dim_override = unity_obs
        self.cfg = cfg

        print(f"[MAPPO] Unity obs dim: {unity_obs} → network input")

        # 네트워크
        self.actor = Actor(cfg).to(self.device)
        self.critic = CentralizedCritic(cfg).to(self.device)

        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters())
        print(f"[MAPPO] Actor: {n_actor:,} params, Critic: {n_critic:,} params")

        # 옵티마이저
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # LR 스케줄러 (ML-Agents: learning_rate_schedule=linear)
        if cfg.lr_schedule == "linear":
            total_updates_est = cfg.total_steps // cfg.rollout_steps
            self.actor_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.actor_optimizer,
                start_factor=1.0, end_factor=0.0,
                total_iters=total_updates_est,
            )
            self.critic_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.critic_optimizer,
                start_factor=1.0, end_factor=0.0,
                total_iters=total_updates_est,
            )
            print(f"[MAPPO] LR schedule: linear decay over {total_updates_est} updates")
        else:
            self.actor_scheduler = None
            self.critic_scheduler = None
            print(f"[MAPPO] LR schedule: constant")

        # 관측 정규화 (ML-Agents: normalize=true)
        if cfg.normalize_obs:
            self.obs_normalizer = RunningMeanStd(unity_obs, self.device)
            print(f"[MAPPO] Observation normalization: ON")
        else:
            self.obs_normalizer = None

        # 환경별 독립 버퍼
        self.buffers = [RolloutBuffer(cfg, self.device) for _ in range(self.num_envs)]

        # 환경별 LSTM hidden states
        self._actor_hiddens = [
            self.actor.init_hidden(cfg.n_agents, self.device)
            for _ in range(self.num_envs)
        ]
        self._critic_hiddens = [
            self.critic.init_hidden(1, self.device)
            for _ in range(self.num_envs)
        ]

        # 환경별 에피소드 추적
        self._ep_trackers = [
            {"reward": 0.0, "individual": 0.0, "group": 0.0, "length": 0}
            for _ in range(self.num_envs)
        ]

        # 로깅
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.log_dir)

        # 통계
        self.total_steps = 0
        self.total_updates = 0

        steps_per_env = cfg.rollout_steps // self.num_envs
        print(f"[MAPPO] Rollout: {cfg.rollout_steps} total "
              f"({steps_per_env} steps × {self.num_envs} envs)")

    # ──────────────────────────────────────────────
    # Hidden State / Episode Tracker 관리
    # ──────────────────────────────────────────────

    def _reset_env_hidden(self, env_idx: int):
        """특정 환경의 LSTM hidden state 리셋"""
        self._actor_hiddens[env_idx] = self.actor.init_hidden(
            self.cfg.n_agents, self.device)
        self._critic_hiddens[env_idx] = self.critic.init_hidden(
            1, self.device)

    def _reset_all_hiddens(self):
        """모든 환경 LSTM hidden state 리셋"""
        for i in range(self.num_envs):
            self._reset_env_hidden(i)

    def _reset_ep_tracker(self, env_idx: int):
        """특정 환경의 에피소드 추적 리셋"""
        self._ep_trackers[env_idx] = {
            "reward": 0.0, "individual": 0.0, "group": 0.0, "length": 0
        }

    def _normalize_obs(self, obs_t: torch.Tensor, update: bool = True) -> torch.Tensor:
        """관측 정규화 (running mean/std)"""
        if self.obs_normalizer is None:
            return obs_t
        if update:
            self.obs_normalizer.update(obs_t)
        return self.obs_normalizer.normalize(obs_t)

    # ──────────────────────────────────────────────
    # 롤아웃 수집 (멀티 환경 병렬)
    # ──────────────────────────────────────────────

    def collect_rollout(self) -> dict:
        """
        N개 환경에서 병렬로 rollout_steps만큼 데이터 수집

        Returns:
            ep_stats: 수집 중 완료된 에피소드 통계
        """
        for buf in self.buffers:
            buf.clear()

        # 모든 환경 리셋
        obs_list = self.vec_env.reset_all()
        self._reset_all_hiddens()
        for i in range(self.num_envs):
            self._reset_ep_tracker(i)

        completed_episodes = []
        steps_per_env = self.cfg.rollout_steps // self.num_envs

        for step in range(steps_per_env):
            # ── 1. 각 환경별 액션 계산 ──
            actions_np_list = []
            env_data = []

            for env_idx in range(self.num_envs):
                obs_t = torch.tensor(
                    obs_list[env_idx], dtype=torch.float32, device=self.device)
                obs_norm = self._normalize_obs(obs_t)
                joint_obs_t = obs_norm.reshape(1, -1)

                with torch.no_grad():
                    actions, log_probs, new_ah = self.actor.act(
                        obs_norm, self._actor_hiddens[env_idx])
                    value, new_ch = self.critic(
                        joint_obs_t, self._critic_hiddens[env_idx])
                    values = value.squeeze(1).expand(self.cfg.n_agents, 1)

                actions_np_list.append(actions.cpu().numpy())
                env_data.append({
                    "obs_norm": obs_norm,
                    "joint_obs": joint_obs_t,
                    "actions": actions,
                    "log_probs": log_probs,
                    "values": values,
                    "new_ah": new_ah,
                    "new_ch": new_ch,
                })

            # ── 2. 모든 환경 병렬 스텝 ──
            results = self.vec_env.step_all(actions_np_list)

            # ── 3. 결과 처리 + 버퍼 저장 ──
            for env_idx in range(self.num_envs):
                next_obs, rewards, done, info = results[env_idx]
                d = env_data[env_idx]
                t = self._ep_trackers[env_idx]

                # 에피소드 보상 추적
                t["reward"] += rewards.sum()
                t["individual"] += info["individual_rewards"].sum()
                t["group"] += info["group_rewards"].sum()
                t["length"] += 1

                # 환경별 버퍼에 저장
                self.buffers[env_idx].add(
                    obs=d["obs_norm"].cpu(),
                    joint_obs=d["joint_obs"].squeeze(0).cpu(),
                    actions=d["actions"].cpu(),
                    log_probs=d["log_probs"].cpu(),
                    rewards=rewards,
                    done=done,
                    values=d["values"].cpu(),
                    actor_hidden=self._actor_hiddens[env_idx],
                    critic_hidden=self._critic_hiddens[env_idx],
                )

                # Hidden state 업데이트
                self._actor_hiddens[env_idx] = d["new_ah"]
                self._critic_hiddens[env_idx] = d["new_ch"]
                obs_list[env_idx] = next_obs

                if done:
                    completed_episodes.append({
                        "total_reward": t["reward"],
                        "individual_reward": t["individual"],
                        "group_reward": t["group"],
                        "length": t["length"],
                    })
                    self._reset_ep_tracker(env_idx)
                    self._reset_env_hidden(env_idx)

            self.total_steps += self.num_envs

        # ── 4. 환경별 GAE 계산 ──
        for env_idx in range(self.num_envs):
            with torch.no_grad():
                obs_t = torch.tensor(
                    obs_list[env_idx], dtype=torch.float32, device=self.device)
                obs_norm = self._normalize_obs(obs_t, update=False)
                joint_obs_t = obs_norm.reshape(1, -1)
                last_value, _ = self.critic(
                    joint_obs_t, self._critic_hiddens[env_idx])
                last_value = last_value.squeeze(1).expand(self.cfg.n_agents, 1)
            self.buffers[env_idx].compute_gae(last_value)

        # 에피소드 통계 반환
        if completed_episodes:
            ep_stats = {
                "mean_reward": np.mean([e["total_reward"] for e in completed_episodes]),
                "mean_individual_reward": np.mean([e["individual_reward"] for e in completed_episodes]),
                "mean_group_reward": np.mean([e["group_reward"] for e in completed_episodes]),
                "mean_length": np.mean([e["length"] for e in completed_episodes]),
                "episode_count": len(completed_episodes),
            }
        else:
            ep_stats = {
                "mean_reward": 0.0,
                "mean_individual_reward": 0.0,
                "mean_group_reward": 0.0,
                "mean_length": 0.0,
                "episode_count": 0,
            }

        return ep_stats

    # ──────────────────────────────────────────────
    # PPO 업데이트
    # ──────────────────────────────────────────────

    def update(self) -> dict:
        """
        PPO 업데이트 (num_epochs × 미니배치)
        모든 환경 버퍼의 청크를 합쳐서 학습
        """
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for epoch in range(self.cfg.num_epochs):
            for batch in RolloutBuffer.combined_chunks(self.buffers, self.cfg):
                # ── Actor 업데이트 ──
                actor_hidden = (batch.actor_h, batch.actor_c)
                new_log_probs, entropy, _ = self.actor.evaluate(
                    batch.obs, batch.actions, actor_hidden)

                # Advantage 정규화
                adv = batch.advantages
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # PPO Clipped Loss
                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps,
                                    1 + self.cfg.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                self.actor_optimizer.zero_grad()
                (actor_loss + self.cfg.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()

                # ── Critic 업데이트 ──
                critic_hidden = (batch.critic_h, batch.critic_c)
                pred_values, _ = self.critic(batch.joint_obs, critic_hidden)
                critic_loss = nn.functional.mse_loss(pred_values, batch.returns)

                self.critic_optimizer.zero_grad()
                (self.cfg.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        self.total_updates += 1
        n_updates = max(n_updates, 1)

        # LR decay
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
            self.critic_scheduler.step()

        current_lr = self.actor_optimizer.param_groups[0]["lr"]

        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "lr": current_lr,
        }

    # ──────────────────────────────────────────────
    # 메인 학습 루프
    # ──────────────────────────────────────────────

    def train(self):
        """메인 학습 루프"""
        print(f"[MAPPO] Training start — total_steps={self.cfg.total_steps}, "
              f"envs={self.num_envs}")
        start_time = time.time()

        while self.total_steps < self.cfg.total_steps:
            # 1. 롤아웃 수집
            ep_stats = self.collect_rollout()

            # 2. PPO 업데이트
            losses = self.update()

            # 3. 로깅
            elapsed = time.time() - start_time
            fps = self.total_steps / max(elapsed, 1)
            mean_reward = ep_stats["mean_reward"]

            # Reward 상세 분리
            self.writer.add_scalar("reward/mean_episode", mean_reward, self.total_steps)
            self.writer.add_scalar("reward/individual", ep_stats["mean_individual_reward"], self.total_steps)
            self.writer.add_scalar("reward/group", ep_stats["mean_group_reward"], self.total_steps)

            # Episode 통계
            self.writer.add_scalar("episode/length", ep_stats["mean_length"], self.total_steps)
            self.writer.add_scalar("episode/count", ep_stats["episode_count"], self.total_steps)

            # Loss
            self.writer.add_scalar("loss/actor", losses["actor_loss"], self.total_steps)
            self.writer.add_scalar("loss/critic", losses["critic_loss"], self.total_steps)
            self.writer.add_scalar("loss/entropy", losses["entropy"], self.total_steps)

            # 기타
            self.writer.add_scalar("lr", losses["lr"], self.total_steps)
            self.writer.add_scalar("perf/fps", fps, self.total_steps)

            if self.total_updates % self.cfg.log_interval == 0:
                print(f"[Step {self.total_steps:>8d}] "
                      f"reward={mean_reward:>7.2f} "
                      f"(ind={ep_stats['mean_individual_reward']:.3f} "
                      f"grp={ep_stats['mean_group_reward']:.3f})  "
                      f"len={ep_stats['mean_length']:.0f}  "
                      f"actor={losses['actor_loss']:.4f}  "
                      f"critic={losses['critic_loss']:.4f}  "
                      f"entropy={losses['entropy']:.4f}  "
                      f"lr={losses['lr']:.6f}  "
                      f"fps={fps:.0f}")

            if self.total_updates % self.cfg.save_interval == 0:
                self.save(f"checkpoint_{self.total_steps}")

        self.save("final")
        self.writer.close()
        print(f"[MAPPO] Training complete — {self.total_steps} steps")

    # ──────────────────────────────────────────────
    # 저장/로드/변환
    # ──────────────────────────────────────────────

    def save(self, name: str):
        """모델 저장"""
        path = os.path.join(self.cfg.log_dir, f"{name}.pt")
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_updates": self.total_updates,
            "config": self.cfg,
        }
        if self.obs_normalizer is not None:
            save_dict["obs_normalizer"] = self.obs_normalizer.state_dict()
        torch.save(save_dict, path)
        print(f"[MAPPO] Model saved: {path}")

    def load(self, path: str):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_updates = checkpoint.get("total_updates", 0)
        if self.obs_normalizer is not None and "obs_normalizer" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["obs_normalizer"])
            print(f"[MAPPO] Obs normalizer restored (count={self.obs_normalizer.count:.0f})")
        print(f"[MAPPO] Model loaded: {path} (step={self.total_steps})")

    def export_onnx(self, path: str = None):
        """Actor를 ONNX로 변환 (Unity Sentis/Barracuda 추론용)"""
        if path is None:
            path = os.path.join(self.cfg.log_dir, "actor.onnx")

        self.actor.eval()
        dummy_obs = torch.zeros(1, 1, self.cfg.stacked_obs_dim, device=self.device)
        dummy_h = self.actor.init_hidden(1, self.device)

        torch.onnx.export(
            self.actor,
            (dummy_obs, dummy_h),
            path,
            input_names=["obs", "h_in", "c_in"],
            output_names=["mean", "log_std", "h_out", "c_out"],
            dynamic_axes={"obs": {0: "batch"}},
            opset_version=14,
        )
        print(f"[MAPPO] ONNX exported: {path}")
