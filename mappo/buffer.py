"""롤아웃 버퍼 + GAE 계산"""

import torch
import numpy as np
from typing import Generator, NamedTuple

from .config import MAPPOConfig


class ChunkBatch(NamedTuple):
    """청크 미니배치 데이터"""
    obs: torch.Tensor            # (batch, chunk_len, stacked_obs_dim)
    joint_obs: torch.Tensor      # (batch, chunk_len, joint_obs_dim)
    actions: torch.Tensor        # (batch, chunk_len, action_dim)
    old_log_probs: torch.Tensor  # (batch, chunk_len, 1)
    advantages: torch.Tensor     # (batch, chunk_len, 1)
    returns: torch.Tensor        # (batch, chunk_len, 1)
    masks: torch.Tensor          # (batch, chunk_len, 1)
    actor_h: torch.Tensor        # (lstm_layers, batch, hidden_dim)
    actor_c: torch.Tensor
    critic_h: torch.Tensor
    critic_c: torch.Tensor


class RolloutBuffer:
    """
    MAPPO 롤아웃 버퍼
    - 2 에이전트 데이터를 스텝 단위로 저장
    - GAE 계산 후 청크 단위 미니배치 생성
    """

    def __init__(self, cfg: MAPPOConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.clear()

    def clear(self):
        self.obs = []           # [(n_agents, stacked_obs_dim), ...]
        self.joint_obs = []     # [(joint_obs_dim,), ...]
        self.actions = []       # [(n_agents, action_dim), ...]
        self.log_probs = []     # [(n_agents, 1), ...]
        self.rewards = []       # [(n_agents,), ...]
        self.dones = []         # [bool, ...]
        self.values = []        # [(n_agents, 1), ...]
        # LSTM hidden states (청크 시작점 복원용)
        self.actor_hiddens = []   # [((layers, n_agents, h), (layers, n_agents, h)), ...]
        self.critic_hiddens = []  # [((layers, 1, h), (layers, 1, h)), ...]

    def add(self, obs, joint_obs, actions, log_probs, rewards, done, values,
            actor_hidden, critic_hidden):
        self.obs.append(obs)
        self.joint_obs.append(joint_obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(values)
        self.actor_hiddens.append(
            (actor_hidden[0].detach().cpu(), actor_hidden[1].detach().cpu()))
        self.critic_hiddens.append(
            (critic_hidden[0].detach().cpu(), critic_hidden[1].detach().cpu()))

    def __len__(self):
        return len(self.obs)

    def compute_gae(self, last_value: torch.Tensor):
        """
        GAE(Generalized Advantage Estimation) 계산
        last_value: (n_agents, 1) - 마지막 스텝의 value 예측
        """
        n = len(self)
        n_agents = self.cfg.n_agents
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        # 텐서 변환
        rewards = torch.stack([torch.tensor(r, dtype=torch.float32) for r in self.rewards])  # (T, n_agents)
        values = torch.stack(self.values).squeeze(-1).cpu()  # (T, n_agents)
        dones = torch.tensor(self.dones, dtype=torch.float32)  # (T,)

        # last_value 추가
        all_values = torch.cat([values, last_value.squeeze(-1).cpu().unsqueeze(0)], dim=0)

        advantages = torch.zeros(n, n_agents)
        gae = torch.zeros(n_agents)

        for t in reversed(range(n)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * all_values[t + 1] * mask - all_values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae

        returns = advantages + values
        self._advantages = advantages  # (T, n_agents)
        self._returns = returns        # (T, n_agents)

    def _build_chunks(self) -> list:
        """개별 (미배치) 청크 리스트 생성"""
        n = len(self)
        chunk_len = self.cfg.chunk_length
        n_agents = self.cfg.n_agents

        obs_t = torch.stack(self.obs)            # (T, n_agents, stacked_obs)
        joint_t = torch.stack(self.joint_obs)    # (T, joint_obs)
        act_t = torch.stack(self.actions)         # (T, n_agents, action_dim)
        logp_t = torch.stack(self.log_probs)     # (T, n_agents, 1)
        adv_t = self._advantages.unsqueeze(-1)   # (T, n_agents, 1)
        ret_t = self._returns.unsqueeze(-1)      # (T, n_agents, 1)
        mask_t = (1.0 - torch.tensor(self.dones, dtype=torch.float32)).unsqueeze(-1)

        all_chunks = []
        for agent_idx in range(n_agents):
            for start in range(0, n - chunk_len + 1, chunk_len):
                end = start + chunk_len
                ah, ac = self.actor_hiddens[start]
                ch, cc = self.critic_hiddens[start]

                chunk = ChunkBatch(
                    obs=obs_t[start:end, agent_idx].to(self.device),
                    joint_obs=joint_t[start:end].to(self.device),
                    actions=act_t[start:end, agent_idx].to(self.device),
                    old_log_probs=logp_t[start:end, agent_idx].to(self.device),
                    advantages=adv_t[start:end, agent_idx].to(self.device),
                    returns=ret_t[start:end, agent_idx].to(self.device),
                    masks=mask_t[start:end].to(self.device),
                    actor_h=ah[:, agent_idx:agent_idx+1, :].to(self.device),
                    actor_c=ac[:, agent_idx:agent_idx+1, :].to(self.device),
                    critic_h=ch.to(self.device),
                    critic_c=cc.to(self.device),
                )
                all_chunks.append(chunk)
        return all_chunks

    @staticmethod
    def _batch_chunks(chunks: list, batch_size: int) -> Generator[ChunkBatch, None, None]:
        """청크 셔플 후 미니배치로 묶어 yield"""
        if not chunks:
            return
        indices = np.random.permutation(len(chunks))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch = [chunks[j] for j in batch_idx]
            yield ChunkBatch(
                obs=torch.stack([b.obs for b in batch]),
                joint_obs=torch.stack([b.joint_obs for b in batch]),
                actions=torch.stack([b.actions for b in batch]),
                old_log_probs=torch.stack([b.old_log_probs for b in batch]),
                advantages=torch.stack([b.advantages for b in batch]),
                returns=torch.stack([b.returns for b in batch]),
                masks=torch.stack([b.masks for b in batch]),
                actor_h=torch.cat([b.actor_h for b in batch], dim=1),
                actor_c=torch.cat([b.actor_c for b in batch], dim=1),
                critic_h=torch.cat([b.critic_h for b in batch], dim=1),
                critic_c=torch.cat([b.critic_c for b in batch], dim=1),
            )

    def get_chunks(self) -> Generator[ChunkBatch, None, None]:
        """단일 버퍼 청크 미니배치 (기존 호환)"""
        all_chunks = self._build_chunks()
        yield from self._batch_chunks(all_chunks, self.cfg.mini_batch_chunks)

    @staticmethod
    def combined_chunks(buffers: list, cfg) -> Generator[ChunkBatch, None, None]:
        """
        여러 버퍼의 청크를 합쳐서 셔플/배치 (멀티 환경용)
        각 버퍼는 독립 환경의 시간적으로 연속된 데이터를 담고 있음
        """
        all_chunks = []
        for buf in buffers:
            all_chunks.extend(buf._build_chunks())
        yield from RolloutBuffer._batch_chunks(all_chunks, cfg.mini_batch_chunks)
