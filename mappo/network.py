"""MAPPO 네트워크: Entity Attention + LSTM Actor/Critic

아키텍처:
  관측(90) → reshape [9 frames, 10 obs/frame]
  → 프레임별 Entity Attention (self/partner/enemy/mothership)
  → [9, embed_dim] → concat → projection → LSTM → MLP → action

Entity 구조 (obs_dim=10, max_enemies=1):
  - Self(2): prev_throttle, prev_steering
  - Partner(3): rel_right, rel_forward, heading_diff
  - Enemy(3): rel_right, rel_forward, heading_diff
  - Mothership(2): rel_right, rel_forward
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .config import MAPPOConfig


def build_mlp(input_dim: int, hidden_layers: list, output_dim: int) -> nn.Sequential:
    """MLP 빌더"""
    layers = []
    prev = input_dim
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════
# Entity Attention Components
# ═══════════════════════════════════════════════════════

class EntityEncoder(nn.Module):
    """엔티티 타입별 인코더: 가변 차원 → 공통 임베딩"""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EntityAttentionBlock(nn.Module):
    """
    ONNX 호환 Multi-Head Self-Attention
    (nn.MultiheadAttention 대신 수동 구현 → opset 11 지원)
    """

    def __init__(self, embed_dim: int, n_heads: int = 2):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """
        entities: (batch, n_entities, embed_dim)
        Returns: (batch, n_entities, embed_dim)
        """
        B, N, D = entities.shape
        H = self.n_heads
        Dh = self.head_dim

        Q = self.q_proj(entities).reshape(B, N, H, Dh).transpose(1, 2)
        K = self.k_proj(entities).reshape(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(entities).reshape(B, N, H, Dh).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (Dh ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, N, Dh)

        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        return self.norm(entities + out)


class FrameEntityEncoder(nn.Module):
    """
    단일 프레임 관측 → 엔티티 분리 → 인코딩 → 어텐션 → 풀링
    적 엔티티는 encoder를 공유 (동일 타입이므로)
    """

    def __init__(self, cfg: MAPPOConfig):
        super().__init__()
        embed_dim = cfg.entity_embed_dim

        # 엔티티 타입별 인코더
        self.self_enc = EntityEncoder(cfg.self_dim, embed_dim)
        self.partner_enc = EntityEncoder(cfg.partner_dim, embed_dim)
        self.enemy_enc = EntityEncoder(cfg.enemy_dim, embed_dim)  # 적 공유
        self.mothership_enc = EntityEncoder(cfg.mothership_dim, embed_dim)

        self.max_enemies = cfg.max_enemies
        self.self_dim = cfg.self_dim
        self.partner_dim = cfg.partner_dim
        self.enemy_dim = cfg.enemy_dim
        self.mothership_dim = cfg.mothership_dim

        # n_entities = 1(self) + 1(partner) + max_enemies + 1(mothership)
        self.n_entities = 1 + 1 + cfg.max_enemies + 1

        # Self-Attention
        self.attention = EntityAttentionBlock(embed_dim, cfg.attention_heads)
        self.embed_dim = embed_dim

    def forward(self, obs_frame: torch.Tensor) -> torch.Tensor:
        """
        obs_frame: (batch, obs_dim)
        Returns: (batch, embed_dim)
        """
        entities = []
        offset = 0

        # Self
        entities.append(self.self_enc(obs_frame[:, offset:offset + self.self_dim]))
        offset += self.self_dim

        # Partner
        entities.append(self.partner_enc(obs_frame[:, offset:offset + self.partner_dim]))
        offset += self.partner_dim

        # Enemies (shared encoder)
        for _ in range(self.max_enemies):
            entities.append(self.enemy_enc(obs_frame[:, offset:offset + self.enemy_dim]))
            offset += self.enemy_dim

        # Mothership
        entities.append(self.mothership_enc(obs_frame[:, offset:offset + self.mothership_dim]))

        # (batch, n_entities, embed_dim)
        entities = torch.stack(entities, dim=1)

        # Self-Attention → residual + LayerNorm
        attended = self.attention(entities)

        # Mean pooling → (batch, embed_dim)
        return attended.mean(dim=1)


class ObservationEncoder(nn.Module):
    """
    스택된 관측 → 프레임별 Entity Attention → 연결 → 투영

    Input:  (batch, stacked_obs_dim)  예: (B, 90)
    Output: (batch, output_dim)       예: (B, 128)

    내부:
      reshape (B, 9, 10) → FrameEntityEncoder(공유) per frame
      → (B, 9, 32) → concat (B, 288) → Linear → (B, 128)
    """

    def __init__(self, cfg: MAPPOConfig):
        super().__init__()
        self.obs_dim = cfg.obs_dim
        self.n_frames = cfg.stacked_obs_dim // cfg.obs_dim
        embed_dim = cfg.entity_embed_dim

        # 프레임 인코더 (모든 프레임이 파라미터 공유)
        self.frame_encoder = FrameEntityEncoder(cfg)

        # 프레임 특징 연결 → LSTM 입력 투영
        self.output_dim = cfg.hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(self.n_frames * embed_dim, self.output_dim),
            nn.ReLU(),
        )

    def _encode(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """obs_flat: (batch, stacked_obs_dim) → (batch, output_dim)"""
        B = obs_flat.shape[0]

        # (B, n_frames, obs_dim)
        frames = obs_flat.reshape(B, self.n_frames, self.obs_dim)

        # 모든 프레임을 한번에 배치 처리
        frames_flat = frames.reshape(B * self.n_frames, self.obs_dim)
        frame_feat = self.frame_encoder(frames_flat)  # (B*n_frames, embed_dim)

        # (B, n_frames * embed_dim)
        concat = frame_feat.reshape(B, -1)

        return self.proj(concat)

    def forward(self, stacked_obs: torch.Tensor) -> torch.Tensor:
        """
        stacked_obs: (batch, stacked_obs_dim) 또는 (batch, seq_len, stacked_obs_dim)
        Returns: same leading dims + output_dim
        """
        if stacked_obs.dim() == 3:
            B, S, D = stacked_obs.shape
            out = self._encode(stacked_obs.reshape(B * S, D))
            return out.reshape(B, S, -1)
        return self._encode(stacked_obs)


# ═══════════════════════════════════════════════════════
# Actor / Critic
# ═══════════════════════════════════════════════════════

class Actor(nn.Module):
    """
    정책 네트워크 (파라미터 공유: 모든 에이전트가 동일 네트워크 사용)

    use_entity_attention=True:
      obs → EntityAttention → LSTM → MLP → Normal
    use_entity_attention=False:
      obs → LSTM → MLP → Normal (기존 flat 방식)
    """

    def __init__(self, cfg: MAPPOConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_entity_attention:
            self.obs_encoder = ObservationEncoder(cfg)
            lstm_input = self.obs_encoder.output_dim
        else:
            self.obs_encoder = None
            lstm_input = cfg.stacked_obs_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        self.mean_head = build_mlp(cfg.hidden_dim, cfg.actor_layers, cfg.action_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

    def init_hidden(self, batch_size: int = 1, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.cfg.lstm_layers, batch_size, self.cfg.hidden_dim, device=device)
        c = torch.zeros(self.cfg.lstm_layers, batch_size, self.cfg.hidden_dim, device=device)
        return (h, c)

    def forward(self, obs: torch.Tensor, hidden: tuple):
        """
        Args:
            obs: (batch, seq_len, stacked_obs_dim) 또는 (batch, stacked_obs_dim)
            hidden: (h, c) LSTM hidden state
        Returns:
            mean, log_std, new_hidden
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)

        # Entity Attention 인코딩
        if self.obs_encoder is not None:
            encoded = self.obs_encoder(obs)
        else:
            encoded = obs

        lstm_out, new_hidden = self.lstm(encoded, hidden)
        mean = self.mean_head(lstm_out)
        log_std = self.log_std.expand_as(mean)

        return mean, log_std, new_hidden

    def get_distribution(self, obs: torch.Tensor, hidden: tuple):
        mean, log_std, new_hidden = self.forward(obs, hidden)
        std = log_std.exp().clamp(min=1e-6)
        return Normal(mean, std), new_hidden

    def act(self, obs: torch.Tensor, hidden: tuple, deterministic: bool = False):
        """
        단일 스텝 액션 샘플링
        Args:
            obs: (n_agents, stacked_obs_dim)
            hidden: (h, c) shape (layers, n_agents, hidden_dim)
        Returns:
            actions, log_probs, new_hidden
        """
        dist, new_hidden = self.get_distribution(obs, hidden)

        if deterministic:
            actions = dist.mean.squeeze(1)
        else:
            actions = dist.rsample().squeeze(1)

        log_probs = dist.log_prob(actions.unsqueeze(1)).squeeze(1)
        log_probs = log_probs.sum(dim=-1, keepdim=True)
        actions = actions.clamp(-1.0, 1.0)

        return actions, log_probs, new_hidden

    def evaluate(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor, hidden: tuple):
        """
        학습 시 시퀀스 평가
        Args:
            obs_seq: (batch, seq_len, stacked_obs_dim)
            actions_seq: (batch, seq_len, action_dim)
            hidden: 청크 시작 시점의 hidden state
        Returns:
            log_probs, entropy, new_hidden
        """
        dist, new_hidden = self.get_distribution(obs_seq, hidden)
        log_probs = dist.log_prob(actions_seq).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_probs, entropy, new_hidden


class CentralizedCritic(nn.Module):
    """
    중앙화 비평가 (MAPPO)

    use_entity_attention=True:
      per-agent obs → EntityAttention(공유) → concat → LSTM → MLP → value
    use_entity_attention=False:
      joint obs → LSTM → MLP → value (기존 flat 방식)
    """

    def __init__(self, cfg: MAPPOConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.use_entity_attention:
            self.obs_encoder = ObservationEncoder(cfg)
            # 에이전트별 인코딩 후 concat
            lstm_input = self.obs_encoder.output_dim * cfg.n_agents
        else:
            self.obs_encoder = None
            lstm_input = cfg.stacked_obs_dim * cfg.n_agents

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
            batch_first=True,
        )

        self.value_head = build_mlp(cfg.hidden_dim, cfg.critic_layers, 1)

    def init_hidden(self, batch_size: int = 1, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(self.cfg.lstm_layers, batch_size, self.cfg.hidden_dim, device=device)
        c = torch.zeros(self.cfg.lstm_layers, batch_size, self.cfg.hidden_dim, device=device)
        return (h, c)

    def forward(self, joint_obs: torch.Tensor, hidden: tuple):
        """
        Args:
            joint_obs: (batch, seq_len, joint_obs_dim) 또는 (batch, joint_obs_dim)
            hidden: (h, c)
        Returns:
            values: (batch, seq_len, 1), new_hidden
        """
        if joint_obs.dim() == 2:
            joint_obs = joint_obs.unsqueeze(1)

        if self.obs_encoder is not None:
            stacked_dim = self.cfg.stacked_obs_dim
            B, S, D = joint_obs.shape

            # 에이전트별로 분리 → 각각 EntityAttention → concat
            agent_features = []
            for i in range(self.cfg.n_agents):
                agent_obs = joint_obs[:, :, i * stacked_dim:(i + 1) * stacked_dim]
                feat = self.obs_encoder(agent_obs)  # (B, S, output_dim)
                agent_features.append(feat)

            encoded = torch.cat(agent_features, dim=-1)  # (B, S, n_agents * output_dim)
        else:
            encoded = joint_obs

        lstm_out, new_hidden = self.lstm(encoded, hidden)
        values = self.value_head(lstm_out)
        return values, new_hidden
