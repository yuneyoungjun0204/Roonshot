"""MAPPO 하이퍼파라미터 설정"""

from dataclasses import dataclass, field


@dataclass
class MAPPOConfig:
    # ── Environment ──
    obs_dim: int = 10          # 에이전트당 관측 차원
    stack_frames: int = 3      # 프레임 스태킹 수
    action_dim: int = 2        # 연속 액션 차원 (throttle, steering)
    n_agents: int = 2          # 에이전트 수

    # ── Network ──
    hidden_dim: int = 256      # MLP/LSTM hidden 크기 (ML-Agents: 256)
    lstm_layers: int = 1       # LSTM 레이어 수
    actor_layers: list = field(default_factory=lambda: [256, 128])
    critic_layers: list = field(default_factory=lambda: [256, 128])
    normalize_obs: bool = True # 관측 정규화 (ML-Agents: normalize=true)

    # ── Entity Attention ──
    use_entity_attention: bool = True   # False면 기존 flat LSTM 사용
    self_dim: int = 2                   # 자신: prev_throttle, prev_steering
    partner_dim: int = 3                # 팀원: rel_right, rel_forward, heading_diff
    enemy_dim: int = 3                  # 적: rel_right, rel_forward, heading_diff
    mothership_dim: int = 2             # 모선: rel_right, rel_forward
    max_enemies: int = 1                # 최대 적 수 (5로 변경 시 obs_dim도 변경)
    entity_embed_dim: int = 32          # 엔티티 임베딩 차원
    attention_heads: int = 2            # 어텐션 헤드 수

    # ── PPO ──
    gamma: float = 0.99        # 할인율 (ML-Agents: 0.99)
    gae_lambda: float = 0.95   # GAE lambda (ML-Agents: 0.95)
    clip_eps: float = 0.2      # PPO 클리핑 (ML-Agents: 0.2)
    value_coef: float = 0.5    # Value loss 계수
    entropy_coef: float = 0.00115  # 엔트로피 (ML-Agents beta: 0.00115)
    max_grad_norm: float = 0.5 # 그래디언트 클리핑

    # ── Training ──
    lr: float = 1.13e-4        # 학습률 (ML-Agents: 0.000113)
    lr_schedule: str = "linear" # "linear" or "constant" (ML-Agents: linear)
    num_envs: int = 1          # 병렬 Unity 환경 수 (빌드 필수)
    rollout_steps: int = 2048  # 롤아웃 수집 총 스텝 (num_envs로 균등 분배)
    buffer_size: int = 20480   # 버퍼 크기 (ML-Agents: 20480)
    chunk_length: int = 64     # LSTM BPTT 청크 (ML-Agents sequence_length: 64)
    time_horizon: int = 128    # 타임 호라이즌 (ML-Agents: 128)
    num_epochs: int = 3        # PPO 에포크 수 (ML-Agents: 3)
    mini_batch_chunks: int = 4 # 미니배치 당 청크 수
    total_steps: int = 25_000_000  # (ML-Agents: 25000000)

    # ── Logging ──
    log_interval: int = 5      # 업데이트 주기마다 로그
    save_interval: int = 50    # 업데이트 주기마다 모델 저장
    log_dir: str = "results/mappo"

    # ── Runtime (Unity 연결 후 자동 설정) ──
    stacked_obs_dim_override: int = 0  # 0이면 obs_dim * stack_frames 사용

    @property
    def stacked_obs_dim(self) -> int:
        if self.stacked_obs_dim_override > 0:
            return self.stacked_obs_dim_override
        return self.obs_dim * self.stack_frames
