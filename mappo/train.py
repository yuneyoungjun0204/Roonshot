"""
MAPPO 학습 진입점
=================
Unity Editor 또는 빌드된 환경과 연결하여 MAPPO 학습을 실행합니다.

사용법:
  # Unity Editor 연결 (Editor에서 Play 먼저 누른 후 실행)
  python -m mappo.train

  # 빌드된 환경 사용
  python -m mappo.train --env-path ./Build/BoatAttack.exe

  # 병렬 환경 (빌드 필수, N배 속도 향상)
  python -m mappo.train --env-path ./Build/BoatAttack.exe --num-envs 4 --no-graphics

  # 하이퍼파라미터 오버라이드
  python -m mappo.train --lr 1e-4 --total-steps 10000000 --hidden-dim 256

  # 이어서 학습
  python -m mappo.train --resume results/mappo/checkpoint_100000.pt
"""

import argparse
import signal
import sys

from .config import MAPPOConfig
from .env_wrapper import UnityEnvWrapper
from .vec_env_wrapper import VecEnvWrapper, SingleEnvAdapter
from .mappo_trainer import MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAPPO Training for BoatAttack Defense")

    # Environment
    parser.add_argument("--env-path", type=str, default=None,
                        help="Unity 빌드 경로 (None=Editor 연결)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="병렬 Unity 환경 수 (빌드 필수, 기본 1)")
    parser.add_argument("--worker-id", type=int, default=0,
                        help="시작 worker_id (멀티 환경 시 base)")
    parser.add_argument("--no-graphics", action="store_true",
                        help="렌더링 비활성화 (멀티 환경 시 권장)")

    # Network
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)

    # Entity Attention
    parser.add_argument("--no-entity-attention", action="store_true",
                        help="Entity Attention 비활성화 (flat LSTM 사용)")
    parser.add_argument("--entity-embed-dim", type=int, default=32)
    parser.add_argument("--attention-heads", type=int, default=2)

    # Normalization
    parser.add_argument("--no-normalize", action="store_true",
                        help="관측 정규화 비활성화")

    # PPO (ML-Agents 설정 기준)
    parser.add_argument("--lr", type=float, default=1.13e-4)
    parser.add_argument("--lr-schedule", type=str, default="linear",
                        choices=["linear", "constant"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.00115)
    parser.add_argument("--num-epochs", type=int, default=3)

    # Training
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--chunk-length", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=25_000_000)

    # Logging
    parser.add_argument("--log-dir", type=str, default="results/mappo")
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="체크포인트 경로 (이어서 학습)")

    # Export
    parser.add_argument("--export-onnx", action="store_true",
                        help="학습 완료 후 ONNX 변환")

    return parser.parse_args()


def main():
    args = parse_args()

    # 검증
    if args.num_envs > 1 and args.env_path is None:
        print("[ERROR] 멀티 환경 (--num-envs > 1) 사용 시 --env-path 필수")
        print("  Unity에서 File > Build Settings > Build로 빌드 후 경로 지정")
        sys.exit(1)

    if args.num_envs > 1 and not args.no_graphics:
        print("[WARNING] 멀티 환경 시 --no-graphics 권장 (GPU 절약)")

    # Config 생성
    cfg = MAPPOConfig(
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        use_entity_attention=not args.no_entity_attention,
        entity_embed_dim=args.entity_embed_dim,
        attention_heads=args.attention_heads,
        normalize_obs=not args.no_normalize,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        num_epochs=args.num_epochs,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        chunk_length=args.chunk_length,
        total_steps=args.total_steps,
        log_dir=args.log_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
    )

    print("=" * 60)
    print("  MAPPO Training for BoatAttack Defense")
    print("=" * 60)
    print(f"  Observations: {cfg.obs_dim} × {cfg.stack_frames} stack = {cfg.stacked_obs_dim}")
    print(f"  Actions: {cfg.action_dim} continuous")
    print(f"  Agents: {cfg.n_agents}")
    print(f"  Hidden: {cfg.hidden_dim}, LSTM layers: {cfg.lstm_layers}")
    attn_str = f"embed={cfg.entity_embed_dim}, heads={cfg.attention_heads}" if cfg.use_entity_attention else "OFF"
    print(f"  Entity Attention: {attn_str}")
    print(f"  Normalize: {cfg.normalize_obs}")
    print(f"  LR: {cfg.lr} ({cfg.lr_schedule}), Entropy: {cfg.entropy_coef}")
    print(f"  Gamma: {cfg.gamma}, Clip: {cfg.clip_eps}")
    print(f"  Envs: {cfg.num_envs}, Rollout: {cfg.rollout_steps}")
    print(f"  Total steps: {cfg.total_steps:,}")
    print(f"  Log dir: {cfg.log_dir}")
    print("=" * 60)

    # 환경 생성
    print("\n[*] Unity 환경 연결 중...")
    if args.num_envs > 1:
        vec_env = VecEnvWrapper(
            cfg,
            file_name=args.env_path,
            num_envs=args.num_envs,
            base_worker_id=args.worker_id,
            no_graphics=args.no_graphics,
        )
    else:
        env = UnityEnvWrapper(
            cfg,
            file_name=args.env_path,
            worker_id=args.worker_id,
            no_graphics=args.no_graphics,
        )
        vec_env = SingleEnvAdapter(env)

    # 트레이너 생성
    trainer = MAPPOTrainer(cfg, vec_env)

    # 체크포인트 복원
    if args.resume:
        trainer.load(args.resume)

    # Ctrl+C 시 모델 저장
    def signal_handler(sig, frame):
        print("\n[!] 학습 중단 — 모델 저장 중...")
        trainer.save("interrupted")
        vec_env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 학습 실행
    try:
        trainer.train()
    finally:
        if args.export_onnx:
            trainer.export_onnx()
        vec_env.close()

    print("[*] 완료")


if __name__ == "__main__":
    main()
