"""
MAPPO Optuna 하이퍼파라미터 튜닝
================================
축소 학습(500K steps)으로 빠르게 탐색 → 베스트 파라미터로 풀 학습.

사용법:
  # 빌드 환경 필수 (trial마다 환경 자동 재시작)
  python -m mappo.optuna_tuner --env-path ./Build/BoatAttack.exe

  # 옵션
  python -m mappo.optuna_tuner --env-path ./Build/BoatAttack.exe \
      --n-trials 30 --tuning-steps 500000 --no-graphics

  # DB 저장 (중단 후 이어서 가능)
  python -m mappo.optuna_tuner --env-path ./Build/BoatAttack.exe \
      --db-path results/optuna/study.db
"""

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("[ERROR] optuna 패키지가 필요합니다: pip install optuna")
    sys.exit(1)

from .config import MAPPOConfig
from .env_wrapper import UnityEnvWrapper
from .vec_env_wrapper import SingleEnvAdapter
from .mappo_trainer import MAPPOTrainer


# ═══════════════════════════════════════════════════════
# Hyperparameter Sampling
# ═══════════════════════════════════════════════════════

def create_config(trial: optuna.Trial, args) -> MAPPOConfig:
    """Optuna trial에서 하이퍼파라미터 샘플링"""

    # ── 탐색 대상 (영향도 높은 순) ──
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    entropy_coef = trial.suggest_float("entropy_coef", 0.001, 0.05, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    entity_embed_dim = trial.suggest_categorical("entity_embed_dim", [16, 32, 64])
    attention_heads = trial.suggest_categorical("attention_heads", [1, 2, 4])

    return MAPPOConfig(
        # Tuned
        lr=lr,
        entropy_coef=entropy_coef,
        gamma=gamma,
        clip_eps=clip_eps,
        hidden_dim=hidden_dim,
        entity_embed_dim=entity_embed_dim,
        attention_heads=attention_heads,
        # Fixed
        use_entity_attention=True,
        total_steps=args.tuning_steps,
        rollout_steps=args.rollout_steps,
        chunk_length=args.chunk_length,
        num_epochs=3,
        log_dir=os.path.join(args.log_dir, f"trial_{trial.number}"),
        log_interval=999999,   # tuning 중 콘솔 로그 최소화
        save_interval=999999,  # tuning 중 체크포인트 저장 안 함
    )


# ═══════════════════════════════════════════════════════
# Objective Function
# ═══════════════════════════════════════════════════════

def objective(trial: optuna.Trial, args) -> float:
    """
    1 trial = 축소 학습 → 평균 보상 반환
    MedianPruner가 나쁜 trial을 조기 종료
    """
    cfg = create_config(trial, args)

    print(f"\n{'='*60}")
    print(f"  Trial {trial.number}")
    print(f"  lr={cfg.lr:.6f}  entropy={cfg.entropy_coef:.4f}  "
          f"gamma={cfg.gamma:.4f}  clip={cfg.clip_eps:.2f}")
    print(f"  hidden={cfg.hidden_dim}  embed={cfg.entity_embed_dim}  "
          f"heads={cfg.attention_heads}")
    print(f"{'='*60}")

    env = None
    try:
        # 환경 연결
        env = UnityEnvWrapper(
            cfg,
            file_name=args.env_path,
            worker_id=args.worker_id,
            no_graphics=args.no_graphics,
        )
        vec_env = SingleEnvAdapter(env)
        trainer = MAPPOTrainer(cfg, vec_env)

        episode_rewards = []
        report_interval = max(1, args.report_every)
        start_time = time.time()

        # ── 학습 루프 (trainer.train() 대신 직접 제어) ──
        while trainer.total_steps < cfg.total_steps:
            ep_stats = trainer.collect_rollout()
            losses = trainer.update()
            mean_reward = ep_stats["mean_reward"]

            if mean_reward != 0:
                episode_rewards.append(mean_reward)

            # Optuna에 중간 결과 보고
            if trainer.total_updates % report_interval == 0 and episode_rewards:
                avg_reward = float(np.mean(episode_rewards[-30:]))
                trial.report(avg_reward, step=trainer.total_steps)

                elapsed = time.time() - start_time
                fps = trainer.total_steps / max(elapsed, 1)
                print(f"  [T{trial.number}] step={trainer.total_steps:>7,d}  "
                      f"reward={avg_reward:>7.2f}  "
                      f"actor={losses['actor_loss']:.4f}  "
                      f"fps={fps:.0f}")

                # Pruning 체크
                if trial.should_prune():
                    print(f"  [T{trial.number}] PRUNED at step {trainer.total_steps:,d}")
                    raise optuna.TrialPruned()

        # 최종 평가 지표
        final_reward = float(np.mean(episode_rewards[-30:])) if episode_rewards else 0.0
        elapsed = time.time() - start_time
        print(f"  [T{trial.number}] DONE  reward={final_reward:.2f}  "
              f"time={elapsed:.0f}s  episodes={len(episode_rewards)}")

        return final_reward

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"  [T{trial.number}] ERROR: {e}")
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════

def print_results(study: optuna.Study, args):
    """튜닝 결과 출력 및 저장"""

    print(f"\n{'='*60}")
    print("  Optuna Tuning Complete!")
    print(f"{'='*60}")

    # 전체 trial 요약
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"\n  Trials: {len(study.trials)} total")
    print(f"    Completed: {len(completed)}")
    print(f"    Pruned:    {len(pruned)}")
    print(f"    Failed:    {len(failed)}")

    if not completed:
        print("\n  [!] 완료된 trial이 없습니다.")
        return

    # Best trial
    best = study.best_trial
    print(f"\n  Best Trial: #{best.number}")
    print(f"  Best Reward: {best.value:.2f}")
    print(f"\n  Best Hyperparameters:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")

    # Top 5 trials
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)
    print(f"\n  Top 5 Trials:")
    print(f"  {'#':>4}  {'Reward':>8}  {'lr':>10}  {'entropy':>10}  "
          f"{'gamma':>7}  {'hidden':>7}  {'embed':>6}")
    print(f"  {'-'*60}")
    for t in sorted_trials[:5]:
        p = t.params
        print(f"  {t.number:>4}  {t.value:>8.2f}  {p['lr']:>10.6f}  "
              f"{p['entropy_coef']:>10.4f}  {p['gamma']:>7.4f}  "
              f"{p.get('hidden_dim', '?'):>7}  {p.get('entity_embed_dim', '?'):>6}")

    # 결과 저장
    os.makedirs(args.log_dir, exist_ok=True)

    # JSON 저장
    results = {
        "best_trial": best.number,
        "best_reward": best.value,
        "best_params": best.params,
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)}
            for t in study.trials
        ],
    }
    json_path = os.path.join(args.log_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 풀 학습 커맨드 출력
    bp = best.params
    print(f"\n  Full training command:")
    print(f"  python -m mappo.train --env-path {args.env_path} \\")
    print(f"      --lr {bp['lr']:.6f} \\")
    print(f"      --entropy-coef {bp['entropy_coef']:.4f} \\")
    print(f"      --gamma {bp['gamma']:.4f} \\")
    print(f"      --clip-eps {bp['clip_eps']:.2f} \\")
    print(f"      --hidden-dim {bp['hidden_dim']} \\")
    print(f"      --entity-embed-dim {bp['entity_embed_dim']} \\")
    print(f"      --attention-heads {bp['attention_heads']} \\")
    print(f"      --total-steps 5000000")

    print(f"\n  Results saved: {json_path}")


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="MAPPO Optuna Hyperparameter Tuning")

    # Environment
    parser.add_argument("--env-path", type=str, default=None,
                        help="Unity 빌드 경로 (빌드 권장, None=Editor)")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--no-graphics", action="store_true",
                        help="렌더링 비활성화 (속도 향상)")

    # Tuning
    parser.add_argument("--n-trials", type=int, default=30,
                        help="총 trial 수 (기본 30)")
    parser.add_argument("--tuning-steps", type=int, default=500_000,
                        help="trial당 학습 스텝 (기본 500K)")
    parser.add_argument("--report-every", type=int, default=5,
                        help="Optuna 보고 주기 (업데이트 횟수)")

    # Training (fixed across trials)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--chunk-length", type=int, default=64)

    # Pruning
    parser.add_argument("--startup-trials", type=int, default=5,
                        help="pruning 시작 전 최소 trial 수")
    parser.add_argument("--warmup-steps", type=int, default=100_000,
                        help="pruning 시작 전 최소 스텝 수")

    # Storage
    parser.add_argument("--log-dir", type=str, default="results/optuna")
    parser.add_argument("--study-name", type=str, default="mappo_defense")
    parser.add_argument("--db-path", type=str, default=None,
                        help="SQLite DB 경로 (중단 후 이어서 가능)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.env_path is None:
        print("[!] Editor 연결은 trial 간 수동 재시작 필요 → --env-path (빌드) 권장")

    # Optuna Study 생성
    storage = f"sqlite:///{args.db_path}" if args.db_path else None

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=args.startup_trials,
            n_warmup_steps=args.warmup_steps,
        ),
        storage=storage,
        load_if_exists=True,
    )

    print("=" * 60)
    print("  MAPPO Optuna Hyperparameter Tuning")
    print("=" * 60)
    print(f"  Trials: {args.n_trials}")
    print(f"  Steps/trial: {args.tuning_steps:,}")
    print(f"  Pruning: after {args.startup_trials} trials, {args.warmup_steps:,} steps")
    print(f"  Build: {args.env_path or 'Editor'}")
    print(f"  Storage: {args.db_path or 'in-memory'}")
    print("=" * 60)

    # Ctrl+C 시 중간 결과 출력
    def signal_handler(sig, frame):
        print("\n[!] 튜닝 중단 — 현재까지 결과 출력...")
        print_results(study, args)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # 튜닝 실행
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
    )

    # 결과 출력
    print_results(study, args)


if __name__ == "__main__":
    main()
