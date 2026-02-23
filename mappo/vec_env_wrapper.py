"""벡터화 환경 래퍼: N개 Unity 인스턴스 병렬 관리

사용법:
  # 빌드 환경 4개 병렬
  vec_env = VecEnvWrapper(cfg, file_name="./Build/BoatAttack.exe", num_envs=4)

  # Editor 단일 환경 (기존 호환)
  env = UnityEnvWrapper(cfg)
  vec_env = SingleEnvAdapter(env)

병렬화 원리:
  - 각 Unity 빌드는 별도 프로세스 (worker_id로 포트 분리)
  - ThreadPoolExecutor로 env.step() 동시 실행
  - gRPC I/O 대기 중 Python GIL 해제 → 실질적 병렬 시뮬레이션
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from .config import MAPPOConfig
from .env_wrapper import UnityEnvWrapper


class VecEnvWrapper:
    """
    N개 Unity 빌드 환경을 병렬 관리

    Requirements:
        - Unity 빌드 (.exe) 필수 (Editor는 단일 환경만 가능)
        - --no-graphics 권장 (헤드리스 학습으로 CPU/GPU 절약)
    """

    def __init__(self, cfg: MAPPOConfig, file_name: str,
                 num_envs: int = 1, base_worker_id: int = 0,
                 no_graphics: bool = True):
        assert file_name is not None, \
            "VecEnvWrapper requires built environment (--env-path). " \
            "Editor mode only supports single env."
        assert num_envs >= 1

        self.cfg = cfg
        self.num_envs = num_envs
        self.envs: List[UnityEnvWrapper] = []

        print(f"[VecEnv] Launching {num_envs} Unity environments...")
        for i in range(num_envs):
            wid = base_worker_id + i
            print(f"  [Env {i}] worker_id={wid}")
            env = UnityEnvWrapper(
                cfg, file_name=file_name,
                worker_id=wid, no_graphics=no_graphics,
            )
            self.envs.append(env)

        self.unity_obs_dim = self.envs[0].unity_obs_dim
        self._executor = ThreadPoolExecutor(max_workers=num_envs)
        print(f"[VecEnv] Ready: {num_envs} envs, obs_dim={self.unity_obs_dim}")

    def reset_all(self) -> List[np.ndarray]:
        """모든 환경 병렬 리셋
        Returns: list of (n_agents, unity_obs_dim) arrays
        """
        futures = [self._executor.submit(env.reset) for env in self.envs]
        return [f.result() for f in futures]

    def step_all(self, actions_list: List[np.ndarray]) -> List[Tuple]:
        """모든 환경 병렬 스텝
        Args:
            actions_list: len=num_envs, each (n_agents, action_dim)
        Returns:
            list of (obs, rewards, done, info) tuples
        """
        futures = [
            self._executor.submit(env.step, actions)
            for env, actions in zip(self.envs, actions_list)
        ]
        return [f.result() for f in futures]

    def close(self):
        """모든 환경 종료"""
        self._executor.shutdown(wait=False)
        for i, env in enumerate(self.envs):
            try:
                env.close()
            except Exception as e:
                print(f"[VecEnv] Env {i} close error: {e}")


class SingleEnvAdapter:
    """
    단일 UnityEnvWrapper를 VecEnvWrapper 인터페이스로 래핑
    - Editor 연결 (file_name=None) 시 사용
    - num_envs=1 빌드 환경에도 사용 가능
    """

    def __init__(self, env: UnityEnvWrapper):
        self.envs = [env]
        self.num_envs = 1
        self.unity_obs_dim = env.unity_obs_dim

    def reset_all(self) -> List[np.ndarray]:
        return [self.envs[0].reset()]

    def step_all(self, actions_list: List[np.ndarray]) -> List[Tuple]:
        return [self.envs[0].step(actions_list[0])]

    def close(self):
        self.envs[0].close()
