"""Unity ML-Agents 환경 래퍼 (Unity 측 스태킹 사용)"""

import numpy as np
from typing import Dict, Tuple, Optional

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple

from .config import MAPPOConfig


class UnityEnvWrapper:
    """
    Unity ML-Agents Low-Level API 래퍼
    - 2 에이전트 관측/보상/액션 관리
    - Unity Inspector의 Stacked Vectors가 이미 프레임 스태킹 처리
    - 그룹 보상 + 개별 보상 합산
    """

    def __init__(self, cfg: MAPPOConfig, file_name: Optional[str] = None,
                 worker_id: int = 0, no_graphics: bool = False):
        self.cfg = cfg
        self.n_agents = cfg.n_agents

        # Unity 환경 연결
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=worker_id,
            no_graphics=no_graphics,
        )
        self.env.reset()

        # Behavior 이름 및 스펙 확인
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        # Unity에서 보내주는 실제 관측 크기 자동 감지
        self.unity_obs_dim = self.spec.observation_specs[0].shape[0]

        print(f"[EnvWrapper] Behavior: {self.behavior_name}")
        print(f"[EnvWrapper] Unity obs dim: {self.unity_obs_dim} "
              f"(base {cfg.obs_dim} × {self.unity_obs_dim // cfg.obs_dim} stack)")
        print(f"[EnvWrapper] Action spec: continuous={self.spec.action_spec.continuous_size}")

        # Agent ID → 인덱스 매핑
        self._agent_id_to_idx: Dict[int, int] = {}

    def _build_agent_mapping(self, agent_ids: np.ndarray):
        """Agent ID를 0, 1 인덱스에 매핑"""
        for i, aid in enumerate(sorted(agent_ids)):
            if i < self.n_agents:
                self._agent_id_to_idx[aid] = i

    def reset(self) -> np.ndarray:
        """
        환경 리셋
        Returns:
            obs: (n_agents, unity_obs_dim)
        """
        self.env.reset()
        self._agent_id_to_idx.clear()

        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self._build_agent_mapping(decision_steps.agent_id)

        obs = np.zeros((self.n_agents, self.unity_obs_dim), dtype=np.float32)

        for agent_id in decision_steps.agent_id:
            idx = self._agent_id_to_idx.get(agent_id, None)
            if idx is None or idx >= self.n_agents:
                continue
            obs[idx] = decision_steps[agent_id].obs[0]

        return obs

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        환경 스텝
        Args:
            actions: (n_agents, action_dim)
        Returns:
            obs: (n_agents, unity_obs_dim)
            rewards: (n_agents,)
            done: bool
            info: dict
        """
        # 액션 전송
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        action_array = np.zeros((len(decision_steps), self.cfg.action_dim), dtype=np.float32)
        for agent_id in decision_steps.agent_id:
            idx = self._agent_id_to_idx.get(agent_id, None)
            if idx is not None and idx < self.n_agents:
                agent_list_idx = list(decision_steps.agent_id).index(agent_id)
                action_array[agent_list_idx] = actions[idx]

        self.env.set_actions(self.behavior_name, ActionTuple(continuous=action_array))
        self.env.step()

        # 결과 수집
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        obs = np.zeros((self.n_agents, self.unity_obs_dim), dtype=np.float32)
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        individual_rewards = np.zeros(self.n_agents, dtype=np.float32)
        group_rewards = np.zeros(self.n_agents, dtype=np.float32)
        done = False
        info = {}

        # 에피소드 종료 처리
        if len(terminal_steps) > 0:
            done = True
            for agent_id in terminal_steps.agent_id:
                idx = self._agent_id_to_idx.get(agent_id, None)
                if idx is None or idx >= self.n_agents:
                    continue
                obs[idx] = terminal_steps[agent_id].obs[0]
                ind_r = terminal_steps[agent_id].reward
                grp_r = terminal_steps[agent_id].group_reward
                individual_rewards[idx] = ind_r
                group_rewards[idx] = grp_r
                rewards[idx] = ind_r + grp_r

            info["terminal_rewards"] = rewards.copy()
            info["terminal_individual_rewards"] = individual_rewards.copy()
            info["terminal_group_rewards"] = group_rewards.copy()

            # 자동 리셋 후 새 decision_steps 수집
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            self._agent_id_to_idx.clear()

            if len(decision_steps) > 0:
                self._build_agent_mapping(decision_steps.agent_id)
                for agent_id in decision_steps.agent_id:
                    idx = self._agent_id_to_idx.get(agent_id, None)
                    if idx is not None and idx < self.n_agents:
                        obs[idx] = decision_steps[agent_id].obs[0]

        # 진행 중인 에이전트 처리
        else:
            for agent_id in decision_steps.agent_id:
                idx = self._agent_id_to_idx.get(agent_id, None)
                if idx is None or idx >= self.n_agents:
                    continue
                obs[idx] = decision_steps[agent_id].obs[0]
                ind_r = decision_steps[agent_id].reward
                grp_r = decision_steps[agent_id].group_reward
                individual_rewards[idx] = ind_r
                group_rewards[idx] = grp_r
                rewards[idx] = ind_r + grp_r

        info["individual_rewards"] = individual_rewards
        info["group_rewards"] = group_rewards

        return obs, rewards, done, info

    def close(self):
        """환경 종료"""
        self.env.close()
