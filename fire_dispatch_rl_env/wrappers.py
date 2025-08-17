import gym
import numpy as np
from .ContinuousDispatchEnv import ContinuousDispatchEnv

class WrappedDispatchEnv(gym.Env):
    """
    WrappedDispatchEnv：用于训练/评估统一化的包装器

    特点：
    - 固定动作空间 Discrete(N)，例如 Discrete(20)
    - 动作表示“在当前按距离排序的可调度车辆列表中第k个”
    - 索引越界时自动 fallback 到第一个合法动作
    """

    def __init__(self, base_env, max_actions=20, fallback_on_invalid=True):
        super().__init__()
        self.env = base_env
        self.max_actions = max_actions
        self.fallback_on_invalid = fallback_on_invalid

        self.action_space = gym.spaces.Discrete(max_actions)
        self.observation_space = self.env.observation_space

        self.current_actions = []

    def reset(self, **kwargs):
        obs_info = self.env.reset(**kwargs)
        if isinstance(obs_info, tuple) and len(obs_info) == 2:
            obs, _ = obs_info
        else:
            obs = obs_info
        self._update_action_map()
        return np.asarray(obs, dtype=np.float32)

    def step(self, action_idxs):
        self._update_action_map()

        if not self.current_actions:
            obs, reward, done, info = self.env.step(0)
            obs = np.asarray(obs, dtype=np.float32)
            info["wrapped_error"] = "no_available_vehicle"
            return obs, -1000.0, True, info

        if isinstance(action_idxs, int):
            action_idxs = [action_idxs]
        elif isinstance(action_idxs, np.ndarray):
            action_idxs = action_idxs.tolist()

        # 获取当前事件风险级别
        event = self.env.sim.pending_events[0]
        dispatch_count = event.get_required_dispatch_count()

        # 重复单个动作索引以构成 dispatch_count 个调度（如派两个一样的动作）
        if len(action_idxs) < dispatch_count:
            action_idxs = (action_idxs * dispatch_count)[:dispatch_count]
        else:
            action_idxs = action_idxs[:dispatch_count]

        selected_ids = []
        for idx in action_idxs:
            if idx < len(self.current_actions):
                selected_ids.append(self.current_actions[idx])
            else:
                if self.fallback_on_invalid:
                    selected_ids.append(self.current_actions[0])
                else:
                    obs = get_observation(self.env.sim)
                    obs = np.asarray(obs, dtype=np.float32)
                    return obs, -1000.0, False, {"error": "invalid_action_index"}

        obs, reward, done, info = self.env.step(selected_ids)
        obs = np.asarray(obs, dtype=np.float32)
        info["wrapped_engine_ids"] = selected_ids
        info["wrapped_action_idxs"] = action_idxs
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
