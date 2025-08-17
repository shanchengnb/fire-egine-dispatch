import gym
import numpy as np

class ContinuousDispatchEnv(gym.Env):
    """
    🔁 ContinuousDispatchEnv: Wrapper to support continuous action space (suitable for SAC)
    """

    def __init__(self, base_env, max_candidates=10, fallback_on_invalid=True):
        super().__init__()
        self.env = base_env
        self.max_dispatch = self.env.max_dispatch_per_event
        self.max_candidates = max_candidates  # Allows selection from the first N available vehicles at most
        self.fallback_on_invalid = fallback_on_invalid

        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_dispatch,),
            dtype=np.float32
        )

        self.current_actions = []  # The current list of engine_ids sorted by distance

    def reset(self, **kwargs):
        obs_info = self.env.reset(**kwargs)
        obs = obs_info[0] if isinstance(obs_info, tuple) else obs_info
        self._update_action_map()
        return np.asarray(obs, dtype=np.float32)

    def _update_action_map(self):
        """Get the dispatchable vehicles for the current event, sorted by distance"""
        full_sorted = self.env.get_sorted_available_actions()
        self.current_actions = full_sorted[:self.max_candidates]  # Limit to the first N closest vehicles

    def step(self, action_cont):
        self._update_action_map()

        if not self.current_actions:
            obs, reward, done, info = self.env.step([])
            obs = np.asarray(obs, dtype=np.float32)
            info["wrapped_error"] = "no_available_vehicle"
            return obs, -1000.0, True, info

        # 获取当前事件，判断调度数量
        event = self.env.sim.pending_events[0]
        dispatch_count = event.get_required_dispatch_count()

        # 检查并裁剪/补齐动作
        if isinstance(action_cont, (list, tuple, np.ndarray)):
            action_cont = np.clip(np.array(action_cont), 0.0, 1.0)
        else:
            raise ValueError("动作必须是连续数组")

        if len(action_cont) < self.max_candidates:
            padding = np.zeros(self.max_candidates - len(action_cont))
            action_cont = np.concatenate([action_cont, padding])
        elif len(action_cont) > self.max_candidates:
            action_cont = action_cont[:self.max_candidates]

        # 选择 top-K 索引（从最近车辆中）
        top_k = np.argsort(action_cont)[-dispatch_count:][::-1]
        selected_idxs = [i for i in top_k if i < len(self.current_actions)]
        selected_ids = [self.current_actions[i] for i in selected_idxs]

        obs, reward, done, info = self.env.step(selected_ids)
        obs = np.asarray(obs, dtype=np.float32)
        info["wrapped_engine_ids"] = selected_ids
        info["wrapped_action_idxs"] = selected_idxs
        return obs, reward, done, info

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
