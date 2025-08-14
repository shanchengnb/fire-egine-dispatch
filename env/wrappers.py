import gym
import numpy as np
from .ContinuousDispatchEnv import ContinuousDispatchEnv

class WrappedDispatchEnv(gym.Env):
    """
    WrappedDispatchEnv: Unified wrapper for training/evaluation

    Features:
    - Fixed action space Discrete(N), e.g., Discrete(20)
    - Action represents "the k-th vehicle in the current sorted list of dispatchable vehicles"
    - Automatically falls back to the first valid action if index is out of bounds
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

        # Get the current event's risk level
        event = self.env.sim.pending_events[0]
        dispatch_count = event.get_required_dispatch_count()

        # Repeat single index to match dispatch_count if needed (e.g. dispatch same vehicle multiple times)
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
