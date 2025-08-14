import gym
from gym import Wrapper
import numpy as np

class GymnasiumAdapter(Wrapper):
    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated, truncated = done, False
        else:
            raise ValueError(f"Invalid step result format: {len(result)}")

        obs = np.array(obs, dtype=np.float32)  # ✅ 保证 obs 格式
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        obs = np.array(obs, dtype=np.float32)  # ✅ 保证 obs 格式
        return obs, info
