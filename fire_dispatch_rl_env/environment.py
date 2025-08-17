import gym
from gym import spaces
import numpy as np

from fire_dispatch_rl_env.simulator_core import Simulator
from fire_dispatch_rl_env.utils import get_observation


class FireDispatchEnv(gym.Env):
    """
    🚒 FireDispatchEnv：城市消防调度强化学习环境
    - 支持多车调度
    - 支持动作索引越界 fallback
    - 返回格式兼容 Stable-Baselines3（obs, reward, done, info）
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, config, event_df=None):
        super(FireDispatchEnv, self).__init__()
        self.config = config
        self.sim = Simulator(config, event_df=event_df)
        self.num_engines = len(self.sim.engines)

        self.obs_dim = config.get("obs_dim", 96)
        self.max_dispatch_per_event = config.get("max_dispatch_per_event", 4)
        self.fallback_on_invalid = config.get("fallback_on_invalid", True)

        self.action_space = spaces.MultiDiscrete([self.num_engines] * self.max_dispatch_per_event)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        self.last_sorted_actions = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)
        self.sim.reset()
        self.num_engines = len(self.sim.engines)
        self.last_sorted_actions = self.get_sorted_available_actions()

        obs = get_observation(self.sim)
        info = {
            "step_count": 0,
            "pending_events": len(self.sim.pending_events),
            "message": "环境已重置"
        }
        return obs, info

    def step(self, action_idxs):
        self.last_sorted_actions = self.get_sorted_available_actions()

        if not self.last_sorted_actions:
            print("⚠️ 无可调度车辆，跳过事件")
            obs = get_observation(self.sim)
            reward, terminated, sim_info = self.sim.step([])
            # 不再因调度失败终止，只在 max_steps 停止
            truncated = self.sim.step_count >= self.sim.max_steps
            done = truncated
            info = {
                "step_count": self.sim.step_count,
                "pending_events": len(self.sim.pending_events),
                "time": self.sim.time,
                "selected_engine_ids": [],
                "selected_engine_ranks": [],
                "last_response_time": self.sim.response_times[-1] if self.sim.response_times else None,
                "avg_response_time": np.mean(self.sim.response_times) if self.sim.response_times else None,
                "terminated_flag": terminated,
            }
            info.update(sim_info)
            return obs, reward, done, info

        # 单动作 -> 列表
        if isinstance(action_idxs, (np.integer, int)):
            action_idxs = [int(action_idxs)]
        elif isinstance(action_idxs, (np.ndarray, list)):
            action_idxs = [int(a) for a in action_idxs]
        else:
            raise ValueError(f"Unsupported action type: {type(action_idxs)}")

        current_event = self.sim.pending_events[0] if self.sim.pending_events else None
        dispatch_count = current_event.get_required_dispatch_count() if current_event else 1
        action_idxs = action_idxs[:dispatch_count]

        selected_engines = []
        for idx in action_idxs:
            if idx < len(self.last_sorted_actions):
                selected_engines.append(self.last_sorted_actions[idx])
            else:
                if self.fallback_on_invalid:
                    fallback_engine = self.last_sorted_actions[0]
                    print(f"⚠️ 动作索引 {idx} 越界，使用 fallback 车辆 {fallback_engine}")
                    selected_engines.append(fallback_engine)
                else:
                    print(f"❌ 动作索引 {idx} 越界，终止")
                    obs = get_observation(self.sim)
                    return obs, -1000.0, True, {"error": "invalid_action_index"}

        if hasattr(self.sim, "step_multi"):
            reward, terminated, sim_info = self.sim.step_multi(selected_engines)
        else:
            reward, terminated, sim_info = self.sim.step(selected_engines)

        obs = get_observation(self.sim)
        truncated = self.sim.step_count >= self.sim.max_steps
        done = truncated  # ✅ 不再因为 terminated 中断，只在 max_steps 时 done

        if sim_info.get("no_engines_dispatched", False):
            print(f"⚠️ 步数 {self.sim.step_count}：事件未能调度车辆")

        info = {
            "step_count": self.sim.step_count,
            "pending_events": len(self.sim.pending_events),
            "time": self.sim.time,
            "selected_engine_ids": selected_engines,
            "selected_engine_ranks": action_idxs,
            "last_response_time": self.sim.response_times[-1] if self.sim.response_times else None,
            "avg_response_time": np.mean(self.sim.response_times) if self.sim.response_times else None,
            "terminated_flag": terminated,
        }
        info.update(sim_info)

        return obs, reward, done, info

    def render(self, mode="human"):
        self.sim.render()

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        np.random.seed(seed)
        if self.sim:
            self.sim.seed(seed)

    def get_available_actions(self):
        return self.sim.get_available_actions()

    def get_sorted_available_actions(self):
        if not self.sim.pending_events:
            return []
        event_node = self.sim.pending_events[0].graph_node
        return self.sim.get_sorted_available_engines(event_node)
