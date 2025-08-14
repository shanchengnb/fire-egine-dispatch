import gym
from gym import spaces
import numpy as np

from fire_dispatch_rl_env.simulator_core import Simulator
from fire_dispatch_rl_env.utils import get_observation


class FireDispatchEnv(gym.Env):
    """
    🚒 FireDispatchEnv: Urban Fire Dispatch Reinforcement Learning Environment
    - Supports multi-engine dispatch
    - Supports fallback for out-of-bound action indices
    - Return format compatible with Stable-Baselines3 (obs, reward, done, info)
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
            "message": "Environment reset"
        }
        return obs, info

    def step(self, action_idxs):
        self.last_sorted_actions = self.get_sorted_available_actions()

        if not self.last_sorted_actions:
            print("⚠️ No available vehicles, skipping event")
            obs = get_observation(self.sim)
            reward, terminated, sim_info = self.sim.step([])
            # No longer terminating on dispatch failure, only at max_steps
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

        # Single action -> convert to list
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
                    print(f"⚠️ Action index {idx} out of bounds, using fallback engine {fallback_engine}")
                    selected_engines.append(fallback_engine)
                else:
                    print(f"❌ Action index {idx} out of bounds, terminating")
                    obs = get_observation(self.sim)
                    return obs, -1000.0, True, {"error": "invalid_action_index"}

        if hasattr(self.sim, "step_multi"):
            reward, terminated, sim_info = self.sim.step_multi(selected_engines)
        else:
            reward, terminated, sim_info = self.sim.step(selected_engines)

        obs = get_observation(self.sim)
        truncated = self.sim.step_count >= self.sim.max_steps
        done = truncated  # ✅ No longer interrupted by terminated, only ends at max_steps

        if sim_info.get("no_engines_dispatched", False):
            print(f"⚠️ Step {self.sim.step_count}: No engines dispatched for event")

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
