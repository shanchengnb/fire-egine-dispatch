import random
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from fire_dispatch_rl_env.fire_engine import FireEngine
from fire_dispatch_rl_env.fire_event import FireEvent


class Simulator:
    def __init__(self, config: Dict, event_df=None, seed: Optional[int] = None):
        self.config = config
        self.cooldown_seconds: int = config.get("cooldown_seconds", 180)
        self.max_steps: int = config.get("max_steps", 100000)

        self.station_dists = config.get("station_dists", {})
        self.station_xy = config.get("station_xy", {})

        self.engines: List[FireEngine] = []
        self.pending_events: deque[FireEvent] = deque()
        self.finished_events: List[FireEvent] = []
        self.response_times: List[float] = []

        self.time: int = 0
        self.step_count: int = 0
        self.dispatch_history: List[Dict] = []

        if "event_station_times" in config and isinstance(config["event_station_times"], pd.DataFrame):
            self.event_station_times = config["event_station_times"].copy()
            print(f"[INFO] Using response time matrix from config, shape: {self.event_station_times.shape}")
        else:
            time_csv_path = r"D:\UCL2\FINAL CODE\drv_time_osrm_renamed.csv"
            print(f"[WARN] Loading response time matrix from default path: {time_csv_path}")
            self.event_station_times = pd.read_csv(time_csv_path, index_col=0)

        self.event_station_times.index.name = "incident_index"

        # ✅ Add missing station columns to the response time matrix
        existing_cols = set(self.event_station_times.columns)
        for eid, name in config.get("station_mapping", {}).items():
            if name not in existing_cols:
                print(f"[Patch] Adding missing column: {name}")
                self.event_station_times[name] = 3600.0

        if seed is not None:
            self.seed(seed)

        self.reset(event_df=event_df)

    def seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self, event_df=None):
        self.time = 0
        self.step_count = 0
        self.response_times = []
        self.finished_events = []
        self.dispatch_history = []

        self._init_engines()
        if event_df is not None:
            self.pending_events = self._generate_events(event_df)

    def _init_engines(self):
        self.engines = []
        self.station_mapping = {}

        station_engine_counts = self.config.get("station_engine_counts", {})
        cooldown = self.cooldown_seconds
        eid = 0

        for station, count in station_engine_counts.items():
            home_node = self.station_xy.get(station)
            if home_node is None:
                dist_dict = self.station_dists.get(station, {})
                home_node = list(dist_dict.keys())[0] if dist_dict else (0, 0)

            for _ in range(count):
                engine = FireEngine(
                    engine_id=eid,
                    home_node=home_node,
                    cooldown_seconds=cooldown
                )
                self.engines.append(engine)
                self.station_mapping[eid] = station
                eid += 1

        self.max_engines = len(self.engines)
        self.config["station_mapping"] = self.station_mapping

    def _generate_events(self, df) -> deque:
        events = []
        for eid, row in enumerate(df.itertuples(index=False), start=0):
            row_data = row._asdict()
            event = FireEvent.from_row(row_data, eid)
            event.incident_index = row_data.get("Incident_Number", eid)
            events.append(event)

        events.sort(key=lambda e: e.timestamp)
        return deque(events)

    def step(self, engine_ids: List[int]) -> Tuple[float, bool, Dict]:
        self.step_count += 1

        if not self.pending_events:
            return 0.0, True, {}

        event = self.pending_events[0]
        event_time = event.timestamp

        if self.time < event_time:
            delta = event_time - self.time
            self.time = event_time
            for engine in self.engines:
                engine.update(seconds=delta)
            print(f"⏩ Advancing time to event time {event_time} (+{delta}s)")

        event = self.pending_events.popleft()
        dispatch_count = event.get_required_dispatch_count()
        engine_ids = engine_ids[:dispatch_count]

        reaction_seconds = getattr(event, "reaction_seconds", 30)
        on_scene_seconds = getattr(event, "on_scene_seconds", 300)

        rewards = []
        response_times = []
        used_engines = []

        for i, engine_id in enumerate(engine_ids):
            engine = self.engines[engine_id]
            if not engine.is_available():
                continue

            station = self.station_mapping.get(engine_id)
            incident_index = getattr(event, "incident_index", event.id)

            try:
                driving_seconds = self.event_station_times.at[incident_index, station]
            except Exception as e:
                print(f"[ERROR] Failed to retrieve response time: event={incident_index}, station={station}, reason: {e}")
                driving_seconds = 3600.0

            print(f"[DEBUG] Engine {engine_id} dispatched from {station} | Time: {driving_seconds:.1f}s")

            engine.assign_to_event(
                event_node=event.graph_node,
                reaction_seconds=reaction_seconds,
                on_scene_seconds=on_scene_seconds,
                driving_seconds=driving_seconds
            )

            event.mark_responded(responder_id=engine_id, response_time=driving_seconds)

            record = {
                "event_id": event.id,
                "incident_index": incident_index,
                "engine_id": engine_id,
                "station": station,
                "response_time": driving_seconds,
                "risk_level": event.risk_level,
                "timestamp": event.timestamp,
            }

            # ✅ Record info of the first responding engine
            if i == 0:
                record["dispatched_vehicle_count"] = dispatch_count

            self.dispatch_history.append(record)

            rewards.append(- (driving_seconds ** 2))
            response_times.append(driving_seconds)
            used_engines.append(engine_id)

        if not used_engines:
            self.dispatch_history.append({
                "event_id": event.id,
                "incident_index": getattr(event, "incident_index", event.id),
                "engine_id": None,
                "station": None,
                "response_time": None,
                "risk_level": event.risk_level,
                "timestamp": event.timestamp,
                "error": "no_engines_dispatched"
            })
            return -1000.0, False, {"error": "no_engines_dispatched"}

        self.finished_events.append(event)
        self.response_times.extend(response_times)
        self.time = max(self.time, event.timestamp)
        done = (self.step_count >= self.max_steps) or not self.pending_events

        info = {
            "event_id": event.id,
            "dispatched_engines": used_engines,
            "min_response_time": min(response_times),  # ✅ Shortest response time
            "response_times": response_times,
        }

        return np.mean(rewards), done, info

    def step_multi(self, engine_ids: List[int]) -> Tuple[float, bool, Dict]:
        return self.step(engine_ids)

    def get_available_actions(self) -> List[int]:
        return [e.id for e in self.engines if e.is_available()]

    def get_sorted_available_engines(self, event_node) -> List[int]:
        candidates = [e for e in self.engines if e.is_available()]
        if not candidates or not self.pending_events:
            return []

        event = self.pending_events[0]
        incident_index = getattr(event, "incident_index", event.id)

        times = []
        for e in candidates:
            station = self.station_mapping.get(e.id)
            try:
                t = self.event_station_times.at[incident_index, station]
            except Exception:
                t = float("inf")
            times.append((e.id, t, e.dispatch_count))

        sorted_ids = [eid for eid, _, _ in sorted(times, key=lambda x: (x[1], x[2], x[0]))]
        return sorted_ids

    def render(self):
        print(f"[Simulator] t={self.time}s | remaining events: {len(self.pending_events)}")

    def close(self):
        pass
