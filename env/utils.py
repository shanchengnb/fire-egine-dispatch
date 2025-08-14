import numpy as np

_DEFAULT_RISK_MAP = {
    "false alarms": 0,
    "secondary fires that attract a 20 minute-response time": 1,
    "low risk": 2,
    "medium risk": 3,
    "high risk": 4
}

def _one_hot(index: int, length: int):
    vec = [0] * length
    if 0 <= index < length:
        vec[index] = 1
    return vec

def get_observation(sim):
    cfg = sim.config
    obs = []

    # === Current Event ===
    event = sim.pending_events[0] if sim.pending_events else None
    if event:
        ex, ey = event.graph_node
        x = ex / cfg.get("map_width", 400000)
        y = ey / cfg.get("map_height", 400000)

        risk_map = cfg.get("risk_map", _DEFAULT_RISK_MAP)
        risk_label = str(event.risk_level).lower()
        risk_idx = risk_map.get(risk_label, len(risk_map) - 1)
        risk_onehot = _one_hot(risk_idx, len(risk_map))

        wait_time = np.clip((sim.time - event.timestamp) / 300.0, 0.0, 1.0)

        obs.extend([x, y])
        obs.extend(risk_onehot)
        obs.append(wait_time)
    else:
        obs.extend([0.0, 0.0])
        obs.extend([0] * len(_DEFAULT_RISK_MAP))
        obs.append(0.0)

    # === Fire Engine Features ===
    N = cfg.get("obs_engine_count", 10)
    event_node = event.graph_node if event else None
    sorted_ids = sim.get_sorted_available_engines(event_node) if event_node else []

    for idx in range(N):
        if idx < len(sorted_ids):
            eng = sim.engines[sorted_ids[idx]]
        elif idx < len(sim.engines):
            eng = sim.engines[idx]
        else:
            eng = None

        if eng is None:
            obs.extend([0, 0, 0])
            obs.extend([0.0] * 5)
            continue

        status = _one_hot({
            "available": 0,
            "driving": 1,
            "cooling": 2
        }.get(eng.status, 0), 3)

        # === Response Time Features ===
        try:
            station = sim.station_mapping.get(eng.id)
            event_idx = event.incident_index  # ✅ Use incident_index instead of event.id
            travel_time = sim.event_station_times.at[event_idx, station]
        except Exception:
            travel_time = 3600.0

        dist = min(travel_time, 3600.0) / 3600.0
        rank_norm = idx / N
        usage = np.clip(eng.dispatch_count / 10.0, 0.0, 1.0)
        remain = np.clip(eng.remaining_time / 600.0, 0.0, 1.0) if eng.status != "available" else 0.0

        obs.extend(status)
        obs.extend([dist, rank_norm, usage, remain, eng.id / cfg.get("max_engines", 40)])

    # === Time Features ===
    time_of_day = (sim.time % 86400) / 86400.0
    progress_ratio = sim.step_count / sim.max_steps
    obs.extend([time_of_day, progress_ratio])

    # === Pad or Trim to Fixed Dimension ===
    obs_dim = cfg.get("obs_dim", 96)
    if len(obs) > obs_dim:
        obs = obs[:obs_dim]
    elif len(obs) < obs_dim:
        obs += [0.0] * (obs_dim - len(obs))

    return np.array(obs, dtype=np.float32)
