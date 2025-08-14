from datetime import datetime
import pandas as pd

class FireEvent:
    """
    🔥 FireEvent: Fire incident object (used in dispatch simulation)

    Supported fields:
    - graph_node (location of the incident)
    - call_time (original timestamp, converted to seconds)
    - incident_profile_label (risk level)
    - reaction_seconds / on_scene_seconds
    - driving_seconds (true response time, useful for evaluation)
    """

    def __init__(self, event_id, location, start_time, extra=None):
        self.id = event_id
        self.graph_node = location
        self.timestamp = start_time
        self.extra = extra or {}

        # 🔍 Key attributes
        self.risk_level = self.extra.get("incident_profile_label", "Unknown")
        self.true_driving_seconds = self.extra.get("driving_seconds", 999)
        self.required_prl = self.extra.get("prl_count", 0)
        self.required_brv = self.extra.get("brv_count", 0)

        self.reaction_seconds = self.extra.get("reaction_seconds", 30)
        self.on_scene_seconds = self.extra.get("on_scene_seconds", 300)

        self.station_position = (
            self.extra.get("station_easting"),
            self.extra.get("station_northing")
        )

        # ✅ Add incident_index for response time matrix mapping
        self.incident_index = self.extra.get("incident_index", self.id)

        # ⛳ State tracking
        self.assigned = False
        self.responder_id = None
        self.response_time = None

    def mark_responded(self, responder_id, response_time):
        """✅ Update status after response"""
        self.assigned = True
        self.responder_id = responder_id
        self.response_time = response_time
        self.true_driving_seconds = response_time  # ✅ Update driving time

    def is_high_risk(self):
        """🚨 Check if this is a high-risk incident"""
        return str(self.risk_level).strip().lower() in [
            "high risk", "secondary fires that attract a 20 minute-response time"
        ]
    
    def get_required_dispatch_count(self):
        return int(self.extra.get("dispatched_vehicle_count", 2 if self.is_high_risk() else 1))

    def to_dict(self):
        """📋 Convert to dictionary (for evaluation logging)"""
        return {
            "id": self.id,
            "incident_index": self.incident_index,
            "graph_node": self.graph_node,
            "timestamp": self.timestamp,
            "risk_level": self.risk_level,
            "driving_seconds": self.true_driving_seconds,
            "assigned": self.assigned,
            "responder_id": self.responder_id,
            "response_time": self.response_time,
            "extra": self.extra
        }

    def __repr__(self):
        return (
            f"<FireEvent#{self.id} Risk:{self.risk_level} "
            f"Time:{self.timestamp}s Response:{self.response_time}s>"
        )

    @classmethod
    def from_row(cls, row, eid):
        """
        🏗️ Create FireEvent object from a CSV/dict row, auto-handling time format
        """
        try:
            call_time_str = row.get("call_time")
            if call_time_str:
                dt = pd.to_datetime(call_time_str)
            else:
                dt = datetime(2009, 1, 1)  # Default origin
            timestamp_seconds = int((dt - datetime(2009, 1, 1)).total_seconds())
        except Exception as e:
            print(f"⚠️ FireEvent time parsing failed: {e}, defaulting to 0")
            timestamp_seconds = 0

        location = row.get("graph_node", (0, 0))
        if isinstance(location, str):
            import ast
            try:
                location = ast.literal_eval(location)
            except Exception:
                location = (0, 0)

        return cls(
            event_id=eid,
            location=location,
            start_time=timestamp_seconds,
            extra=row
        )
