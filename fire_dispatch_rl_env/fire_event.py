from datetime import datetime
import pandas as pd

class FireEvent:
    """
    🔥 FireEvent：火警事件对象（用于调度模拟）

    支持字段：
    - graph_node（事件位置）
    - call_time（原始时间，转换为秒）
    - incident_profile_label（风险等级）
    - reaction_seconds / on_scene_seconds
    - driving_seconds（真实响应时长，可用于评估）
    """

    def __init__(self, event_id, location, start_time, extra=None):
        self.id = event_id
        self.graph_node = location
        self.timestamp = start_time
        self.extra = extra or {}

        # 🔍 关键字段
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

        # ✅ 添加 incident_index：用于响应时间矩阵映射
        self.incident_index = self.extra.get("incident_index", self.id)

        # ⛳ 状态追踪
        self.assigned = False
        self.responder_id = None
        self.response_time = None

    def mark_responded(self, responder_id, response_time):
        """✅ 响应后更新状态"""
        self.assigned = True
        self.responder_id = responder_id
        self.response_time = response_time
        self.true_driving_seconds = response_time  # ✅ 更新 driving time

    def is_high_risk(self):
        """🚨 是否高风险事件"""
        return str(self.risk_level).strip().lower() in [
            "high risk", "secondary fires that attract a 20 minute-response time"
        ]
    
    def get_required_dispatch_count(self):
        return int(self.extra.get("dispatched_vehicle_count", 2 if self.is_high_risk() else 1))

    def to_dict(self):
        """📋 转换为 dict（用于评估记录）"""
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
            f"<FireEvent#{self.id} 等级:{self.risk_level} "
            f"时间:{self.timestamp}s 响应:{self.response_time}s>"
        )

    @classmethod
    def from_row(cls, row, eid):
        """
        🏗️ 从 CSV/Dict 创建事件对象，自动处理时间格式等
        """
        try:
            call_time_str = row.get("call_time")
            if call_time_str:
                dt = pd.to_datetime(call_time_str)
            else:
                dt = datetime(2009, 1, 1)  # 默认起点
            timestamp_seconds = int((dt - datetime(2009, 1, 1)).total_seconds())
        except Exception as e:
            print(f"⚠️ FireEvent 时间解析失败: {e}，默认设为 0")
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
