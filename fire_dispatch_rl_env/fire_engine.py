class FireEngine:
    """
    🚒 FireEngine 类：单辆消防车调度状态模拟器

    状态流转：
        available → driving（任务总时长） → available

    特性：
    - 可记录派出次数和总行驶时间
    - 冷却时长已包含在任务时长中（不单独处理）
    - 可视化输出友好
    """

    def __init__(self, engine_id, home_node, vehicle_type="PRL", cooldown_seconds=180):
        self.id = engine_id
        self.home_node = home_node
        self.current_node = home_node
        self.vehicle_type = vehicle_type.upper()

        self.status = 'available'
        self.remaining_time = 0
        self.cooldown_duration = cooldown_seconds

        self.dispatch_count = 0
        self.total_driving_time = 0

    def assign_to_event(self, event_node, driving_seconds, reaction_seconds=30, on_scene_seconds=300):
        """
        🚨 分派当前车辆去响应火警事件

        参数：
        - event_node: 目标事件图节点
        - driving_seconds: 去现场时间（秒）
        - reaction_seconds: 出动准备时间（秒）
        - on_scene_seconds: 现场处置时间（秒）
        """
        total_active_time = (
            reaction_seconds +
            driving_seconds +
            on_scene_seconds +
            driving_seconds  # 假设返回时间 = 去现场时间
        )

        self.status = 'driving'
        self.remaining_time = total_active_time
        self.current_node = event_node
        self.dispatch_count += 1
        self.total_driving_time += driving_seconds

    def update(self, seconds: int = 1):
        """
        ⏱️ 模拟时间推进

        支持推进任意秒数。若剩余时间归零或负，恢复为 available。
        """
        time_left = seconds
        while time_left > 0 and self.status != 'available':
            if self.remaining_time >= time_left:
                self.remaining_time -= time_left
                if self.remaining_time == 0:
                    if self.status == 'driving':
                        self.status = 'cooling'
                        self.remaining_time = self.cooldown_duration
                        print(f"[Engine #{self.id}] ➡️ DRIVING → COOLING")
                    elif self.status == 'cooling':
                        self.status = 'available'
                        self.remaining_time = 0
                        self.current_node = self.home_node
                        print(f"[Engine #{self.id}] ✅ COOLING → AVAILABLE")
                break
            else:
                time_left -= self.remaining_time
                if self.status == 'driving':
                    self.status = 'cooling'
                    self.remaining_time = self.cooldown_duration
                    print(f"[Engine #{self.id}] ➡️ DRIVING → COOLING")
                elif self.status == 'cooling':
                    self.status = 'available'
                    self.remaining_time = 0
                    self.current_node = self.home_node
                    print(f"[Engine #{self.id}] ✅ COOLING → AVAILABLE")

            
            

    def is_available(self):
        """✅ 当前是否可调度"""
        return self.status == 'available'

    def get_average_response_time(self):
        """📈 返回平均响应时间（如有记录）"""
        if self.dispatch_count == 0:
            return 0.0
        return self.total_driving_time / self.dispatch_count

    def reset(self):
        """🔄 重置车辆状态"""
        self.status = 'available'
        self.remaining_time = 0
        self.current_node = self.home_node
        self.dispatch_count = 0
        self.total_driving_time = 0

    def __repr__(self):
        """📋 字符串打印格式"""
        return (
            f"<Engine#{self.id} 状态:{self.status.upper()} "
            f"位置:{self.current_node} 类型:{self.vehicle_type} "
            f"剩余:{self.remaining_time:.1f}s 派遣:{self.dispatch_count}次>"
        )
