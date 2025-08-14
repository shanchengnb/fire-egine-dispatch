class FireEngine:
    """
    🚒 FireEngine Class: Simulator for the dispatch state of a single fire engine

    State transitions:
        available → driving (total task duration) → available

    Features:
    - Tracks number of dispatches and total driving time
    - Cooldown duration is included in the task time (not handled separately)
    - Friendly visual output
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
        🚨 Assign this engine to respond to a fire event

        Parameters:
        - event_node: Target event graph node
        - driving_seconds: Time to reach the scene (in seconds)
        - reaction_seconds: Preparation time before departure (in seconds)
        - on_scene_seconds: Time spent handling the scene (in seconds)
        """
        total_active_time = (
            reaction_seconds +
            driving_seconds +
            on_scene_seconds +
            driving_seconds  # Assume return time = travel time to scene
        )

        self.status = 'driving'
        self.remaining_time = total_active_time
        self.current_node = event_node
        self.dispatch_count += 1
        self.total_driving_time += driving_seconds

    def update(self, seconds: int = 1):
        """
        ⏱️ Simulate time progression

        Supports advancing by any number of seconds.
        If remaining time reaches zero or below, resets to available.
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
        """✅ Check if the engine is currently dispatchable"""
        return self.status == 'available'

    def get_average_response_time(self):
        """📈 Return average response time (if any)"""
        if self.dispatch_count == 0:
            return 0.0
        return self.total_driving_time / self.dispatch_count

    def reset(self):
        """🔄 Reset engine status"""
        self.status = 'available'
        self.remaining_time = 0
        self.current_node = self.home_node
        self.dispatch_count = 0
        self.total_driving_time = 0

    def __repr__(self):
        """📋 String representation format"""
        return (
            f"<Engine#{self.id} STATUS:{self.status.upper()} "
            f"Location:{self.current_node} Type:{self.vehicle_type} "
            f"Remaining:{self.remaining_time:.1f}s Dispatches:{self.dispatch_count}>"
        )
