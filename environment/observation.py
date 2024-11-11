from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces

class EmergencyObservationFunction(ObservationFunction):
    """Observation function with normalized features and priority indicators."""

    def __init__(self, ts: TrafficSignal):
        super().__init__(ts)
        
    def __call__(self) -> np.ndarray:
        # Basic state features
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        
        # Regular traffic features (normalized)
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        
        # Emergency vehicle features
        emergency_queue = self.get_emergency_queue_ratio()
        priority_needed = self.get_priority_indicators()
        
        observation = np.array(
            phase_id + 
            min_green + 
            density + 
            queue + 
            emergency_queue +
            priority_needed,
            dtype=np.float32
        )
        return observation

    def observation_space(self) -> spaces.Box:
        num_features = (
            self.ts.num_green_phases +  # phase encoding
            1 +                         # min green
            len(self.ts.lanes) +        # density
            len(self.ts.lanes) +        # queue
            len(self.ts.lanes) +        # emergency queue ratio
            len(self.ts.lanes)          # priority indicators
        )
        return spaces.Box(
            low=np.zeros(num_features, dtype=np.float32),
            high=np.ones(num_features, dtype=np.float32),
        )
    
    def get_emergency_queue_ratio(self):
        """Calculate ratio of emergency vehicles to total vehicles in queue per lane."""
        ratios = []
        for lane in self.ts.lanes:
            total_stopped = 0
            emergency_stopped = 0
            for veh_id in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
                if self.ts.sumo.vehicle.getSpeed(veh_id) < 0.1:  # stopped vehicle
                    total_stopped += 1
                    if self.ts.sumo.vehicle.getTypeID(veh_id) == 'emergency':
                        emergency_stopped += 1
            ratio = emergency_stopped / max(total_stopped, 1)
            ratios.append(ratio)
        return ratios
    
    def get_priority_indicators(self):
        """Binary indicators for lanes needing immediate attention for emergency vehicles."""
        indicators = []
        for lane in self.ts.lanes:
            needs_priority = 0
            for veh_id in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
                if (self.ts.sumo.vehicle.getTypeID(veh_id) == 'emergency' and 
                    self.ts.sumo.vehicle.getSpeed(veh_id) < 0.1 and
                    self.ts.sumo.vehicle.getAccumulatedWaitingTime(veh_id) > 10):  # 10 seconds threshold
                    needs_priority = 1
                    break
            indicators.append(needs_priority)
        return indicators



