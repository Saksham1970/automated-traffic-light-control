from sumo_rl.environment.observations import ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
import numpy as np
from gymnasium import spaces
from custom_sumo import CustomSumoEnv

class EmergencyObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        emergency_halts = self.get_emergency_halts()
        observation = np.array(phase_id + min_green + density + queue + emergency_halts, dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 3 * len(self.ts.lanes), dtype=np.float32),
        )
    
    def get_emergency_halts(self):
        lane_counts = []
        for lane in self.ts.lanes:
            lane_count = 0
            for vehId in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
                if self.ts.sumo.vehicle.getTypeID(vehId) == 'emergency' and self.ts.sumo.vehicle.getSpeed(vehId) < 0.1:
                    lane_count += 1
            lane_counts.append(lane_count)
        return lane_counts

def get_accumulated_emergency_waiting_time_per_lane(ts):
        wait_time_per_lane = []
        for lane in ts.lanes:
            veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
            wait_time = 0.0
            for veh in veh_list:
                if ts.sumo.vehicle.getTypeID(veh) == 'emergency':
                    acc = ts.sumo.vehicle.getAccumulatedWaitingTime(veh)
                    wait_time += acc
            wait_time_per_lane.append(wait_time)
        return wait_time_per_lane
    
def emergency_reward(ts):
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane())/ 100.0
    ts_wait += sum(get_accumulated_emergency_waiting_time_per_lane(ts))
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    return reward

class EmergencySUMOEnvironment(CustomSumoEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_info(self):
        """Compute the info dict."""
        info = {"step": self.sim_step}
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        accumulated_waiting_time = [
            sum(get_accumulated_emergency_waiting_time_per_lane(self.traffic_signals[ts])) for ts in self.ts_ids
        ]
        info['emergency_accumulated_waiting_time'] = sum(accumulated_waiting_time)
        self.metrics.append(info.copy())
        return info