from sumo_rl.environment.traffic_signal import TrafficSignal

def emergency_reward(ts: TrafficSignal) -> float:
    """Normalized reward function with emergency priority."""
    # Base components
    regular_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    regular_reward = -regular_wait / len(ts.lanes)  # Normalize by number of lanes
    
    # Emergency component
    emergency_reward = 0
    emergency_count = 0
    max_speed = max([ts.sumo.lane.getMaxSpeed(lane) for lane in ts.lanes])
    
    for lane in ts.lanes:
        for veh_id in ts.sumo.lane.getLastStepVehicleIDs(lane):
            if ts.sumo.vehicle.getTypeID(veh_id) == 'emergency':
                emergency_count += 1
                current_speed = ts.sumo.vehicle.getSpeed(veh_id)
                speed_ratio = current_speed / max_speed
                waiting_time = ts.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
                
                # Normalize waiting time penalty (max 30 seconds)
                wait_penalty = min(waiting_time / 30.0, 1.0)
                
                # Combine speed and waiting penalties
                emergency_reward -= (0.7 * (1 - speed_ratio) + 0.3 * wait_penalty)
    
    # Normalize emergency reward
    if emergency_count > 0:
        emergency_reward = emergency_reward / emergency_count
    
    # Combine rewards (keeping scale similar to vanilla)
    total_reward = 0.6 * regular_reward + 0.4 * emergency_reward
    
    # Delta reward (similar to vanilla)
    reward = total_reward - ts.last_measure
    ts.last_measure = total_reward
    
    return reward