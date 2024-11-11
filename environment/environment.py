import random
import os
from typing import List, Optional, Tuple, Union, Callable

import traci

import gymnasium as gym
from sumo_rl import SumoEnvironment, TrafficSignal
from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction


LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ

class MultiRouteSumoEnvironment(SumoEnvironment):
    def __init__(
        self,
        net_file: str,
        route_files: List[str],
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        virtual_display: Tuple[int, int] = (3200, 1800),
        begin_time: int = 0,
        num_seconds: int = 20000,
        max_depart_delay: int = -1,
        waiting_time_memory: int = 1000,
        time_to_teleport: int = -1,
        delta_time: int = 5,
        yellow_time: int = 2,
        min_green: int = 5,
        max_green: int = 50,
        single_agent: bool = True,
        reward_fn: Union[str, Callable, dict] = "diff-waiting-time",
        observation_class: ObservationFunction = DefaultObservationFunction,
        add_system_info: bool = True,
        add_per_agent_info: bool = True,
        sumo_seed: Union[str, int] = "random",
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        additional_sumo_cmd: Optional[str] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize CustomSumoEnv."""
        # Call the parent class constructor with a dummy route_file
        super().__init__(
            net_file=net_file,
            route_file=route_files[0],  # Pass the first route file as a dummy
            out_csv_name=out_csv_name,
            use_gui=use_gui,
            virtual_display=virtual_display,
            begin_time=begin_time,
            num_seconds=num_seconds,
            max_depart_delay=max_depart_delay,
            waiting_time_memory=waiting_time_memory,
            time_to_teleport=time_to_teleport,
            delta_time=delta_time,
            yellow_time=yellow_time,
            min_green=min_green,
            max_green=max_green,
            single_agent=single_agent,
            reward_fn=reward_fn,
            observation_class=observation_class,
            add_system_info=add_system_info,
            add_per_agent_info=add_per_agent_info,
            sumo_seed=sumo_seed,
            fixed_ts=fixed_ts,
            sumo_warnings=sumo_warnings,
            additional_sumo_cmd=additional_sumo_cmd,
            render_mode=render_mode,
        )
        self.route_files = route_files
        self.current_route_file = None

    def reset(self, seed: Optional[int] = None, **kwargs):
        """Reset the environment with a new route file."""
        if self.episode != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.episode)
        self.episode += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
            random.seed(seed)

        # Choose a new route file randomly
        self.current_route_file = random.choice(self.route_files)

        self._start_simulation()

        if isinstance(self.reward_fn, dict):
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn[ts],
                    self.sumo,
                )
                for ts in self.reward_fn.keys()
            }
        else:
            self.traffic_signals = {
                ts: TrafficSignal(
                    self,
                    ts,
                    self.delta_time,
                    self.yellow_time,
                    self.min_green,
                    self.max_green,
                    self.begin_time,
                    self.reward_fn,
                    self.sumo,
                )
                for ts in self.ts_ids
            }

        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]], self._compute_info()
        else:
            return self._compute_observations()

    def _start_simulation(self):
        sumo_cmd = [
            self._sumo_binary,
            "-n",
            self._net,
            "-r",
            self.current_route_file,  # Use the current route file
            "--max-depart-delay",
            str(self.max_depart_delay),
            "--waiting-time-memory",
            str(self.waiting_time_memory),
            "--time-to-teleport",
            str(self.time_to_teleport),
        ]
        if self.begin_time > 0:
            sumo_cmd.append(f"-b {self.begin_time}")
        if self.sumo_seed == "random":
            sumo_cmd.append("--random")
        else:
            sumo_cmd.extend(["--seed", str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append("--no-warnings")
        if self.additional_sumo_cmd is not None:
            sumo_cmd.extend(self.additional_sumo_cmd.split())
        if self.use_gui or self.render_mode is not None:
            sumo_cmd.extend(["--start", "--quit-on-end"])
            if self.render_mode == "rgb_array":
                sumo_cmd.extend(["--window-size", f"{self.virtual_display[0]},{self.virtual_display[1]}"])
                from pyvirtualdisplay.smartdisplay import SmartDisplay

                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")

        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui or self.render_mode is not None:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
            
class EmergencySumoEnvironment(MultiRouteSumoEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _compute_info(self):
        """Simplified info dict focusing on key metrics."""
        info = {"step": self.sim_step}
        
        if self.add_system_info:
            info.update(self._get_system_info())
        if self.add_per_agent_info:
            info.update(self._get_per_agent_info())
        
        # Track basic emergency metrics
        total_emergency_waiting_time = 0
        total_emergency_vehicles = 0
        
        for ts_id in self.ts_ids:
            ts = self.traffic_signals[ts_id]
            for lane in ts.lanes:
                for veh_id in ts.sumo.lane.getLastStepVehicleIDs(lane):
                    if ts.sumo.vehicle.getTypeID(veh_id) == 'emergency':
                        total_emergency_waiting_time += ts.sumo.vehicle.getAccumulatedWaitingTime(veh_id)
                        total_emergency_vehicles += 1
        
        info['emergency_waiting_time'] = total_emergency_waiting_time
        info['emergency_vehicle_count'] = total_emergency_vehicles
        
        self.metrics.append(info.copy())
        return info
