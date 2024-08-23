from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.road.objects import Obstacle
from highway_env.vehicle.kinematics import Vehicle
import numpy as np
#for llm
from .merge_env import *
# from highway_env.envs.llm_agent import *


class MergeEnv(AbstractEnv):
    """
    A highway merge negotiation environment.

    The ego-vehicle is driving on a highway and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """
    n_a = 5
    n_s = 25

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"},
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True},
            "controlled_vehicles": 1,
            "screen_width": 1200,
            "screen_height": 120,
            "centering_position": [0.5, 0.5],  #[0.3, 0.5]
            "scaling": 2,
            "simulation_frequency": 15, #15,  # [Hz]
            "duration": 40,  # time step
            "policy_frequency": 5,  #5  # [Hz]
            "reward_speed_range": [10, 30],
            "COLLISION_REWARD": 30,  # default=200
            "HIGH_SPEED_REWARD": 1,  # default=0.5
            "HEADWAY_COST": 4,  # default=1
            "HEADWAY_TIME": 1.2,  # default=1.2[s]
            "MERGING_LANE_COST": 4,  # default=4
            "traffic_density": 3,  # easy or hard modes
        })
        return config

    def _reward(self, action: list, obs, env) -> float:
        # Cooperative multi-agent reward
        return sum(self._agent_reward(action, vehicle, obs, self.road, env) for vehicle in self.controlled_vehicles) \
               / len(self.controlled_vehicles)

    def calculate_custom_reward(self, rl_action, llm_suggested_action):
        if rl_action == llm_suggested_action:
            return 1  # Reward for matching action
        else:
            return 0

    def _agent_reward(self, action: int, vehicle: Vehicle, obs, road, env) -> float:
        """
            The vehicle is rewarded for driving with high speed on lanes and avoiding collisions
            :param action: the action performed
            :return: the reward of the state-action transition
       """
        scaled_speed = utils.lmap(vehicle.speed, self.config["reward_speed_range"], [0, 1])
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 1):
            Merging_lane_cost = - np.exp(-(vehicle.position[0] - sum(self.ends[:3])) ** 2 / (
                    10 * self.ends[2]))
        else:
            Merging_lane_cost = 0

        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = np.log(
            headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed)) if vehicle.speed > 0 else 0

        reward = self.config["COLLISION_REWARD"] * (-1 * vehicle.crashed) \
             + (self.config["HIGH_SPEED_REWARD"] * np.clip(scaled_speed, 0, 1)) \
             + self.config["MERGING_LANE_COST"] * Merging_lane_cost \
             + self.config["HEADWAY_COST"] * (Headway_cost if Headway_cost < 0 else 0)
        # print("reward:",reward)
        return reward

    def _regional_reward(self):
        for vehicle in self.controlled_vehicles:
            neighbor_vehicle = []

            # vehicle is on the main road
            if vehicle.lane_index == ("a", "b", 0) or vehicle.lane_index == ("b", "c", 0) or vehicle.lane_index == (
                    "c", "d", 0):
                v_fl, v_rl = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the ramp on this road
                elif vehicle.lane_index == ("a", "b", 0) and vehicle.position[0] > self.ends[0]:
                    v_fr, v_rr = self.road.surrounding_vehicles(vehicle, ("k", "b", 0))
                else:
                    v_fr, v_rr = None, None
            else:
                # vehicle is on the ramp
                v_fr, v_rr = self.road.surrounding_vehicles(vehicle)
                if len(self.road.network.side_lanes(vehicle.lane_index)) != 0:
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle,
                                                                self.road.network.side_lanes(
                                                                    vehicle.lane_index)[0])
                # assume we can observe the straight road on the ramp
                elif vehicle.lane_index == ("k", "b", 0):
                    v_fl, v_rl = self.road.surrounding_vehicles(vehicle, ("a", "b", 0))
                else:
                    v_fl, v_rl = None, None
            for v in [v_fl, v_fr, vehicle, v_rl, v_rr]:
                if type(v) is MDPVehicle and v is not None:
                    neighbor_vehicle.append(v)
            regional_reward = sum(v.local_reward for v in neighbor_vehicle)
            vehicle.regional_reward = regional_reward / sum(1 for _ in filter(None.__ne__, neighbor_vehicle))

    def step(self, action: int, env) -> Tuple[np.ndarray, float, bool, dict]:
        agent_info = []
        obs, reward, done, info = super().step(action, env)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        for v in self.controlled_vehicles:
            agent_info.append([v.position[0], v.position[1], v.speed])
        info["agents_info"] = agent_info

        obs = np.asarray(obs).reshape((len(obs), -1))
        return obs, reward, done, info

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        self.crashed_and_clean()
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return vehicle.crashed \
               or self.steps >= self.config["duration"] * self.config["policy_frequency"]

    def crashed_and_clean(self):

        crashed_agent = False  # 判定agent是否碰撞
        crashed_hv = False  # 判定除agent之外的HV是否发生碰撞
        for vehicle in self.controlled_vehicles:  # 判定：Agent发生碰撞才是真正的碰撞
            crashed_agent = crashed_agent or vehicle.crashed
        if crashed_agent == False:  # agent 没有碰撞
            for vehicle in self.road.vehicles:
                if (vehicle not in self.controlled_vehicles) and (vehicle.crashed):  # 选择背景车进行判定
                    self.road.vehicles.remove(vehicle)  # 背景车之间如果发生碰撞，直接删除该背景车

    def _reset(self, num_CAV=0) -> None:
        # num_CAV = self.config['controlled_vehicles']
        # print(f'first num cav:{num_CAV}')
        self._make_road()
        # num_CAV == 0
        if self.config["traffic_density"] == 1:
            # easy mode: 1-3 CAVs + 1-3 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(1, 4), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(1, 4), 1)[0]

        elif self.config["traffic_density"] == 2:
            # hard mode: 2-4 CAVs + 2-4 HDVs
            if num_CAV == 0:
                num_CAV = np.random.choice(np.arange(2, 5), 1)[0]
            else:
                num_CAV = num_CAV
            num_HDV = np.random.choice(np.arange(2, 5), 1)[0]

        elif self.config["traffic_density"] == 3:
            # hard mode: 4-6 CAVs + 3-5 HDVs
            if num_CAV == 0:
                # num_CAV = np.random.choice(np.arange(4, 7), 1)[0]  # TODO: change to fix number
                num_CAV = 3
            else:
                num_CAV = 3
            # num_HDV = np.random.choice(np.arange(3, 6), 1)[0]
            num_HDV = 3
        self._make_vehicles(num_CAV, num_HDV)
        self.action_is_safe = True
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self, ) -> None:
        """
        Make a road composed of a straight highway and a merging lane.
        :return: the road
        """
        net = RoadNetwork()

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        net.add_lane("a", "b", StraightLane([0, 0], [sum(self.ends[:2]), 0], line_types=[c, c]))
        net.add_lane("b", "c", StraightLane([sum(self.ends[:2]), 0], [sum(self.ends[:3]), 0], line_types=[s, s]))
        lcd = StraightLane([sum(self.ends[:3]), 0], [sum(self.ends), 0], line_types=[c, c])
        net.add_lane("c", "d", lcd)

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4], [sum(self.ends[:2]), 6.5 + 4], line_types=[c, c], forbidden=True)
        # lkc = SineLane(ljk.position(self.ends[0], -amplitude), ljk.position(sum(self.ends[:2]), -amplitude), amplitude, 2 * np.pi / (2 * self.ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lkc = StraightLane([sum(self.ends[:2]), 6.5 + 4], [sum(self.ends[:3]), 0], line_types=[s, s], forbidden=True)
        # lbc = StraightLane(lkb.position(self.ends[1], 0), lkb.position(self.ends[1], 0) + [self.ends[2], 0], line_types=[n, c], forbidden=True)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "c", lkc)
        # net.add_lane("c", "d", lcd)
        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        # road.objects.append(Obstacle(road, lbc.position(self.ends[2], 0)))
        self.road = road

    def _make_vehicles(self, num_CAV=3, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as the ego vehicles.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        self.controlled_vehicles = []

        spawn_points_s = [10, 50, 90, 130, 170, 210]
        spawn_points_m = [5, 45, 85, 125, 165, 205]

        """Spawn points for CAV"""
        # spawn point indexes on the straight road
        spawn_point_s_c = np.random.choice(spawn_points_s, num_CAV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_c = np.random.choice(spawn_points_m, num_CAV - num_CAV // 2,
                                           replace=False)
        spawn_point_s_c = list(spawn_point_s_c)
        spawn_point_m_c = list(spawn_point_m_c)
        # remove the points to avoid duplicate
        for a in spawn_point_s_c:
            spawn_points_s.remove(a)
        for b in spawn_point_m_c:
            spawn_points_m.remove(b)

        """Spawn points for HDVs"""
        # spawn point indexes on the straight road
        spawn_point_s_h = np.random.choice(spawn_points_s, num_HDV // 2, replace=False)
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(spawn_points_m, num_HDV - num_HDV // 2, replace=False)
        spawn_point_s_h = list(spawn_point_s_h)
        spawn_point_m_h = list(spawn_point_m_h)

        # initial speed with noise and location noise
        initial_speed = np.random.rand(num_CAV + num_HDV) * 2 + 15  # range from [25, 27]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 3 - 1.5  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)
        print(f'The number of CAV:{num_CAV}')

        """spawn the CAV on the straight road first"""
        for _ in range(num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("a", "b", 0)).position(
                spawn_point_s_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            destination = 'd'
            ego_vehicle.plan_route_to(destination)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the rest CAV on the merging road"""
        for _ in range(num_CAV - num_CAV // 2):
            ego_vehicle = self.action_type.vehicle_class(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0), speed=initial_speed.pop(0))
            destination = 'd'
            ego_vehicle.plan_route_to(destination)
            self.controlled_vehicles.append(ego_vehicle)
            road.vehicles.append(ego_vehicle)

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 2):
            idm_vehicle = other_vehicles_type(road, road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h.pop(0) + loc_noise.pop(0), 0),
                                    speed=initial_speed.pop(0))
            destination = 'd'
            idm_vehicle.plan_route_to(destination)
            road.vehicles.append(idm_vehicle)

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - num_HDV // 2):
            idm_vehicle = other_vehicles_type(road, road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_h.pop(0) + loc_noise.pop(0), 0),
                                speed=initial_speed.pop(0))
            destination = 'd'
            idm_vehicle.plan_route_to(destination)
            road.vehicles.append(idm_vehicle)

        for v in self.road.vehicles:  # Prevent early collisions
            if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                self.road.vehicles.remove(v)

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 15) -> bool:
        return "d" in vehicle.lane_index[1] and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def terminate(self):
        return

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds


class MergeEnvMARL(MergeEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                    "lateral": True,
                    "longitudinal": True
                }},
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics"
                }},
            "controlled_vehicles": 4  # here to change to number of CAV
        })
        return config


register(
    id='merge-v1',
    entry_point='highway_env.envs:MergeEnv',
)

register(
    id='merge-multi-agent-v0',
    entry_point='highway_env.envs:MergeEnvMARL',
)
