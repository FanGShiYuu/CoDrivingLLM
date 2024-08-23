# ### define basic element for high way
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field
from math import sqrt
import re

@dataclass
class Lane:
    id: str
    laneIdx: int
    left_lanes: List[str] = field(default_factory=list)
    right_lanes: List[str] = field(default_factory=list)

    def export2json(self):
        return {
            'id': self.id,
            'lane index': self.laneIdx,
            'left_lanes': self.left_lanes,
            'right_lanes': self.right_lanes,
        }


@dataclass
class Vehicle:
    id: str
    lane_id: str = ''
    x: float = 0.0
    y: float = 0.0
    speedx: float = 0.0
    speedy: float = 0.0
    presence: bool = False

    def clear(self) -> None:
        self.lane_id = ''
        self.x = 0.0
        self.y = 0.0
        self.speedx = 0.0
        self.speedy = 0.0
        self.presence = False

    def updateProperty(
        self, x: float, y: float, vx: float, vy: float
    ) -> None:
        self.x = x
        self.y = y
        self.speedx = vx
        self.speedy = vy
        laneIdx = round(y/4.0)
        self.lane_id = 'lane_' + str(laneIdx)

    @property
    def speed(self) -> float:
        return sqrt(pow(self.speedx, 2) + pow(self.speedy, 2))

    @property
    def lanePosition(self) -> float:
        return self.x

    def export2json(self) -> Dict:
        return {
            'id': self.id,
            'current lane': self.lane_id,
            # float() is used to transfer np.float32 to float, since np.float32
            # can not be serialized by JSON
            'lane position': round(float(self.x), 2),
            'speed': round(float(self.speed), 2),
        }
# ### defined scenarios
#from baseClass import Lane, Vehicle
from typing import List, Dict
from datetime import datetime
import sqlite3
import json
import os


class Scenario:
    def __init__(self, road_info, vehicleCount: int,  database: str = None) -> None:
        self.lanes: Dict[str, Lane] = {}
        self.road_info =  road_info
        # self.getRoadgraph()
        self.vehicles: Dict[str, Vehicle] = {}
        self.vehicleCount = vehicleCount
        self.initVehicles()
        self.all_vehicles = []

    def export2json(self):
        scenario = {}
        scenario['lanes'] = []
        scenario['vehicles'] = []
        # Iterate through the lanes dictionary and append each lane's export2json() value to the scenario['lanes'] list
        for lv in self.lanes.values():
            scenario['lanes'].append(lv.export2json())
        # Append the ego vehicle's export2json() value to the scenario['ego_info'] list
        scenario['ego_info'] = self.vehicles['ego'].export2json()

        # Iterate through the vehicles dictionary and append each vehicle's export2json() value to the scenario['vehicles'] list if the vehicle is present
        for vv in self.vehicles.values():
            if vv.presence:
                scenario['vehicles'].append(vv.export2json())

        # Return the scenario dictionary as a JSON string
        return json.dumps(scenario)

    def which_lane(land_index):
        "input:j,k, 0. output:lane_1"
        if land_index == ("j", "k", 0) or land_index == ("k", "b", 0) or land_index == ("b", "c", 1):
            return "lane_1"
        elif land_index == ("a", "b", 0) or land_index == ("b", "c", 0) or land_index == ("c", "d", 0):
            return "lane_0"
        else:
            print("err:land_index is not belong to lane_0 or lane_1")
            return "land_index is not belong to lane_0 or lane_1"

    def getRoadgraph(self,):
        road_info = self.road_info.vehicles
        leftLanes = []
        rightLanes = []
        for i, item in enumerate(road_info):
            lane_id = item.lane_index #("b", "c", 1)

            if lane_id == ("j", "k", 0) or lane_id == ("k", "b", 0) or lane_id == ("b", "c", 1):
                currentLaneID = 'lane_1'
                if lane_id == ("b", "c", 1):
                    leftLanes.append('lane_0')
            elif lane_id == ("a", "b", 0) or lane_id == ("b", "c", 0) or lane_id == (
                    "c", "d", 0):
                currentLaneID = 'lane_0'
            else:
                currentLaneID = 'lane_1'
        # self.lanes[lid] = Lane(
        #         id=lid, laneIdx=i,
        #         left_lanes=leftLanes,
        #         right_lanes=rightLanes
        #     )

    def initVehicles(self):
        for i in range(self.vehicleCount):
            if i == 0:
                vid = 'ego'
            else:
                vid = 'veh' + str(i)
            self.vehicles[vid] = Vehicle(id=vid)
        # if database:
        #     self.database = database
        # else:
        #     self.database = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.db'
        #
        # if os.path.exists(self.database):
        #     os.remove(self.database)
        #
        # conn = sqlite3.connect(self.database)
        # cur = conn.cursor()
        # cur.execute(
        #     """CREATE TABLE IF NOT EXISTS vehINFO(
        #         frame INT,
        #         control_num INT,
        #         id TEXT,
        #         x REAL,
        #         y REAL,
        #         lane_id TEXT,
        #         speedx REAL,
        #         speedy REAL,
        #         PRIMARY KEY (frame, id));"""
        # )
        # cur.execute(
        #     """CREATE TABLE IF NOT EXISTS decisionINFO(
        #         frame INT PRIMARY KEY,
        #         scenario TEXT,
        #         thoughtsAndActions TEXT,
        #         finalAnswer TEXT,
        #         outputParser TEXT);"""
        # )
        # conn.commit()
        # conn.close()

        # self.frame = 0



    # def updateVehicles(self, observation, frame, control_num):
    #     self.frame = frame
    #     conn = sqlite3.connect(self.database)
    #     cur = conn.cursor()
    #
    #     for i, vehicle_obs in enumerate(observation):
    #         presence = vehicle_obs[0]  # Assuming the first element of each vehicle observation is 'presence'
    #         x, y, vx, vy = vehicle_obs[1:]
    #
    #         if presence == 1:  # Check if the vehicle is present
    #             vid = f'veh{i}' if i != 0 else 'ego'
    #             veh = self.vehicles[vid]
    #             veh.updateProperty(x, y, vx, vy)
    #             cur.execute(
    #                 '''INSERT INTO vehINFO VALUES (?,?,?,?,?,?,?,?);''',
    #                 (frame, control_num, vid, x, y, veh.lane_id, vx, vy)
    #             )
    #         else:
    #             if i != 0:  # Skip 'ego' when it's not present
    #                 vid = f'veh{i}'
    #                 self.vehicles[vid].clear()
    #
    #     conn.commit()
    #     conn.close()

