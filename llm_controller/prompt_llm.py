from .Scenario_description import Scenario
import re
import numpy as np

# from merge_env import Scenario
class PRE_DEF_PROMPT():
    """
    These rules can be modified to test if changing prompt leads to different behaviour pattern of our agent.
    """

    def __init__(self):
        self.SYSTEM_MESSAGE_PREFIX = """
    You are ChatGPT, a large language model trained by OpenAI.
    You are now act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
    The information in 'current scenario' :

    """

        self.TRAFFIC_RULES = """
    1. Try to keep a safe distance to the car in front of you.
    2. DONOT change lane frequently. If you want to change lane, double-check the safety of vehicles on target lane.
    """

        self.DECISION_CAUTIONS = """
    1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example you cannot say "I can either keep lane or accelerate at current time".
    2. You need to always remember your current lane ID, your available actions and available lanes before you make any decision.
    3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.
    4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch.
    5. The most important information is: Your final decision have to be one of the following action: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER
    """

        self.SYSTEM_MESSAGE_PREFIX_intersection = """
    You are ChatGPT, a large language model trained by OpenAI.
    You are now act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
    This environment you need to face is the intersection.
    The information in 'current scenario' :

    """

        self.TRAFFIC_RULES_intersection = """
    1. Try to keep a safe distance to the car in front of you.
    2. Remeber you can't change your lane. In this intersection environment, you just can choose three kinds of actions to control the longitudinal movement of the vehicle: IDLE, FASTER, SLOWER.
    """

        self.DECISION_CAUTIONS_intersection = """
    1. You must output a decision when you finish this task. Your final output decision must be unique and not ambiguous. For example you cannot say "I can either keep lane or accelerate at current time".
    2. You need to always remember your current lane ID, your available actions and available lanes before you make any decision.
    3. Once you have a decision, you should check the safety with all the vehicles affected by your decision.
    4. If you verify a decision is unsafe, you should start a new one and verify its safety again from scratch.
    5. The most important information is: Your final decision have to be one of the following action: IDLE, FASTER, SLOWER (you can not output other decisions like ACCELERATE, DECELERATE...)
    """

    def get_traffic_rules(self, is_intersection=False):
        return self.TRAFFIC_RULES_intersection if is_intersection else self.TRAFFIC_RULES


    def get_decision_cautions(self, is_intersection=False):
        return self.DECISION_CAUTIONS_intersection if is_intersection else self.DECISION_CAUTIONS

    # ## Ask LLM
    # ### generate prompt for llm and ask for response

ACTIONS_ALL = {
    0: 'LANE_LEFT',
    1: 'IDLE',
    2: 'LANE_RIGHT',
    3: 'FASTER',
    4: 'SLOWER'
}

ACTIONS_DESCRIPTION = {
    0: 'change lane to the left of the current lane,',
    1: 'remain in the current lane with current speed',
    2: 'change lane to the right of the current lane',
    3: 'accelerate the vehicle',
    4: 'decelerate the vehicle'
}

def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

def which_lane(land_index, env):
    # get scenario name
    scene_name = env.spec.id
    match = re.search(r'(merge|intersection|highway)', scene_name)
    simplified_scene_name = match.group(0) if match else 'unknown'
    if simplified_scene_name == 'highway':
        return land_index
        # if land_index == ("j", "k", 0) or land_index == ("k", "b", 0) or land_index == ("b", "c", 1):
        #     return "lane_1"
        # elif land_index == ("a", "b", 0) or land_index == ("b", "c", 0) or land_index == ("c", "d", 0):
        #     return "lane_0"
        # else:
        #     # print("err:land_index is not belong to lane_0 or lane_1")
        #     return "land_index is not belong to lane_0 or lane_1"
    else:
        return land_index


class getAvailableActions:
    # def __init__(self, ) -> None:

    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def _get_available_actions1(self, vehicle, env_copy, is_intersection=False):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        # if not isinstance(self.action_type, DiscreteMetaAction):
        #     raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [env_copy.ACTIONS_ALL['IDLE']]
        if is_intersection:  # there is no lane changing in the intersection env
            if vehicle.speed_index < vehicle.SPEED_COUNT - 1:
                actions.append(env_copy.ACTIONS_ALL['FASTER'])
            if vehicle.speed_index > 0:
                actions.append(env_copy.ACTIONS_ALL['SLOWER'])
        else:
            for l_index in env_copy.road.network.side_lanes(vehicle.lane_index):
                if l_index[2] < vehicle.lane_index[2] \
                        and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                    actions.append(env_copy.ACTIONS_ALL['LANE_LEFT'])
                if l_index[2] > vehicle.lane_index[2] \
                        and env_copy.road.network.get_lane(l_index).is_reachable_from(vehicle.position):
                    actions.append(env_copy.ACTIONS_ALL['LANE_RIGHT'])
            if vehicle.speed_index < vehicle.SPEED_COUNT - 1:
                actions.append(env_copy.ACTIONS_ALL['FASTER'])
            if vehicle.speed_index > 0:
                actions.append(env_copy.ACTIONS_ALL['SLOWER'])
        return actions

    def inference(self, input: str, ego_veh, road, env, is_intersection=False) -> str:
        outputPrefix = 'You can ONLY use one of the following actions: \n '
        availableActions = self._get_available_actions1(ego_veh, env, is_intersection)

        for action in availableActions:
            outputPrefix += ACTIONS_ALL[action] + \
                            '--' + ACTIONS_DESCRIPTION[action] + '; \n'
        if is_intersection:
            if 1 in availableActions:
                outputPrefix += 'You should check idle action as FIRST priority. '

            if 3 in availableActions:
                outputPrefix += 'Consider acceleration action carefully. '
            if 4 in availableActions:
                outputPrefix += 'The deceleration action is LAST priority. '
        else:
            if 1 in availableActions:
                outputPrefix += 'You should check idle action as FIRST priority. '

            if 0 in availableActions or 2 in availableActions:
                outputPrefix += 'For change lane action, CAREFULLY CHECK the safety of vehicles on target lane. '
            if 3 in availableActions:
                outputPrefix += 'Consider acceleration action carefully. '
            if 4 in availableActions:
                outputPrefix += 'The deceleration action is LAST priority. '
        outputPrefix += """\nTo check decision safety you should follow steps:
      Step 1: Get the vehicles in this lane that you may affect. Acceleration, deceleration and idle action affect the Current lane, while left and right lane changes affect the Adjacent lane.
      Step 2: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
      Step 3: If you find There is no car driving on your "current lane" and you have no conflict with any other vehicle, you can drive faster ! but not too fast to follow the traffic rules.
      Step 4: If you want to make lane change consider :"Safety Assessment for Lane Changes:" Safe means it is safe to change ,If you want to do IDLE, FASTER, SLOWER, you should consider "Safety Assessment in Current Lane:"
      """
        # print(f'out:::{outputPrefix}')
        return outputPrefix


class isActionSafe:
    def __init__(self) -> None:
        pass

    # @prompts(name='Check Action Safety',
    #          description="""Use this tool when you want to check the proposed action's safety. The input to this tool should be a string, which is ONLY the action name.""")
    @prompts(name='Decision-making Instructions',
             description="""This tool gives you a brief intruduction about how to ensure that the action you make is safe. The input to this tool should be a string, which is ONLY the action name.""")
    def inference(self, action: str) -> str:
        return f"""To check action safety you should follow three steps:
      Step 1: Identify the lanes affected by this action. Acceleration, deceleration and idle affect the current lane, while left and right lane changes affect the corresponding lane.
      Step 2:(Optional) Get the vehicles in this lane that you may affect, ONLY when you don't know.
      Step 3: If there are vehicles, check safety between ego and all vehicles in the action lane ONE by ONE.
      Follow the instructions and remember to use the proper tools mentioned in the tool list once a time.
      """


class getAvailableLanes:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get Available Lanes',
             description="""useful when you want to know the available lanes of the vehicles. like: I want to know the available lanes of the vehicle `ego`. The input to this tool should be a string, representing the id of the vehicle.""")
    def inference(self, vid: str, ego_veh, road, env) -> str:
        # veh = self.sce.vehicles[vid]
        # currentLaneID = veh.lane_id
        lane_id = ego_veh.lane_index
        availabel_lane = {}
        scene_name = env.spec.id
        match = re.search(r'(merge|intersection|highway)', scene_name)
        simplified_scene_name = match.group(0) if match else 'unknown'
        currentLaneID = None
        if simplified_scene_name == 'highway':
            if lane_id == ("b", "c", 1):
                leftLane = 'lane_0'
                currentLaneID = 'lane_1'
                availabel_lane = {"currentLaneID":'lane_1',"leftLane": "lane_0",  "rightLane": ""}
                return availabel_lane, f"""The availabel lane of `{vid}` is `{currentLaneID}` and `{leftLane}`. `{currentLaneID}` is the current lane. `{leftLane}` is to the right of the current lane."""
            elif lane_id == ("a", "b", 0) or lane_id == ("b", "c", 0) or lane_id == ("c", "d", 0):
                currentLaneID = 'lane_0'
                availabel_lane = {"currentLaneID":'lane_0',"leftLane": "",  "rightLane": ""}
                return availabel_lane, f"""The availabel lane of `{vid}` is  `{currentLaneID}`. `{currentLaneID}` is the current lane."""
            else:
                currentLaneID = 'lane_1'
                availabel_lane = {"currentLaneID": 'lane_1', "leftLane": "", "rightLane": ""}
                return availabel_lane, f"""The availabel lane of `{vid}` is  `{currentLaneID}`. `{currentLaneID}` is the current lane."""
        else:
            currentLaneID = lane_id
            availabel_lane = {"currentLaneID": str(lane_id), "leftLane": "",  "rightLane": ""}
            return availabel_lane, f"""The availabel lane of `{vid}` is  `{currentLaneID}`. `{currentLaneID}` is the current lane."""

class getLaneInvolvedCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get Lane Involved Car',
             description="""useful whent want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
    def inference(self, laneID: str,  ego_veh, road, env) -> str:
        if laneID == None:
            print("Not a valid lane id! Make sure you have used the tool `Get Available Lanes` first.")
            return "Not a valid lane id! Make sure you have used the tool `Get Available Lanes` first."

        ego_laneID = which_lane(ego_veh.lane_index, env)  # ego information

        laneVehicles = []
        for i, vehi in enumerate(road.vehicles):
            if vehi != ego_veh:  # 剔除ego vehi
                other_laneID = which_lane(vehi.lane_index, env)
                if other_laneID == ego_laneID:
                    laneVehicles.append(vehi)

        # laneVehicles.sort(key=lambda x: x[1])
        ego_veh_destination = ego_veh.destination
        # laneVehicles = sorted(laneVehicles, key=lambda x: np.sqrt((x[1][0] - ego_veh_destination[0])**2 + (x[1][1] - ego_veh_destination[1])**2))  # TODO: lambda x: x[1][0] is for highway, bigger x equal closer to destination; while for conflict sce, closer to destination means front vehicle
        ego_veh_distance_to_destination = np.sqrt((ego_veh.position[0] - ego_veh_destination[0])**2 + (ego_veh.position[1] - ego_veh_destination[1])**2)
        distance = [np.sqrt((vehi.position[0] - ego_veh_destination[0])**2 + (vehi.position[1] - ego_veh_destination[1])**2) - ego_veh_distance_to_destination for vehi in laneVehicles if vehi != ego_veh and vehi.lane_index == ego_veh.lane_index]

        # find front vehicle
        rearing_distance, rearingCarIdx = min(
            ((d, i) for i, d in enumerate(distance) if d > 0),
            default=(float('inf'), -1),
            key=lambda x: x[0]
        )

        # find rear vehicle
        leading_distance, leadingCarIdx = max(
            ((d, i) for i, d in enumerate(distance) if d < 0),
            default=(-float('inf'), -1),
            key=lambda x: x[0]
        )

        # if surrounded by other vehicle
        if leadingCarIdx == -1 and rearingCarIdx == -1:
            return None, None, f'There is no car driving on {laneID}. This lane is safe.'
        elif leadingCarIdx == -1 and rearingCarIdx != -1:
            rearingCar = laneVehicles[rearingCarIdx]
            return None, rearingCar, f"{rearingCar} is driving on {laneID} which is the same lane as you are driving, and it's driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        elif leadingCarIdx != -1 and rearingCarIdx == -1:
            leadingCar = laneVehicles[leadingCarIdx]
            return leadingCar, None, f"{leadingCar} is driving at {round(leadingCar.speed, 1)}m/s on {laneID} which is the same lane as you are driving, and it's driving in front of the ego car for {round(distance[leadingCarIdx], 2)} meters. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        else:
            rearingCar = laneVehicles[rearingCarIdx]
            leadingCar = laneVehicles[leadingCarIdx]
            return leadingCar, rearingCar, f"{leadingCar} and {rearingCar} are driving on {laneID} which is the same lane as you are driving, and {leadingCar} is driving at {round(leadingCar.speed, 1)}m/s in front of the ego car for {round(distance[leadingCarIdx], 2)} meters, while {rearingCar} is driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."


class isChangeLaneConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Change Lane Confict With Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
    def inference(self, vid: str, ego_veh,laneID) -> str:
        # laneID, vid = inputs.replace(' ', '').split(',')
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ego_veh
        if veh.position[0] >= ego.position[0]:
            relativeSpeed = ego.speed - veh.speed
            if veh.position[0] - ego.position[0] - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."
        else:
            relativeSpeed = veh.speed - ego.speed
            if ego.position[0] - veh.position[0] - self.VEHICLE_LENGTH > self.TIME_HEAD_WAY * relativeSpeed:
                return f"change lane to `{laneID}` is safe with `{vid}`."
            else:
                return f"change lane to `{laneID}` may be conflict with `{vid}`, which is unacceptable."


class isAccelerationConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0
        self.acceleration = 4.0

    @prompts(name='Is Acceleration Conflict With Car',
             description="""useful when you want to know whether acceleration is safe with a specific car, ONLY when your decision is accelerate. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str, ego_veh) -> str:

        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ego_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            ego_destination = ego.destination
            ego_distance_to_destination = np.sqrt((ego.position[0] - ego_destination[0]) ** 2 + (ego_veh.position[1] - ego_destination[1]) ** 2)
            veh_distance_to_destination = np.sqrt((veh.position[0] - ego_destination[0]) ** 2 + (veh.position[1] - ego_destination[1]) ** 2)
            if ego_distance_to_destination >= veh_distance_to_destination:  # veh is leading of ego, can not accelerate too fast
                relativeSpeed = ego.speed + self.acceleration - veh.speed
                distance = ego_distance_to_destination - veh_distance_to_destination - self.VEHICLE_LENGTH * 2
                ttc = distance / relativeSpeed
                if ttc > 20:
                    return f"acceleration is safe with {vid}"
                elif 20 >= ttc > 10:
                    return f"acceleration may not safe with {vid}, should be careful if you want to accelerate"
                elif 10 >= ttc > 5:
                    return f'acceleration will cause danger, you can not accelerate'
                else:
                    return f'acceleration will cause serious danger, must decelerate.'
                # if distance > self.TIME_HEAD_WAY * relativeSpeed:
                #     return f"acceleration is safe with `{vid}`."
                # else:
                #     return f"acceleration may be conflict with `{vid}`, which is unacceptable."
            else:
                return f"acceleration is safe with {vid}"
        else:
            return f"acceleration is safe with {vid}"


class isKeepSpeedConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 5.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Keep Speed Conflict With Car',
             description="""useful when you want to know whether keep speed is safe with a specific car, ONLY when your decision is keep_speed. The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str, ego_veh) -> str:
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ego_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            ego_destination = ego.destination
            ego_distance_to_destination = np.sqrt((ego.position[0] - ego_destination[0]) ** 2 + (ego_veh.position[1] - ego_destination[1]) ** 2)
            veh_distance_to_destination = np.sqrt((veh.position[0] - ego_destination[0]) ** 2 + (veh.position[1] - ego_destination[1]) ** 2)
            if ego_distance_to_destination >= veh_distance_to_destination:
                relativeSpeed = ego.speed - veh.speed
                distance = ego_distance_to_destination - veh_distance_to_destination - self.VEHICLE_LENGTH * 2
                ttc = distance / relativeSpeed
                if ttc > 20:
                    return f"keep lane with current speed is safe with {vid}"
                elif 20 >= ttc > 10:
                    return f"keep lane with current speed may not safe with {vid}, should consider decelerate"
                elif 10 >= ttc > 5:
                    return f'keep lane with current speed will cause danger, you should consider decelerate'
                else:
                    return f'keep lane with current speed will cause serious danger, must decelerate.'
                # if distance > self.TIME_HEAD_WAY * relativeSpeed:
                #     return f"keep lane with current speed is safe with {vid}"
                # else:
                #     return f"keep lane with current speed may be conflict with {vid}, you need consider decelerate"
            else:
                return f"keep lane with current speed is safe with {vid}"
        else:
            return f"keep lane with current speed is safe with {vid}"


class isDecelerationSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 6.0

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str, ego_veh) -> str:
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego = ego_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            ego_destination = ego.destination
            ego_distance_to_destination = np.sqrt((ego.position[0] - ego_destination[0]) ** 2 + (ego_veh.position[1] - ego_destination[1]) ** 2)
            veh_distance_to_destination = np.sqrt((veh.position[0] - ego_destination[0]) ** 2 + (veh.position[1] - ego_destination[1]) ** 2)
            if veh_distance_to_destination >= ego_distance_to_destination:  # veh is the rearing vehicle of ego
                relativeSpeed = veh.speed - (ego.speed - self.deceleration)
                distance = veh_distance_to_destination - ego_distance_to_destination - self.VEHICLE_LENGTH
                ttc = distance / relativeSpeed
                if ttc > 20:
                    return f"deceleration with current speed is safe with {vid}"
                elif 20 >= ttc > 10:
                    return f"deceleration with current speed may not safe with {vid}"
                elif 10 >= ttc > 5:
                    return f'deceleratione with current speed will cause danger, if you have no other choice, try not to decelerate so fast as much as possible'
                else:
                    return f"deceleration with current speed may be conflict with {vid}, you should maintain speed or accelerate"

            else:
                return f"deceleration with current speed is safe with {vid}"
        else:
            return f"deceleration with current speed is safe with {vid}"


# ## Analysis obs

def available_action(toolModels, ego_veh, road, env, is_intersection = False):
    available_action_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableActions)), None)
    # Use tools to analyze the situation
    available_action = {}
    ego_vehicle_id = 'ego'
    available_lanes_analysis = available_action_tool.inference(ego_vehicle_id, ego_veh, road, env, is_intersection)
    available_action[available_action_tool] = available_lanes_analysis
    return available_action


def get_available_lanes(toolModels, ego_veh, road, env):
    available_lanes_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableLanes)), None)
    # Use tools to analyze the situation
    situation_analysis = {}
    ego_vehicle_id = 'ego'
    availabel_lane, available_lanes_analysis = available_lanes_tool.inference(ego_vehicle_id, ego_veh, road, env)
    situation_analysis[available_lanes_tool] = available_lanes_analysis
    return availabel_lane, situation_analysis


def get_involved_cars(toolModels, ego_veh, road, env, availabel_lane):
    lane_cars_info = {}
    lane_involved_car_tool = next((tool for tool in toolModels if isinstance(tool, getLaneInvolvedCar)), None)
    lane_ids = [value.strip() for value in availabel_lane.values() if value.strip()]
    lane_cars_id = {}
    for lane_id in lane_ids:
        leadingCar, rearingCar, cars_in_lane_info = lane_involved_car_tool.inference(lane_id, ego_veh, road, env)
        lane_cars_info[lane_id] = cars_in_lane_info
        lane_cars_id[lane_id] = {"leadingCar":leadingCar, "rearingCar":rearingCar}  # record, MDPVehicle #752: [91.02253718 10.5       ]

    return lane_cars_info, lane_cars_id


# function get info from available lanes
def extract_lanes_info(available_lanes_info):
    lanes = {
        'current': None,
        'left': None,
        'right': None
    }

    parts = available_lanes_info.split(". ")
    for part in parts:
        if "is the current lane" in part:
            lanes['current'] = part.split("`")[1]  # Extract the current lane
        elif "to the left of the current lane" in part:
            lanes['left'] = part.split("`")[1]  # Extract the left adjacent lane
        elif "to the right of the current lane" in part:
            lanes['right'] = part.split("`")[1]  # Extract the right adjacent lane

    return lanes


# These two functions get info from get_involved_cars

def extract_car_id_from_info(lane_info):
    # Extracts the car ID from the lane information string
    if "is driving" in lane_info:
        parts = lane_info.split()
        car_id_index = parts.index("is") - 1
        return parts[car_id_index]
    return None


def extract_lane_and_car_ids(lanes_info, lane_cars_info):
    lane_car_ids = {
        'current_lane': {'lane_id': None, 'car_id': None},
        'left_lane': {'lane_id': None, 'car_id': None},
        'right_lane': {'lane_id': None, 'car_id': None}
    }
    current_lane_id = lanes_info['current']
    left_lane_id = lanes_info['left']
    right_lane_id = lanes_info['right']

    # Extract car ID for the left adjacent lane, if it exists
    if current_lane_id and current_lane_id in lane_cars_info:
        current_lane_info = lane_cars_info[current_lane_id]
        current_car_id = extract_car_id_from_info(current_lane_info)
        lane_car_ids['current_lane'] = {'lane_id': current_lane_id, 'car_id': current_car_id}

    # Extract car ID for the left adjacent lane, if it exists
    if left_lane_id and left_lane_id in lane_cars_info:
        left_lane_info = lane_cars_info[left_lane_id]
        left_car_id = extract_car_id_from_info(left_lane_info)
        lane_car_ids['left_lane'] = {'lane_id': left_lane_id, 'car_id': left_car_id}

    # Extract car ID for the right adjacent lane, if it exists
    if right_lane_id and right_lane_id in lane_cars_info:
        right_lane_info = lane_cars_info[right_lane_id]
        right_car_id = extract_car_id_from_info(right_lane_info)
        lane_car_ids['right_lane'] = {'lane_id': right_lane_id, 'car_id': right_car_id}

    return lane_car_ids


# F
def assess_lane_change_safety(toolModels, lane_car_ids, availabel_lane, ego_veh):
    lane_change_tool = next((tool for tool in toolModels if isinstance(tool, isChangeLaneConflictWithCar)), None)
    safety_assessment = {
        'left_lane_change_safe': True,
        'right_lane_change_safe': True
    }
    #lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
    #availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}
    current_lane_car_id = []
    if availabel_lane['leftLane']:
        currentLaneID = availabel_lane['leftLane']
        current_lane_ca = lane_car_ids[currentLaneID]
        for key, value in current_lane_ca.items():
            if value is not None:
                current_lane_car_id = value  # IDMVehicle #456: [173.94198546   0.        ]
                leading_or_rearing = key  # 'rearingCar'
        left_lane_safety = lane_change_tool.inference(current_lane_car_id, ego_veh, currentLaneID)
        safety_assessment['left_lane_change_safe'] = 'safe' in left_lane_safety
    else:
        # If no car is in the left lane, consider it safe to change
        safety_assessment['left_lane_change_safe'] = True

    return safety_assessment


def check_safety_in_current_lane(toolModels, lane_cars_id, availabel_lane, ego_veh):
    # lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
    # availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}
    safety_analysis = {
        'acceleration_conflict': None,
        'keep_speed_conflict': None,
        'deceleration_conflict': None
    }

    # Extract tools from toolModels
    acceleration_tool = next((tool for tool in toolModels if isinstance(tool, isAccelerationConflictWithCar)), None)
    keep_speed_tool = next((tool for tool in toolModels if isinstance(tool, isKeepSpeedConflictWithCar)), None)
    deceleration_tool = next((tool for tool in toolModels if isinstance(tool, isDecelerationSafe)), None)
    currentLaneID = availabel_lane['currentLaneID']
    current_lane_ca = lane_cars_id[currentLaneID]
    current_lane_car_id = None
    for key, value in current_lane_ca.items():
        if value is not None:
            current_lane_car_id = value #IDMVehicle #456: [173.94198546   0.        ]
            leading_or_rearing = key #'rearingCar'

    # current_lane_car_id = lane_and_car_ids['current_lane']['car_id']

    if current_lane_car_id:
        # Check for conflicts if there is a car in the current lane
        if leading_or_rearing == "leadingCar":  # if there is leading vehicle for ego
            safety_analysis['acceleration_conflict'] = acceleration_tool.inference(current_lane_car_id, ego_veh)
        if leading_or_rearing == "rearingCar" or leading_or_rearing == "leadingCar":
            safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(current_lane_car_id, ego_veh)
        if leading_or_rearing == "rearingCar":  # if there is rearing vehicle for ego
            safety_analysis['deceleration_conflict'] = deceleration_tool.inference(current_lane_car_id, ego_veh)

    return safety_analysis

def cal_ttcp(speed_limit, veh_dis2cp, veh_v):
    MAX_ACCELERATION = 2  # in highway controlled vehicle acc is 2
    t_acc2max = (speed_limit - veh_v) / MAX_ACCELERATION
    dis_acc2max = veh_v * t_acc2max + 0.5 * MAX_ACCELERATION * t_acc2max ** 2
    if dis_acc2max < veh_dis2cp:
        t_left = (veh_dis2cp - dis_acc2max) / speed_limit
        ttcp = t_acc2max + t_left
    else:
        v = np.sqrt(veh_v ** 2 + 2 * veh_dis2cp * MAX_ACCELERATION)
        ttcp = (v - veh_v) / MAX_ACCELERATION
    return ttcp

def check_safety_with_conflict_vehicles(ego_veh, negotiation_results, conflicting_info, env):
    '''calculate TTCP to determine dangerous, then based on negotiation results provided by LLM, if is leading vehicle acceleration conflict TRUe'''
    safety_analysis = {
        'acceleration_conflict': None,
        'keep_speed_conflict': None,
        'deceleration_conflict': None
    }
    most_dangerous_info = {'delta ttcp': None, 'distance to conflict': None, 'speed': None, 'distance to conflict (others)': None, 'speed (others)': None}

    pattern = re.compile(r"- You have conflict with (MDPVehicle #[0-9]+|IDMVehicle #[0-9]+). It is suggested that you should passes second.")  # TODO: must same pattern as negotiation_results
    vehicles_pass_second = pattern.findall(negotiation_results)  # vehicles_pass_second = [MDPVehicle #800] without pos
    speed_limit = env.road.network.get_lane(ego_veh.lane_index).speed_limit
    dangerous_level = 0
    for vehicle in vehicles_pass_second:
        for conflict_group in conflicting_info:
            if (conflict_group['vehicle_i'] == ego_veh and str(conflict_group['vehicle_j']).split(':')[0].strip() == vehicle) or \
                    (conflict_group['vehicle_j'] == ego_veh and str(conflict_group['vehicle_i']).split(':')[0].strip() == vehicle):
                ttcp_i = cal_ttcp(speed_limit, conflict_group['vehicle_i distance to conflict'], conflict_group['vehicle_i speed'])
                ttcp_j = cal_ttcp(speed_limit, conflict_group['vehicle_j distance to conflict'], conflict_group['vehicle_j speed'])
                ttcp = abs(ttcp_j - ttcp_i)
                if ttcp > 8:
                    level = 0
                elif 8 >= ttcp > 5:
                    level = 1
                elif 5 >= ttcp > 2:
                    level = 2
                else:
                    level = 3
                if level > dangerous_level:
                    dangerous_level = level
                    most_dangerous_info['delta ttcp'] = ttcp
                    if conflict_group['vehicle_i'] == ego_veh:
                        most_dangerous_info['distance to conflict'] = conflict_group['vehicle_i distance to conflict']
                        most_dangerous_info['speed'] = conflict_group['vehicle_i speed']
                        most_dangerous_info['distance to conflict (others)'] = conflict_group['vehicle_i distance to conflict']
                        most_dangerous_info['speed (others)'] = conflict_group['vehicle_j speed']
                    else:
                        most_dangerous_info['distance to conflict'] = conflict_group['vehicle_j distance to conflict']
                        most_dangerous_info['speed'] = conflict_group['vehicle_j speed']
                        most_dangerous_info['distance to conflict (others)'] = conflict_group['vehicle_i distance to conflict']
                        most_dangerous_info['speed (others)'] = conflict_group['vehicle_i speed']

    if dangerous_level == 0:
        safety_analysis['acceleration_conflict'] = f'acceleration is safe'
    if dangerous_level == 1:
        safety_analysis['acceleration_conflict'] = f'acceleration may not safe, should be careful if you want to accelerate'
    if dangerous_level == 2:
        safety_analysis['acceleration_conflict'] = f'acceleration will cause danger, you can not accelerate'
    if dangerous_level == 3:
        safety_analysis['acceleration_conflict'] = f'acceleration will cause serious danger, must decelerate.'  # You output decision have to be SLOWER (note that it is in capital letters)!!!
    return safety_analysis, most_dangerous_info


def format_training_info(available_actions_msg, lanes_info_msg, all_lane_info_msg, lanes_adjacent_info, cars_near_lane, lane_change_safety, current_lane_safety, conflict_safety, most_dangerous_info):  # msg0, msg1, msg2, availabel_lane, lane_cars_id, safety_assessment, safety_msg
    formatted_message = ""

    # Add available actions information
    formatted_message += "Available Actions:\n"
    for tool, action_info in available_actions_msg.items():
        formatted_message += f"- {action_info}\n"

    # Add information about lanes
    formatted_message += "\nLane Information:\n"
    formatted_message += f"- Current Lane ID: {lanes_adjacent_info['currentLaneID']}\n"
    formatted_message += f"- Left Adjacent Lane ID: {lanes_adjacent_info['leftLane'] or 'None'}\n"
    formatted_message += f"- Right Adjacent Lane ID: {lanes_adjacent_info['rightLane'] or 'None'}\n"
    formatted_message += f"- Adjacent Lane ID is None stands for adjacent Lane not exist, you can only drive on current lane\n"

    # Add details about vehicles in each lane
    formatted_message += "\nVehicles in the same the Lane with you:\n"
    for lane_id, car_info in all_lane_info_msg.items():
        formatted_message += f"- Lane ID {lane_id}: {car_info}\n"


    # Safety assessment for lane changes
    formatted_message += "\nSafety Assessment for Lane Changes:\n"
    try:
        formatted_message += f"- Left Lane Change: {'Safe' if lane_change_safety['left_lane_change_safe'] else 'Not Safe'}\n"
        formatted_message += f"- Right Lane Change: {'Safe' if lane_change_safety['right_lane_change_safe'] else 'Not Safe'}\n"
    except:
        formatted_message += lane_change_safety

    # Safety assessment in the current lane
    formatted_message += "\nSafety Assessment in Current Lane:\n"
    for action, safety in current_lane_safety.items():
        formatted_message += f"- {action.capitalize().replace('_', ' ')}: {safety}\n"

    # Safety assessment with conflict vehicles
    formatted_message += "\nSafety Assessment with Vehicle Not in Current Lane but Have Conflict with Ego Vehicle:\n"
    for action, safety in conflict_safety.items():
        formatted_message += f"- {action.capitalize().replace('_', ' ')}: {safety}\n"

    # Most dangerous conflict info which same pattern as memory
    if most_dangerous_info['delta ttcp'] is not None:
        formatted_message += f"\n Currently, the most dangerous collision information with the ego vehicle is as follows: \n"
        formatted_message += f"The time to collision is {most_dangerous_info['delta ttcp']}s, ego vehicle's the distance to the collision point is {most_dangerous_info['distance to conflict']} m, " \
                             f"and the current speed of the ego vehicle is {most_dangerous_info['speed']}, the distance to the collision point of conflict vehicle is {most_dangerous_info['distance to conflict (others)']} m, its speed is {most_dangerous_info['speed (others)']}\n"
    else:
        formatted_message += f"\n Currently, ego vehicle do not have conflict \n"
        formatted_message += f"\n Conflict info is empty \n"
    # print(formatted_message)
    return formatted_message
