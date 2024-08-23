from .merge_env import Scenario


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
    """

    def get_traffic_rules(self):
        return self.TRAFFIC_RULES

    def get_decision_cautions(self):
        return self.DECISION_CAUTIONS

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

# **Use following cell if using a second best result.**

# def send_to_chatgpt(last_action, current_scenario, sce):
#     client = OpenAI(api_key=API_KEY)
#
#     action_id = int(last_action[0])  # Convert to integer
#     message_prefix = pre_prompt.SYSTEM_MESSAGE_PREFIX
#     traffic_rules = pre_prompt.get_traffic_rules()
#     decision_cautions = pre_prompt.get_decision_cautions()
#     action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
#     action_description = ACTIONS_DESCRIPTION.get(action_id, "No description available")
#
#     prompt = (f"{message_prefix}"
#               f"You, the 'ego' car, are now driving on a highway. You have already driven for {sce.frame} seconds.\n"
#               f"The decision made by the agent LAST time step was `{action_name}` ({action_description}).\n\n"
#               "Here is the current scenario:\n"
#               f"{current_scenario}\n\n"
#               "There are several rules you need to follow when you drive on a highway:\n"
#               f"{traffic_rules}\n\n"
#               "Here are your attention points:\n"
#               f"{decision_cautions}\n\n"
#               "Please generate one best answer and one second best answer for this situation. Once you make a final decision, output it in the following format:\n"
#               "```\n"
#               "Final Answer: \n"
#               "    \"decision1\": {\"<ego car's 1 st decision, ONE of the available actions>\"},\n"
#               "    \"decision2\": {\"<ego car's 2 nd decision, ONE of the available actions>\"},\n"
#               "```\n")
#     print(prompt)
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo-1106",
#         messages=[
#             {"role": "system", "content": prompt},
#         ]
#     )
#
#     return completion.choices[0].message
# ## Customization TOOLs
from typing import Any


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator

def which_lane(land_index):
    "input:j,k, 0. output:lane_1"
    if land_index == ("j", "k", 0) or land_index == ("k", "b", 0) or land_index == (
                     "b", "c", 1):
        return "lane_1"
    elif land_index == ("a", "b", 0) or land_index == ("b", "c", 0) or land_index == (
                     "c", "d", 0):
        return "lane_0"
    else:
        print("err:land_index is not belong to lane_0 or lane_1")
        return "land_index is not belong to lane_0 or lane_1"


class getAvailableActions:
    # def __init__(self, ) -> None:

    @prompts(name='Get Available Actions',
             description="""Useful before you make decisions, this tool let you know what are your available actions in this situation. The input to this tool should be 'ego'.""")
    def _get_available_actions1(self, vehicle, env_copy):
        """
        Get the list of currently available actions.
        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.
        :return: the list of available actions
        """
        # if not isinstance(self.action_type, DiscreteMetaAction):
        #     raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [env_copy.ACTIONS_ALL['IDLE']]
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

    def inference(self, input: str, ege_veh, road, env) -> str:
        outputPrefix = 'You can ONLY use one of the following actions: \n '
        # availableActions = self.env.get_available_actions() _get_available_actions
        availableActions = self._get_available_actions1(ege_veh, env)
        # available_action = env._get_available_actions(self.controlled_vehicles[i], self)
        # availableActions = [0, 1, 2, 3, 4]
        for action in availableActions:
            outputPrefix += ACTIONS_ALL[action] + \
                            '--' + ACTIONS_DESCRIPTION[action] + '; \n'
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
      Step 3: If you find There is no car driving on your "current lane" you can drive faster ! but not too fast to follow the traffic rules.
      Step 4: If you want to make lane change consider :"Safety Assessment for Lane Changes:" Safe means it is safe to change ,If you want to do IDLE, FASTER, SLOWER, you should consider "Safety Assessment in Current Lane:"
      """
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
    def inference(self, vid: str, ege_veh, road, env) -> str:
        # veh = self.sce.vehicles[vid]
        # currentLaneID = veh.lane_id
        lane_id = ege_veh.lane_index
        availabel_lane = {}
        if lane_id == ("b", "c", 1):
            leftLane = 'lane_0'
            currentLaneID = 'lane_1'
            availabel_lane = {"currentLaneID":'lane_1',"leftLane": "lane_0",  "rightLane": ""}
            return availabel_lane, f"""The availabel lane of `{vid}` is `{currentLaneID}` and `{leftLane}`. `{currentLaneID}` is the current lane. `{leftLane}` is to the right of the current lane."""
        elif lane_id == ("a", "b", 0) or lane_id == ("b", "c", 0) or lane_id == (
                        "c", "d", 0):
            currentLaneID = 'lane_0'
            availabel_lane = {"currentLaneID":'lane_0',"leftLane": "",  "rightLane": ""}
            return availabel_lane, f"""The availabel lane of `{vid}` is  `{currentLaneID}`. `{currentLaneID}` is the current lane."""
        else:
            currentLaneID = 'lane_1'
            availabel_lane = {"currentLaneID": 'lane_1', "leftLane": "", "rightLane": ""}
            return availabel_lane, f"""The availabel lane of `{vid}` is  `{currentLaneID}`. `{currentLaneID}` is the current lane."""

class getLaneInvolvedCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce

    @prompts(name='Get Lane Involved Car',
             description="""useful whent want to know the cars may affect your action in the certain lane. Make sure you have use tool `Get Available Lanes` first. The input is a string, representing the id of the specific lane you want to drive on, DONNOT input multiple lane_id once.""")
    def inference(self, laneID: str,  ege_veh, road, env) -> str:
        if laneID not in {'lane_0', 'lane_1'}:
            return "Not a valid lane id! Make sure you have used the tool `Get Available Lanes` first."

        ego_laneID = which_lane(ege_veh.lane_index) #ego information

        laneVehicles = []  # 存储同一车道上的其他车辆信息
        for i, vehi in enumerate(road.vehicles):
            if vehi != ege_veh: #剔除ego vehi
                other_laneID = which_lane(vehi.lane_index)
                if other_laneID == ego_laneID:  # 判断车辆是否在目标车道上
                    laneVehicles.append((vehi, vehi.position))  # 存储车辆的id和在车道上的位置
        # 按车辆在车道上的位置进行排序
        # laneVehicles.sort(key=lambda x: x[1])
        laneVehicles = sorted(laneVehicles, key=lambda x: x[1][0])
        leadingCarIdx = -1  # 初始化前车索引为-1，表示没有前车
        # 确定前方是否有车辆
        for i in range(len(laneVehicles)):
            vp = laneVehicles[i]
            if vp[1][0] >= ege_veh.position[0]:  # 找到前方第一辆车的索引
                leadingCarIdx = i
                break
        if leadingCarIdx == -1:  # 如果没有前车
            try:
                rearingCar = laneVehicles[-1][0]  # 获取最后一辆车的id
            except IndexError:  # 如果没有车辆
                return None, None, f'There is no car driving on {laneID}. This lane is safe. You do not need to check for any vehicles for safety! You can drive on this lane as fast as you can.'
            # 返回最后一辆车的信息
            return None, rearingCar, f"{rearingCar} is driving on {laneID}, and it's driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        elif leadingCarIdx == 0:  # 如果第一辆车s是前车
            leadingCar = laneVehicles[0][0]  # 获取前车的id
            distance = round(leadingCar.position[0] - ege_veh.position[0], 2)  # 计算与前车的距离
            leading_car_vel = round(leadingCar.speed, 1)  # 获取前车的速度
            # 返回前车的信息
            return leadingCar, None, f"{leadingCar} is driving at {leading_car_vel}m/s on {laneID}, and it's driving in front of the ego car for {distance} meters. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        else:  # 如果存在前车和后车
            leadingCar = laneVehicles[leadingCarIdx][0]  # 获取前车的id
            rearingCar = laneVehicles[leadingCarIdx - 1][0]  # 获取后车的id
            distance = round(leadingCar.position[0] - ege_veh.position[0], 2)  # 计算与前车的距离
            leading_car_vel = round(leadingCar.speed, 1)  # 获取前车的速度
            # 返回前车和后车的信息
            return leadingCar, rearingCar,f"{leadingCar} and {rearingCar} are driving on {laneID}, and {leadingCar} is driving at {leading_car_vel}m/s in front of the ego car for {distance} meters, while {rearingCar} is driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."



"""
        # 检查 laneID 是否有效
        if laneID not in {'lane_0', 'lane_1'}:
            return "Not a valid lane id! Make sure you have used the tool `Get Available Lanes` first."
        ego = self.sce.vehicles['ego']  # 获取自车信息
        laneVehicles = []  # 存储同一车道上的其他车辆信息
        # 筛选出同一车道上的车辆
        for vk, vv in self.sce.vehicles.items():
            if vk != 'ego':  # 排除自车
                if vv.lane_id == laneID:  # 判断车辆是否在目标车道上
                    laneVehicles.append((vv.id, vv.lanePosition))  # 存储车辆的id和在车道上的位置
        # 按车辆在车道上的位置进行排序
        laneVehicles.sort(key=lambda x: x[1])
        leadingCarIdx = -1  # 初始化前车索引为-1，表示没有前车
        # 确定前方是否有车辆
        for i in range(len(laneVehicles)):
            vp = laneVehicles[i]
            if vp[1] >= ego.lanePosition:  # 找到前方第一辆车的索引
                leadingCarIdx = i
                break

        if leadingCarIdx == -1:  # 如果没有前车
            try:
                rearingCar = laneVehicles[-1][0]  # 获取最后一辆车的id
            except IndexError:  # 如果没有车辆
                return f'There is no car driving on {laneID}. This lane is safe. You do not need to check for any vehicles for safety! You can drive on this lane as fast as you can.'
            # 返回最后一辆车的信息
            return f"{rearingCar} is driving on {laneID}, and it's driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        elif leadingCarIdx == 0:  # 如果前车是第一辆车
            leadingCar = laneVehicles[0][0]  # 获取前车的id
            distance = round(laneVehicles[0][1] - ego.lanePosition, 2)  # 计算与前车的距离
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed, 1)  # 获取前车的速度
            # 返回前车的信息
            return f"{leadingCar} is driving at {leading_car_vel}m/s on {laneID}, and it's driving in front of the ego car for {distance} meters. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
        else:  # 如果存在前车和后车
            leadingCar = laneVehicles[leadingCarIdx][0]  # 获取前车的id
            rearingCar = laneVehicles[leadingCarIdx - 1][0]  # 获取后车的id
            distance = round(laneVehicles[leadingCarIdx][1] - ego.lanePosition, 2)  # 计算与前车的距离
            leading_car_vel = round(self.sce.vehicles[leadingCar].speed, 1)  # 获取前车的速度
            # 返回前车和后车的信息
            return f"{leadingCar} and {rearingCar} are driving on {laneID}, and {leadingCar} is driving at {leading_car_vel}m/s in front of the ego car for {distance} meters, while {rearingCar} is driving behind the ego car. You need to make sure that your actions do not conflict with each of the vehicles mentioned."
"""

class isChangeLaneConflictWithCar:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0

    @prompts(name='Is Change Lane Confict With Car',
             description="""useful when you want to know whether change lane to a specific lane is confict with a specific car, ONLY when your decision is change_lane_left or change_lane_right. The input to this tool should be a string of a comma separated string of two, representing the id of the lane you want to change to and the id of the car you want to check.""")
    def inference(self, vid: str, ege_veh,laneID) -> str:
        # laneID, vid = inputs.replace(' ', '').split(',')
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ege_veh
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
    def inference(self, vid: str, ege_veh) -> str:

        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ege_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            if veh.position[0] >= ego.position[0]:
                relativeSpeed = ego.speed + self.acceleration - veh.speed
                distance = veh.position[0] - ego.position[0] - self.VEHICLE_LENGTH * 2
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"acceleration is safe with `{vid}`."
                else:
                    return f"acceleration may be conflict with `{vid}`, which is unacceptable."
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
    def inference(self, vid: str, ege_veh) -> str:
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ege_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            if veh.position[0] >= ego.position[0]:
                relativeSpeed = ego.speed - veh.speed
                distance = veh.position[0] - ego.position[0] - self.VEHICLE_LENGTH * 2
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"keep lane with current speed is safe with {vid}"
                else:
                    return f"keep lane with current speed may be conflict with {vid}, you need consider decelerate"
            else:
                return f"keep lane with current speed is safe with {vid}"
        else:
            return f"keep lane with current speed is safe with {vid}"


class isDecelerationSafe:
    def __init__(self, sce: Scenario) -> None:
        self.sce = sce
        self.TIME_HEAD_WAY = 3.0
        self.VEHICLE_LENGTH = 5.0
        self.deceleration = 3.0

    @prompts(name='Is Deceleration Safe',
             description="""useful when you want to know whether deceleration is safe, ONLY when your decision is decelerate.The input to this tool should be a string, representing the id of the car you want to check.""")
    def inference(self, vid: str, ege_veh) -> str:
        if vid not in self.sce.road_info.vehicles:
            return "Your input is not a valid vehicle id, make sure you use `Get Lane Involved Car` tool first!"
        if vid == 'ego':
            return "You are checking the acceleration of ego car, which is meaningless, input a valid vehicle id please!"
        # veh = self.sce.vehicles[vid]
        # ego = self.sce.vehicles['ego']
        veh = vid
        ego =  ege_veh
        if veh.lane_index != ego.lane_index:
            return f'{vid} is not in the same lane with ego, please call `Get Lane Involved Car` and rethink your input.'
        if veh.lane_index == ego.lane_index:
            if veh.position[0] >= ego.position[0]:
                relativeSpeed = ego.speed - veh.speed - self.deceleration
                distance = veh.position[0] - ego.position[0] - self.VEHICLE_LENGTH
                if distance > self.TIME_HEAD_WAY * relativeSpeed:
                    return f"deceleration with current speed is safe with {vid}"
                else:
                    return f"deceleration with current speed may be conflict with {vid}, if you have no other choice, slow down as much as possible"
            else:
                return f"deceleration with current speed is safe with {vid}"
        else:
            return f"deceleration with current speed is safe with {vid}"


# ## Analysis obs

def available_action(toolModels, ege_veh, road, env):
    available_action_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableActions)), None)
    # Use tools to analyze the situation
    available_action = {}
    ego_vehicle_id = 'ego'
    available_lanes_analysis = available_action_tool.inference(ego_vehicle_id, ege_veh, road, env)
    available_action[available_action_tool] = available_lanes_analysis

    return available_action


def get_available_lanes(toolModels, ege_veh, road, env):
    available_lanes_tool = next((tool for tool in toolModels if isinstance(tool, getAvailableLanes)), None)
    # Use tools to analyze the situation
    situation_analysis = {}
    ego_vehicle_id = 'ego'
    availabel_lane, available_lanes_analysis = available_lanes_tool.inference(ego_vehicle_id, ege_veh, road, env)
    situation_analysis[available_lanes_tool] = available_lanes_analysis

    return  availabel_lane, situation_analysis


def get_involved_cars(toolModels, ege_veh, road, env, availabel_lane):
    lane_cars_info = {}
    lane_involved_car_tool = next((tool for tool in toolModels if isinstance(tool, getLaneInvolvedCar)), None)
    lane_ids = [value.strip() for value in availabel_lane.values() if value.strip()]
    lane_cars_id = {}
    for lane_id in lane_ids:
        leadingCar, rearingCar, cars_in_lane_info = lane_involved_car_tool.inference(lane_id, ege_veh, road, env)
        lane_cars_info[lane_id] = cars_in_lane_info
        lane_cars_id[lane_id] = {"leadingCar":leadingCar, "rearingCar":rearingCar} #record, MDPVehicle #752: [91.02253718 10.5       ]

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
def assess_lane_change_safety(toolModels, lane_car_ids, availabel_lane, ege_veh):
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
        left_lane_safety = lane_change_tool.inference(current_lane_car_id, ege_veh, currentLaneID)
        safety_assessment['left_lane_change_safe'] = 'safe' in left_lane_safety
    else:
        # If no car is in the left lane, consider it safe to change
        safety_assessment['left_lane_change_safe'] = True

    return safety_assessment

#
# def assess_lane_change_safety(toolModels, lane_cars_id, availabel_lane, ege_veh):
#     # lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
#     # availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}
#     safety_assessment = {
#         'left_lane_change_safe': True,
#         'right_lane_change_safe': True
#     }
#
#
#     if availabel_lane['leftLane']:
#         safety_analysis = {
#             'acceleration_conflict': None,
#             'keep_speed_conflict': None,
#             'deceleration_conflict': None
#         }
#         # Extract tools from toolModels
#         acceleration_tool = next((tool for tool in toolModels if isinstance(tool, isAccelerationConflictWithCar)), None)
#         keep_speed_tool = next((tool for tool in toolModels if isinstance(tool, isKeepSpeedConflictWithCar)), None)
#         deceleration_tool = next((tool for tool in toolModels if isinstance(tool, isDecelerationSafe)), None)
#         leftLane = availabel_lane['leftLane']
#         leftLane_ca = lane_cars_id[leftLane]
#         for key, value in leftLane_ca.items():
#             if value is not None:
#                 leftLane_lane_car_id = value  # IDMVehicle #456: [173.94198546   0.        ]
#                 leading_or_rearing = key  # 'rearingCar'
#
#         # current_lane_car_id = lane_and_car_ids['current_lane']['car_id']
#
#         # if current_lane_car_id:
#         # Check for conflicts if there is a car in the current lane
#         if leading_or_rearing == "leadingCar":  # 如果ego前面有车
#             safety_analysis['acceleration_conflict'] = acceleration_tool.inference(leftLane_lane_car_id, ege_veh)
#         if leading_or_rearing == "rearingCar" or leading_or_rearing == "leadingCar":
#             safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(leftLane_lane_car_id, ege_veh)
#         if leading_or_rearing == "rearingCar":  # 如果ego后面有车
#             safety_analysis['deceleration_conflict'] = deceleration_tool.inference(leftLane_lane_car_id, ege_veh)
#
#
#         # currentLaneID = availabel_lane['leftLane']
#         # current_lane_ca = lane_cars_id[currentLaneID]
#         # for key, value in current_lane_ca.items():
#         #     if value is not None:
#         #         current_lane_car_id = value  # IDMVehicle #456: [173.94198546   0.        ]
#         #         leading_or_rearing = key  # 'rearingCar'
#         # left_lane_safety = lane_change_tool.inference(lane_car_ids)
#         safety_assessment['left_lane_change_safe'] = 'safe' in left_lane_safety
#     else:
#         # If no car is in the left lane, consider it safe to change
#         safety_assessment['left_lane_change_safe'] = True
#     return safety_assessment

def check_safety_in_current_lane(toolModels, lane_cars_id, availabel_lane, ege_veh):
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
        if leading_or_rearing == "leadingCar":#如果ego前面有车
            safety_analysis['acceleration_conflict'] = acceleration_tool.inference(current_lane_car_id, ege_veh)
        if leading_or_rearing == "rearingCar" or leading_or_rearing == "leadingCar":
            safety_analysis['keep_speed_conflict'] = keep_speed_tool.inference(current_lane_car_id, ege_veh)
        if leading_or_rearing == "rearingCar":#如果ego后面有车
            safety_analysis['deceleration_conflict'] = deceleration_tool.inference(current_lane_car_id, ege_veh)

    return safety_analysis


def format_training_info(available_actions_msg, lanes_info_msg, all_lane_info_msg, lanes_adjacent_info,
                         cars_near_lane, lane_change_safety, current_lane_safety):
    formatted_message = ""

    # Add available actions information
    formatted_message += "Available Actions:\n"
    for tool, action_info in available_actions_msg.items():
        formatted_message += f"- {action_info}\n"

    # Add information about lanes
    formatted_message += "\nLane Information:\n"
    formatted_message += f"- Current Lane: {lanes_adjacent_info['currentLaneID']}\n"
    formatted_message += f"- Left Adjacent Lane: {lanes_adjacent_info['leftLane'] or 'None'}\n"
    formatted_message += f"- Right Adjacent Lane: {lanes_adjacent_info['rightLane'] or 'None'}\n"

    # Add details about vehicles in each lane
    formatted_message += "\nOther Vehicles in all the Lanes:\n"
    for lane_id, car_info in all_lane_info_msg.items():
        formatted_message += f"- {lane_id}: {car_info}\n"

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

    return formatted_message
