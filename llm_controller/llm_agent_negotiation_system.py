from llm_controller.prompt_llm import *
from llm_controller.Scenario_description import Scenario
from openai import OpenAI
import numpy as np
import highway_env
import json

api_key = "your key here"

class LlmAgent_negotiation_module():
    def __init__(self, env):
        self.sce = Scenario(env.road, vehicleCount=len(env.controlled_vehicles))
        self.controlled_vehicles = env.controlled_vehicles
        self.pre_prompt = PRE_DEF_PROMPT()
        self.toolModels = [
            getAvailableActions(),
            getAvailableLanes(self.sce),
            getLaneInvolvedCar(self.sce),
            isChangeLaneConflictWithCar(self.sce),
            isAccelerationConflictWithCar(self.sce),
            isKeepSpeedConflictWithCar(self.sce),
            isDecelerationSafe(self.sce),
        ]

    def llm_controller_run(self, env):
        conflict = self.detect_conflicts(env)
        llm_negotiation = self.send_to_chatgpt(env, conflict)
        return llm_negotiation

    def get_scene_name(self, env):
        scene_name = env.spec.id
        # 使用正则表达式提取 'merge'、'intersection' 或 'highway' 部分
        match = re.search(r'(merge|intersection|highway)', scene_name)
        simplified_scene_name = match.group(0) if match else 'unknown'
        return simplified_scene_name

    def detect_conflicts(self, env):
        vehicles = []
        conflicts = []
        scene_name = self.get_scene_name(env)
        if scene_name == 'intersection':
            vehicle_list = self.controlled_vehicles  # TODO: if intersection env use self.controlled_vehicles else use env.road.vehicles; reason: to align with other cooperative driving method in experiment CAV in intersection can not negotiate with HDV
        elif scene_name == 'merge' or scene_name == 'highway':
            vehicle_list = env.road.vehicles
        else:
            raise ValueError('no such scenario')

        for i in range(len(vehicle_list)):
            vehicle_i = vehicle_list[i]
            vehicles.append(vehicle_i)
            conflict_with_i = []
            distance_with_i = []
            for j in range(len(vehicle_list)):
                if i != j:
                    vehicle_j = vehicle_list[j]
                    is_conflicts, distance_to_conflict_point = self.is_conflict(env, vehicle_i, vehicle_j)
                    if is_conflicts:
                        conflict_with_i.append(vehicle_j)
                        distance_with_i.append(distance_to_conflict_point)
            conflicts.append(dict(zip(conflict_with_i, distance_with_i)))
        return dict(zip(vehicles, conflicts))  # to search if conflict: inter in detect_conflicts(env)[ego] == Ture; to get the distance of ego to the conflict point with inter: detect_conflicts(env)[ego][inter]

    def is_conflict(self, env, vehicle_i, vehicle_j):
        # Implement your conflict detection logic
        route_i = self.record_route_xy(env, vehicle_i)
        route_j = self.record_route_xy(env, vehicle_j)
        if len(route_i) != 0 and len(route_j) != 0 and vehicle_i.lane_index != vehicle_j.lane_index:
            points_distance_between_routes = [np.min((route_i[point, 0] - route_j[:, 0]) ** 2 + (route_i[point, 1] - route_j[:, 1]) ** 2) for point in range(len(route_i))]
            distance_between_routes = np.min(points_distance_between_routes)  # 两个轨迹线的距离
            if distance_between_routes < 1:
                # conflict_point_index = np.argmin(points_distance_between_routes)
                conflict_point_index = np.where(np.array(points_distance_between_routes) < 1)[0][0]  # first element < 1
                current_position, conflict_point_position = vehicle_i.position, route_i[conflict_point_index]
                distance_to_conflict_point = np.sqrt((current_position[0] - conflict_point_position[0]) ** 2 + (current_position[1] - conflict_point_position[1]) ** 2)
                vehicle_i_distance_to_destination = np.sqrt((current_position[0] - vehicle_i.destination[0]) ** 2 + (current_position[1] - vehicle_i.destination[1]) ** 2)
                vehicle_i_conflict_point_distance_to_destination = np.sqrt((conflict_point_position[0] - vehicle_i.destination[0]) ** 2 + (conflict_point_position[1] - vehicle_i.destination[1]) ** 2)
                vehicle_j_distance_to_destination = np.sqrt((vehicle_j.position[0] - vehicle_j.destination[0]) ** 2 + (vehicle_j.position[1] - vehicle_j.destination[1]) ** 2)
                vehicle_j_conflict_point_distance_to_destination = np.sqrt((conflict_point_position[0] - vehicle_j.destination[0]) ** 2 + (conflict_point_position[1] - vehicle_j.destination[1]) ** 2)

                # info of vehicle j, to determine whether should conflict added
                vehicle_j_points_distance_between_routes = [np.min((route_j[point, 0] - route_i[:, 0]) ** 2 + (route_j[point, 1] - route_i[:, 1]) ** 2) for point in range(len(route_j))]
                # vehicle_j_conflict_point_index = np.argmin(vehicle_j_points_distance_between_routes)
                vehicle_j_conflict_point_index = np.where(np.array(vehicle_j_points_distance_between_routes) < 1)[0][0]
                vehicle_j_current_position, vehicle_j_conflict_point_position = vehicle_j.position, route_j[vehicle_j_conflict_point_index]
                vehicle_j_to_conflict_point = np.sqrt((vehicle_j_current_position[0] - vehicle_j_conflict_point_position[0]) ** 2 + (vehicle_j_current_position[1] - vehicle_j_conflict_point_position[1]) ** 2)

                scene_name = self.get_scene_name(env)
                if scene_name == 'intersection':
                    reaction_range = 50
                elif scene_name == 'merge' or scene_name == 'highway':
                    reaction_range = 120
                else:
                    raise ValueError('no such scenario')
                if (distance_to_conflict_point < reaction_range and vehicle_j_to_conflict_point < reaction_range) and vehicle_i_distance_to_destination > vehicle_i_conflict_point_distance_to_destination and vehicle_j_distance_to_destination > vehicle_j_conflict_point_distance_to_destination:
                    return True, distance_to_conflict_point
                else:
                    return False, None  # Placeholder logic
            else:
                return False, None
        else:
            return False, None

    def get_lane_xy(self, env, current_lane_index, point_num=1000):
        current_lane_index_list = list(current_lane_index)
        current_lane_index_list[2] = 0  # 将 lane_id 从 None 修改为 0
        current_lane_index = tuple(current_lane_index_list)
        current_lane = env.road.network.get_lane(current_lane_index)
        if isinstance(current_lane, highway_env.road.lane.StraightLane):
            start = current_lane.start
            end = current_lane.end
            x = np.linspace(start[0], end[0], point_num)
            y = np.linspace(start[1], end[1], point_num)
        elif isinstance(current_lane, highway_env.road.lane.CircularLane):
            center = current_lane.center
            radius = current_lane.radius
            start_phase = current_lane.start_phase
            end_phase = current_lane.end_phase
            theta = np.linspace(start_phase, end_phase, point_num)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
        else:
            x, y = None, None
            print("This is another type of Lane.")
        return x, y

    def record_route_xy(self, env, vehicle):
        ref_x_list = []
        ref_y_list = []
        # print(vehicle)
        # print(vehicle.route)
        if not hasattr(vehicle, 'route') or vehicle.route is None:
            return []
        for lane_index in vehicle.route:
            x, y = self.get_lane_xy(env, lane_index)
            ref_x_list = np.hstack((ref_x_list, x))
            ref_y_list = np.hstack((ref_y_list, y))
        ref_list = np.column_stack((ref_x_list, ref_y_list))
        return ref_list

    def send_to_chatgpt(self, env, conflict):
        # Implement your LLM interaction here (similar to LlmAgent_action_module)
        # This method sends the scenario description to LLM and retrieves the suggested action
        proxy_url = "http://127.0.0.1:7890"
        import httpx
        http_client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})

        client = OpenAI(api_key=api_key,  # put your api key here
                        base_url="https://api.openai.com/v1",
                        http_client=http_client)

        conflicting_vehicles_info = []
        conflict = self.detect_conflicts(env)
        print(conflict)

        scene_name = self.get_scene_name(env)
        if scene_name == 'intersection':
            vehicle_list = self.controlled_vehicles
        elif scene_name == 'merge' or scene_name == 'highway':
            vehicle_list = env.road.vehicles
        else:
            raise ValueError('no such scenario')

        for vehicle_i in range(len(vehicle_list)):
            if vehicle_list[vehicle_i] in conflict:
                for vehicle_j in range(vehicle_i):
                    if vehicle_list[vehicle_j] in conflict[vehicle_list[vehicle_i]] and vehicle_list[vehicle_i] in conflict[vehicle_list[vehicle_j]]:
                        vehicle_i_info = vehicle_list[vehicle_i]
                        vehicle_j_info = vehicle_list[vehicle_j]
                        conflicting_vehicles_info.append(
                            {'vehicle_i': vehicle_i_info, 'vehicle_i speed': vehicle_i_info.speed,
                             'vehicle_i distance to conflict': conflict[vehicle_i_info][vehicle_j_info],
                             'vehicle_j': vehicle_j_info, 'vehicle_j speed': vehicle_j_info.speed,
                             'vehicle_j distance to conflict': conflict[vehicle_j_info][vehicle_i_info]})

        conflicting_info = "\n".join(
            [f"- Vehicle i: {conflict_group['vehicle_i']}, Vehicle i Speed: {conflict_group['vehicle_i speed']} m/s, "
             f"Vehicle i distance to conflict point: {conflict_group['vehicle_i distance to conflict']} meters, "
             f"- Vehicle j: {conflict_group['vehicle_j']}, Vehicle j Speed: {conflict_group['vehicle_j speed']} m/s, "
             f"Vehicle j distance to conflict point: {conflict_group['vehicle_j distance to conflict']} meters, "
             for conflict_group in conflicting_vehicles_info]
        )

        # print(conflicting_info)
        prompt = (
            "You are simulating as a traffic police officer overseeing traffic conflicts. Here are the current conflict scenarios:\n"
            f"{conflicting_info}\n\n"
            "For each conflict, decide which vehicle should pass first and which should pass later to ensure safety:\n"
            "Output your answer in the following format:\n"
            "```\n"
            "Final Answer: \n"
            "    \"decisions\": [\n"
            "        {\"first_vehicle\": \"<vehicle_i>\", \"second_vehicle\": \"<vehicle_j>\"},\n"
            "        {\"first_vehicle\": \"<vehicle_i>\", \"second_vehicle\": \"<vehicle_j>\"},\n"
            "        ...\n"
            "    ]\n"
            "```\n"
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}, ])

        llm_response = completion.choices[0].message
        negotiation_content = llm_response.content
        # print(prompt)
        # print(f"LLM decision: {negotiation_content}")
        return negotiation_content, conflicting_vehicles_info

    def retrun_sce(self):
        return self.sce

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
