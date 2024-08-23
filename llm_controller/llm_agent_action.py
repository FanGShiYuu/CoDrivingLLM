from .prompt_llm import *
from .Scenario_description import Scenario
import json
from openai import OpenAI
import numpy as np
import gym
import re

api_key = "your key here"

class LlmAgent_action_module():
    def __init__(self, env):
        # self.env = env
        # self.action_space = self.env.action_space
        # self.observation_space = self.env.observation_space
        self.sce = Scenario(env.road, vehicleCount=10)
        self.frame = 0
        # self.obs = self.env.reset()
        self.done = False
        self.toolModels = [
            getAvailableActions(),
            getAvailableLanes(self.sce),
            getLaneInvolvedCar(self.sce),
            isChangeLaneConflictWithCar(self.sce),
            isAccelerationConflictWithCar(self.sce),
            isKeepSpeedConflictWithCar(self.sce),
            isDecelerationSafe(self.sce),
        ]
        self.pre_prompt = PRE_DEF_PROMPT()
        self.get_actions(env)


    def llm_controller_run(self, env, negotiation_prompt, conflicting_info, controlled_vehicles, memory):
        llm_actions = []
        for i, ego_veh in enumerate(controlled_vehicles):
            scene_name = self.get_scene_name(env)
            if scene_name == 'intersection':
                speed_limit = 5
            else:
                speed_limit = 20
            ego_veh.speed = speed_limit if ego_veh.speed > speed_limit else ego_veh.speed
            negotiation_results = self.transfer_negotiation_prompts_to_results(ego_veh, negotiation_prompt)
            prompt_info = self.prompt_engineer(ego_veh, env.road, env, negotiation_results, conflicting_info)  # prompt engineer
            # print("prompt_info:", prompt_info)
            llm_action = self.send_to_chatgpt(ego_veh, prompt_info, negotiation_results, memory)
            # self.memory_update(memory, prompt_info, llm_action)  # active this line to restore new memory during interaction
            llm_actions.append(llm_action)
            print("llm_action:", llm_action, ego_veh, 'speed now:', ego_veh.speed)
        return llm_actions

    def get_scene_name(self, env):
        scene_name = env.spec.id
        match = re.search(r'(merge|intersection|highway)', scene_name)
        simplified_scene_name = match.group(0) if match else 'unknown'
        return simplified_scene_name

    def get_actions(self, env):
        scene_name = self.get_scene_name(env)
        if scene_name == 'highway':
            self.ACTIONS_ALL = {
                0: 'LANE_LEFT',
                1: 'IDLE',
                2: 'LANE_RIGHT',
                3: 'FASTER',
                4: 'SLOWER'
            }
            self.is_intersection = False
        elif scene_name == 'merge' or scene_name == 'intersection':
            self.ACTIONS_ALL = {
                1: 'IDLE',
                3: 'FASTER',
                4: 'SLOWER',
            }
            self.is_intersection = True
        else:
            self.ACTIONS_ALL = None
            self.is_intersection = None

    def retrun_sce(self):
        return self.sce

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_action_id_from_name(self, action_name, actions_all):
        """
        Get the action ID corresponding to the given action name.

        Parameters:
        action_name (str): The name of the action (e.g., 'LANE_LEFT').
        actions_all (dict): Dictionary mapping IDs to action names.

        Returns:
        int: The ID corresponding to the action name, or -1 if not found.
        """
        for id, name in actions_all.items():
            if name == action_name or name == action_name.upper():
                return id
        return -1  # Return -1 or any suitable value to indicate 'not found'

    def transfer_negotiation_prompts_to_results(self, ego_veh, negotiation_prompt):
        vehicle_conflicts = self.extract_vehicle_conflicts(negotiation_prompt, str(ego_veh).split(":")[0].strip())
        negotiation_results = ""
        for conflict in vehicle_conflicts:
            first_vehicle, second_vehicle, order = conflict
            if first_vehicle == str(ego_veh).split(":")[0].strip():
                negotiation_results += f"- You have conflict with {second_vehicle}. It is suggested that you should {'passes first' if order == 'first' else 'passes second'}.\n"
            else:
                negotiation_results += f"- You have conflict with {first_vehicle}. It is suggested that you should {'passes first' if order == 'first' else 'passes second'}.\n"
        return negotiation_results

    def relative_memory(self, memory, prompt_info):
        experience = ""
        extract_prompt = prompt_info.strip().split('\n')
        query_scenario = '\n'.join(extract_prompt[-2:])  # only save the last two line of prompt_info which store the most dangerous conflict as memory page_content
        past_decisions = memory.retrieveMemory(query_scenario, top_k=2)
        for past_decision in past_decisions:
            experience += f"- Last time {past_decision['negotiation_result']}, you choose to {past_decision['final_action']}, it is {past_decision['comments']}\n"
        experience += f"Above messages are some examples of how you make a decision in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario."
        return experience

    def memory_update(self, memory, prompt_info, llm_action):
        human_question = str(None)
        negotiation = 'Ego should pass second'  # memory store the dangerous info (which negotiation is ego to yield) for ego vehicle
        action = str(llm_action)
        comments = 'Cause more danger' if llm_action == 'FASTER' else 'Safe now but should pay attention to avoid future danger'
        memory.addMemory(prompt_info, human_question, negotiation, action, comments)


    def send_to_chatgpt(self, ego_veh, current_scenario, negotiation_results, memory):
        proxy_url = "http://127.0.0.1:7890"
        import httpx
        http_client = httpx.Client(proxies={"http://": proxy_url, "https://": proxy_url})

        client = OpenAI(api_key=api_key,  # put your api key here
                        base_url="https://api.openai.com/v1",
                        http_client=http_client)

        if self.is_intersection:
            message_prefix = self.pre_prompt.SYSTEM_MESSAGE_PREFIX_intersection
            traffic_rules = self.pre_prompt.get_traffic_rules(self.is_intersection)
            decision_cautions = self.pre_prompt.get_decision_cautions(self.is_intersection)
        else:
            message_prefix = self.pre_prompt.SYSTEM_MESSAGE_PREFIX
            traffic_rules = self.pre_prompt.get_traffic_rules()
            decision_cautions = self.pre_prompt.get_decision_cautions()
        # action_name = ACTIONS_ALL.get(action_id, "Unknown Action")
        # action_description = ACTIONS_DESCRIPTION.get(action_id, "No description available")
        # past_memory = self.relative_memory(memory, current_scenario)  # with this line to active memory retrivel
        past_memory = ''

        prompt = (f"{message_prefix}"
                  f"You, the 'ego' car, are now driving. You have already driven for some seconds.\n"
                  "Here is the current scenario:\n"
                  f"{current_scenario}\n\n"
                  "There are several rules you need to follow when you drive:\n"
                  f"{traffic_rules}\n\n"
                  "Here are your attention points:\n"
                  f"{decision_cautions}\n\n"
                  "Here is your action when scenarios are similar to the current scenario in the past, you should learn from past memory try not to take the cation that cause more danger:\n"
                  f"{past_memory}\n\n"
                  "Based on the planning trajectory, you have the following conflicts with other vehicles.\n"
                  "Here are the conflicts and the suggested passing orders (when you are suggested to passes second, better to slow down): \n"
                  f"{negotiation_results}\n\n"
                  "Once you make a final decision, output it in the following format:\n"
                  "```\n"
                  "Final Answer: \n"
                  "    \"decision\": {\"<ego car's decision, ONE of the available actions (decision have to be one of the following action!!!:  LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER)>\"},\n"
                  "```\n")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # "gpt-3.5-turbo-16k-0613" # "gpt-3.5-turbo-1106"(cheaper)
            messages=[{"role": "system", "content": prompt},])

        llm_response = completion.choices[0].message
        decision_content = llm_response.content
        llm_suggested_action = self.extract_decision(decision_content)
        print(f"llm action: {llm_suggested_action}")

        llm_action_id = self.get_action_id_from_name(llm_suggested_action, self.ACTIONS_ALL)
        llm_action = np.array([llm_action_id])
        return llm_action

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs  # Make sure to return the observation

    def get_available_actions(self):
        """Get the list of available actions from the underlying Highway environment."""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        else:
            raise NotImplementedError(
                "The method get_available_actions is not implemented in the underlying environment.")

    def extract_decision(self, response_content):
        try:
            # print(response_content)
            start = response_content.find('"decision": {') + len('"decision": {')
            end = response_content.find('}', start)
            decision = response_content[start:end].strip('"')
            if self.is_intersection:
                if "FASTER" in decision:
                    decision = "FASTER"
                elif "SLOWER" in decision:
                    decision = "SLOWER"
                elif "IDLE" in decision:
                    decision = "IDLE"
            else:
                if "LANE_LEFT" in decision:
                    decision = "LANE_LEFT"
                elif "LANE_RIGHT" in decision:
                    decision = "LANE_RIGHT"
                elif "FASTER" in decision:
                    decision = "FASTER"
                elif "SLOWER" in decision:
                    decision = "SLOWER"
                elif "IDLE" in decision:
                    decision = "IDLE"
            return decision
        except Exception as e:
            print(f"Error in extracting decision: {e}")
            return None

    def prompt_engineer(self,  ego_veh, road, env, negotiation_results, conflicting_info):
        # self.sce.updateVehicles(obs, frame, i)
        # Observation translation
        msg0 = available_action(self.toolModels, ego_veh, road, env)
        availabel_lane, msg1 = get_available_lanes(self.toolModels, ego_veh, road, env)
        msg2, lane_cars_id = get_involved_cars(self.toolModels, ego_veh, road, env, availabel_lane)
        #lane_cars_id -- {'lane_0': {'leadingCar': None, 'rearingCar': IDMVehicle #224: [173.94198546   0.        ]}}
        #availabel_lane -- {'currentLaneID': 'lane_0', 'leftLane': '', 'rightLane': ''}

        #msg1_info = next(iter(msg1.values()))
        # lanes_info = extract_lanes_info(msg1_info) #{'current': 'lane_3', 'left': 'lane_2', 'right': None}

        # lane_car_ids = extract_lane_and_car_ids(lanes_info, msg2) #{'current_lane': {'car_id': 'veh1', 'lane_id': 'lane_3'}, 'left_lane': {'car_id': 'veh3', 'lane_id': 'lane_2'}, 'right_lane': {'car_id': None, 'lane_id': None}}
        if availabel_lane["leftLane"] != "" or availabel_lane["rightLane"] != "":  # 如果需要换道
            safety_assessment = assess_lane_change_safety(self.toolModels, lane_cars_id, availabel_lane, ego_veh) #{'left_lane_change_safe': True, 'right_lane_change_safe': True}
        else:
            safety_assessment = "There is no need to assess lane change safety."
        safety_msg = check_safety_in_current_lane(self.toolModels, lane_cars_id, availabel_lane, ego_veh) #{'acceleration_conflict': 'acceleration may be conflict with `veh1`, which is unacceptable.', 'keep_speed_conflict': 'keep lane with current speed may be conflict with veh1, you need consider decelerate', 'deceleration_conflict': 'deceleration with current speed is safe with veh1'}
        safety_msg2, most_dangerous_info = check_safety_with_conflict_vehicles(ego_veh, negotiation_results, conflicting_info, env)
        prompt_info = format_training_info(msg0, msg1, msg2, availabel_lane, lane_cars_id, safety_assessment, safety_msg, safety_msg2, most_dangerous_info)  # msg0, msg2, availabel_lane, safety_assessment, safety_msg
        return prompt_info

    def extract_vehicle_conflicts(self, prompt: str, vehicle_id: str) -> list:
        pattern = re.compile(r'"first_vehicle": "(MDPVehicle #[0-9]+|IDMVehicle #[0-9]+)", "second_vehicle": "(MDPVehicle #[0-9]+|IDMVehicle #[0-9]+)"')
        conflicts = pattern.findall(prompt)

        vehicle_conflicts = []
        for first, second in conflicts:
            if first == vehicle_id:
                vehicle_conflicts.append((first, second, 'first'))
            elif second == vehicle_id:
                vehicle_conflicts.append((first, second, 'second'))
        return vehicle_conflicts
