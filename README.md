# CoDrivingLLM: Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-making Framework
[Shiyu Fang](https://fangshiyuu.github.io/), [Jiaqi Liu](https://jiaqiliu-aca.netlify.app/), [Peng Hang](https://www.researchgate.net/profile/Peng-Hang-3), [Jian Sun](https://www.researchgate.net/profile/Jian-Sun-56)  
[Department of Traffic Engineering and Key Laboratory of Road and Traffic Engineering, Ministry of Education, Tongji University](https://tops.tongji.edu.cn/)  
[[Project web](https://fangshiyuu.github.io/CoDrivingLLM/)]

## Getting started 🚀
Install the dependent package
```shell
pip install -r requirements.txt
```
Run CoDrivingLLM
```shell
python Run_multi_CAV_LLM.py
```
remember to add your API key first 

## Overview 🔍
### Repo description
In this repository, you can expect to find the following features:

Included:
* Code for CoDrivingLLM (including highway, merge, intersection)
* Video and Raw Data of our experiment 

Not included:
* Code for Comparison Algorithm (including iDFST, Cooperative Game, MADQN)

### Files description
#### Run_multi_CAV_LLM.py 📄:
Main script for debuging and running LLM-based agent at different scenarios

#### llm_controller folder 📂:
Main modules for LLM agent and other tools

* llm_agent_action.py : 
main script for building the LLM agent, which needs to take the prompt and
scenario information as input, request the remote servers from OpenAI,
get the feedback, parse the feedback, and then output the final decision-making for each CAV.

* llm_agent_negotiation_system.py :
script for generating the advisory passing sequence of vehicles in each conflict pair based on the
severity of the conflict for each CAV final decision, features a centralized-distributed coupled architecture.

* memory.py :
stores all the functions required by the memory module, including the acquisition of similar memories and the storage of new memories.

* prompt_llm.py: all prompts used to connect the traffic scenario, CAV and ChatGPT.
Different scenarios may need different prompt, which needs to be revised and updated according
to the scenario's meets.

* scenario_description.py : the main function file for translating the traffic scenarios into natural
languages that LLM can understand.

#### highway_env folder 📂: 
Open source highway-env simulator with different traffic scenarios, 
including single-lane unsignalized intersection, on-ramp scenario, and highway scenario,which are corresponded to
intersection_env.py, merge_env_v1.py, and highway_env.py, respectively.

#### videos&data 📂: 
Video of vehicle operation and raw data of different cooperative driving method in each experiment.

### If you have any questions, feel free to contact us (2111219@tongji.edu.cn) 📧.

#### Citation

Our paper has been pre-printed! If you find our work helpful, please consider citing us using the following reference 😊:

```bibtex
@misc{fang2024interactivelearnablecooperativedriving,
      title={Towards Interactive and Learnable Cooperative Driving Automation: a Large Language Model-Driven Decision-Making Framework}, 
      author={Shiyu Fang and Jiaqi Liu and Mingyu Ding and Yiming Cui and Chen Lv and Peng Hang and Jian Sun},
      year={2024},
      eprint={2409.12812},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.12812}, 
}

