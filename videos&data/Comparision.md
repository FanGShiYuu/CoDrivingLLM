In addition to the collaborative methods compared in the main manuscript, we further evaluated the success rate differences between the proposed CoDrivingLLM and single-vehicle decision-making approaches. Specifically, we selected the rule-based Projected Intelligent Driver Model (PIDM), the optimization-based Non-Cooperative Game, the reinforcement learning-based PPO, and TeLL, which integrates reinforcement learning with large models, as representative algorithms for single-vehicle decision-making.

|   | PIDM  | Non-Cooperative game  | PPO  | TeLL  | CoDrivingLLM  |
|:------:|:------:|:------:|:------:|:------:|:------:|
| Success rate | 25% | 50% | 11% | 52% | 90%* |

According to the results in the table, the proposed CoDrivingLLM achieves the best performance, while the Non-Cooperative Game and TeLL follow, reaching around 50%. This is primarily because the scenarios considered in this study are more complex than those in the original models. Additionally, PIDM exhibits overly conservative behavior, leading to significantly lower efficiency.
