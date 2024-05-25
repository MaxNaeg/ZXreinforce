# ZXreinforce
This project contains the code used to produce the results in the two publications: "Optimizing ZX-Diagrams with Deep Reinforcement Learning" (arxiv.org/abs/2311.18588) and "Tackling Decision Processes with Non-Cumulative Objectives using Reinforcement Learning" (https://arxiv.org/abs/2405.13609).

* Main code of the algorithm is in zxreinforce

For results relating to "Optimizing ZX-Diagrams with Deep Reinforcement Learning" see:
* A script showing how to train an agent is at experiments/train_rl_agent/runner_final.py.
* The agent's training progress can be monitored with experiments/evaluation_rl_agent/evaluation_training_logger.ipynb
* An example notebook showing how to simplify a diagram with the trained agent is at experiments/evaluation_rl_agent/simplify_example_traj.ipynb.
* Scripts to compare the performance of the RL agent to a greedy strategy and simulated annealing are in experiments/evaluation_performance.
* The evaluation of the Copy action is done in experiments/eval_copy_action.
* The evaluation of the action dependence on the local environment is done in experiments/prob_vs_layer.
* The network weights of the agents trained for the ablation studies can be found in saved_agents.

For results relating to "Tackling Decision Processes with Non-Cumulative Objectives using Reinforcement Learning" see:
* A script showing how to train an agent is at ncmdp_experiments/runner_max_20s_bench.py.
* Trained agent weights and training progress logs are in ncmdp_experiments/runs.
* The results are analyzed in ncmdp_experiments/evalt_result.ipynb.

To run this code install the requirements.txt or use the docker image as described in https://github.com/alexandrupaler/zxreinforce_small.

## Citation
``` bib
@article{nagele_optimizing_2023,
  title={Optimizing {ZX}-Diagrams with Deep Reinforcement Learning},
  author={N\"agele, Maximilian and Marquardt, Florian},
  url = {https://arxiv.org/abs/2311.18588},
  doi = {10.48550/arXiv.2311.18588}
  journal={arXiv},
  year = {2023},
}
```
