# ZXreinforce
This project contains the code used to produce the results in "Optimizing ZX-Diagrams with Deep Reinforcement Learning".
* Main code of the algorithm is in zxreinforce
* A script showing how to train an agent is at experiments/train_rl_agent/runner_final.py
* The agent's training progress can be monitored with experiments/evaluation_rl_agent/evaluation_training_logger.ipynb
* An example notebook showing how to simplify a diagram with the trained agent is at experiments/evaluation_rl_agent/simplify_example_traj.ipynb
* Scripts to compare the performance of the RL agent to a greedy strategy and simulated annealing are in experiments/evaluation_performance
* The evaluation of the Copy action is done in experiments/eval_copy_action
* The evaluation of the action dependence on the local environment is done in experiments/prob_vs_layer
* The network weights of the agents trained for the ablation studies can be found in saved_agents