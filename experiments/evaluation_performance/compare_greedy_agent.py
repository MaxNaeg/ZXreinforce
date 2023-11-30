# Evaluate the greedy agent on a list of initial observations
import sys
sys.path.append("../../")
sys.path.append("../../../")


import pickle 

from pathlib import Path
import numpy as np
from time import time


from zxreinforce.compare_agents import GreedyCleverAgent
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.VecAsyncEnvironment import VecZXCalculus

max_steps = 200
n_envs = 1
add_reward_per_step = 0

seed=0

# Params for initial env observations
n_in_min=1
n_in_max=3
pi_fac=0.4
pi_half_fac=0.4
arb_fac=0.4
p_hada=0.2
min_mean_neighbours=2
max_mean_neighbours=4
fac = 1
min_spiders=10 * fac
max_spiders=15 * fac


# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path("../../saved_observations/initial_1000_obs_list_100_150.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)

# Load greedy agent
greedy_agent = GreedyCleverAgent()


resetter_list=[Resetter_ZERO_PI_PIHALF_ARB_hada(n_in_min,
                            n_in_max,
                            min_spiders,
                            max_spiders,
                            pi_fac,
                            pi_half_fac,
                            arb_fac,
                            p_hada,
                            min_mean_neighbours,
                            max_mean_neighbours,    
                            np.random.default_rng(seed+idx)) for idx in range(n_envs)]

env = VecZXCalculus(resetter_list,
                    n_envs=n_envs, 
                    max_steps=max_steps, 
                    add_reward_per_step=add_reward_per_step,
                    check_consistencty=False,
                    dont_allow_stop=True)

reward_list_greedy = []

add_save= "add save name here"

start_time = time()
print(start_time, flush=True)

for n, inital_obs in enumerate(initial_obs_list):

    env.env_list[0].load_observation(*inital_obs)

    _, _, reward_list, _, _ = greedy_agent.step_trajectory(env, reset=False)
    reward_list_greedy.append(reward_list)

end_time = time()
print(end_time, flush=True)
print(end_time-start_time, flush=True)       



with open('results/reward_list_greedy'+add_save+'.pkl', 'wb') as f:
    pickle.dump(reward_list_greedy, f)
