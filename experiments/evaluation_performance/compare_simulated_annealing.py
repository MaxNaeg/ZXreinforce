#This script evaluates the performance of simulated annealing on a list of initial observations.
import sys
sys.path.append("../../")
sys.path.append("../../../")


import pickle 

from pathlib import Path
import numpy as np
from time import time


from zxreinforce.compare_agents import AnnealingAgent
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.VecAsyncEnvironment import VecZXCalculus
from zxreinforce.own_constants import ARBITRARY, HADAMARD


add_reward_per_step = 0
seed=0

# Params for initial env observations, not actually used
n_envs = 1
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
n_average = 1000

# Annealing paras
temp=0.5
max_steps=20000
exp_fac=0.0001

size = 10
# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path(f"../../saved_observations/initial_1000_obs_list_{size}_{int(size*1.5)}.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)

# Add to saved result
add_save= f"{size}_{temp}_{max_steps}_{exp_fac}"


# Load greedy agent
anneal = AnnealingAgent()


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
                    dont_allow_stop=True,
                    adapted_reward=False)

reward_list_anneal = []
diff_arbitrary_anneal = []

print(f"Sim ann {add_save}")
start_time = time()
print(start_time, flush=True)

for n, inital_obs in enumerate(initial_obs_list):
    init_arbitrary = np.sum([np.all(angle == ARBITRARY) for angle in inital_obs[0][1]])
    # print(n, flush=True)
    env.env_list[0].load_observation(*inital_obs)
    obs_list, _, rew_list_anneal, _ = anneal.optimize_env(env, temp, max_steps, anneal_type="exponential", 
                                            exp_factor=exp_fac, 
                                            seed=seed, allow_stop_action=False, reset=False, 
                                            start_unm_neg_rew=True)
    
    obs = obs_list[-1]
    final_arbitrary_pyzx_greedy = np.sum([np.all(angle == ARBITRARY) for angle in obs[1]])
    diff_arbitrary_anneal.append([init_arbitrary - final_arbitrary_pyzx_greedy])

    reward_list_anneal.append(rew_list_anneal)


end_time = time()
print(end_time, flush=True)
print(end_time-start_time, flush=True)       


with open(f'results_sim_ann/reward_list_sim_ann'+add_save+'.pkl', 'wb') as f:
    pickle.dump(reward_list_anneal, f)

with open(f'results_sim_ann/diff_arbitrary_sim_ann'+add_save+'.pkl', 'wb') as f:
    pickle.dump(diff_arbitrary_anneal, f)



