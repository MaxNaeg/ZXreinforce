# Evaluate the greedy agent on a list of initial observations
import sys
sys.path.append("../../")
sys.path.append("../../../")


import pickle 

from pathlib import Path
import numpy as np
from time import time
import pyzx as zx
from pyzx.hsimplify import from_hypergraph_form


from zxreinforce.compare_agents import GreedyCleverAgent, AfterPYZXAgent
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.VecAsyncEnvironment import VecZXCalculus
from zxreinforce.ZX_env_max import ZXCalculus, apply_auto_actions, check_consistent_diagram, get_neighbours
from zxreinforce.pyzx_utils import obs_to_pzx, pyzx_to_obs
from zxreinforce.own_constants import ARBITRARY, HADAMARD

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

size = 10
# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path(f"../../saved_observations/initial_1000_obs_list_{size}_{int(size*1.5)}.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)

# Load greedy agent
greedy_agent = GreedyCleverAgent()
afterpzxagent = AfterPYZXAgent()



reward_list_pyzx = []
diff_arbitrary_pyzx = []


failed = []

add_save= f"pyzx_{size}"
print(add_save, flush=True)
start_time = time()
print(start_time, flush=True)

for n, inital_obs in enumerate(initial_obs_list):
    # print(n, flush=True)
    try:
        inital_obs = inital_obs[0]

        init_spiders = len(inital_obs[0])
        init_arbitrary = np.sum([np.all(angle == ARBITRARY) for angle in inital_obs[1]])

        graph = obs_to_pzx(inital_obs)
        from_hypergraph_form(graph)
        zx.full_reduce(graph, quiet=True)
        obs = pyzx_to_obs(graph)

        final_spiders_pyzx = len(obs[0])
        final_arbitrary_pyzx = np.sum([np.all(angle == ARBITRARY) for angle in obs[1]])

        reward_list_pyzx.append([init_spiders - final_spiders_pyzx])
        diff_arbitrary_pyzx.append([init_arbitrary - final_arbitrary_pyzx])

    except ValueError as e:
        failed.append(n)
        print("Failed", n, e, flush=True)

end_time = time()
print(end_time, flush=True)
print(end_time-start_time, flush=True)       

with open('results_pyzx/reward_list_'+add_save+'.pkl', 'wb') as f:
    pickle.dump(reward_list_pyzx, f)
with open('results_pyzx/diff_arbitrary_'+add_save+'.pkl', 'wb') as f:
    pickle.dump(diff_arbitrary_pyzx, f)
