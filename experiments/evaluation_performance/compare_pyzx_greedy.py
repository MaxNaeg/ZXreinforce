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

size = 100
# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path(f"../../saved_observations/initial_1000_obs_list_{size}_{int(size*1.5)}.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)

# Load greedy agent
greedy_agent = GreedyCleverAgent()


reward_list_pyzx_greedy = []
diff_arbitrary_pyzx_greedy = []

failed = []


resetter = Resetter_ZERO_PI_PIHALF_ARB_hada(n_in_min,
                                        n_in_max,
                                        min_spiders,
                                        max_spiders,
                                        pi_fac,
                                        pi_half_fac,
                                        arb_fac,
                                        p_hada,
                                        min_mean_neighbours,
                                        max_mean_neighbours,
                                        np.random.default_rng(seed))


env = VecZXCalculus([resetter],
                    n_envs=n_envs, 
                    max_steps=max_steps, 
                    add_reward_per_step=add_reward_per_step,
                    check_consistencty=False,
                    dont_allow_stop=False,
                    adapted_reward=False)


add_save= f"pyzx_greedy_{size}"
print(add_save, flush=True)

start_time = time()
print(start_time, flush=True)

for n, inital_obs in enumerate(initial_obs_list):
    print(n, flush=True)
    try:
        inital_obs = inital_obs[0]

        init_spiders = len(inital_obs[0])
        init_arbitrary = np.sum([np.all(angle == ARBITRARY) for angle in inital_obs[1]])

        graph = obs_to_pzx(inital_obs)
        from_hypergraph_form(graph)
        zx.full_reduce(graph, quiet=True)
        obs = pyzx_to_obs(graph)


        (colors, angles, selected_node, source, target, 
        selected_edges, n_nodes, n_edges, _) = obs

        colors, angles, source, target = apply_auto_actions(colors, angles, source, target)
        n_nodes = len(colors)
        n_edges = len(source)
        selected_node = np.zeros(n_nodes)
        selected_edges = np.zeros(n_edges)

        obs_new = (colors, angles, selected_node, source, target, 
        selected_edges, n_nodes, n_edges, None)

    
        env.env_list[0].load_observation(obs_new)

        obs_list, _, reward_list, _, _ = greedy_agent.step_trajectory(env, reset=False)
        obs = obs_list[-1]
        
        final_spiders_pyzx_greedy = len(obs[0][0])
        final_arbitrary_pyzx_greedy = np.sum([np.all(angle == ARBITRARY) for angle in obs[0][1]])
        
        reward_list_pyzx_greedy.append([init_spiders - final_spiders_pyzx_greedy])
        diff_arbitrary_pyzx_greedy.append([init_arbitrary - final_arbitrary_pyzx_greedy])
    except ValueError as e:
        failed.append(n)
        print("Failed", n, e, flush=True)

end_time = time()
print(end_time, flush=True)
print(end_time-start_time, flush=True)       

with open('results_pyzx/reward_list'+add_save+'.pkl', 'wb') as f:
    pickle.dump(reward_list_pyzx_greedy, f)
with open('results_pyzx/diff_arbitrary'+add_save+'.pkl', 'wb') as f:
    pickle.dump(diff_arbitrary_pyzx_greedy, f)
