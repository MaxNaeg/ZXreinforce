# This script evaluates the performance of a trained agent on a list of initial observations.
import sys
sys.path.append("../../")
sys.path.append("../../../")


import pickle 
import keras
from time import time

from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format


from zxreinforce.rl_schemas import OBSERVATION_SCHEMA_ZX_MAX
from zxreinforce.VecAsyncEnvironment import VecZXCalculus
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.PPO_Agent_mult_GPU import PPOAgentPara
from zxreinforce.RL_Models_Max import build_gnn_actor_model, build_gnn_critic_model
from zxreinforce.batch_utils import batch_mask_combined, batch_obs_combined_traj
from zxreinforce.own_constants import ARBITRARY


graph_schema = text_format.Merge(OBSERVATION_SCHEMA_ZX_MAX, schema_pb2.GraphSchema())
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])


with strategy.scope():
    actor_model = build_gnn_actor_model(graph_tensor_spec=graph_tensor_spec)
    critic_model = build_gnn_critic_model(graph_tensor_spec=graph_tensor_spec)
    optimizer = keras.optimizers.Adam()


max_steps = 200
n_envs = 1
add_reward_per_step = 0

seed=103

# Params for initial env observations, NOT USED
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

size = 100
# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path(f"../../saved_observations/initial_1000_obs_list_{size}_{int(size*1.5)}.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)
# Load agent

to_add_list = [
            #    "no_clip_anneal_seed=0_20240412-204732",
            #    "no_counter_seed=0_20240412-204606",
            #    "no_entropy_anneal_seed=0_20240412-204700",
            #    "no_entropy_seed=0_20240412-204634",
            #    "no_kl_limit_seed=0_20240412-204905",
            #    "no_stop_seed=0_20240422-120103",
            #    "normal_seed=0_20240412-204251",
               "normal_seed=1_20240412-204341"
               ]

for add_save in to_add_list:
    
    load_path_agent = Path(f"../../saved_agents/{add_save}/saved_agent")

    if "no_counter" in add_save:
        count_down_from=0
    else:
        count_down_from=20
    
    dont_allow_stop=True

    load_idx = 400
    #ppo_agent = PPOAgentPara.load_from_folder(load_path_agent, actor_model, critic_model, optimizer, strategy, load_idx)
    ppo_agent = PPOAgentPara.load_from_folder(load_path_agent, actor_model, critic_model, optimizer, strategy, load_idx)



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
                        count_down_from=count_down_from,
                        check_consistencty=False,
                        dont_allow_stop=dont_allow_stop,
                        extra_state_info=False,
                        adapted_reward=False,)

    reward_list_agent = []
    diff_arbitrary_agent = []

    start_time = time()
    print(f"starting {add_save}", flush=True)
    print(start_time, flush=True)
    for n, inital_obs in enumerate(initial_obs_list):
        print(n, flush=True)

        init_arbitrary = np.sum([np.all(angle == ARBITRARY) for angle in inital_obs[0][1]])

        env.env_list[0].load_observation(*inital_obs)

        observation, mask  = env.env_list[0].get_observation_mask()
        observation = [observation]
        mask = [mask]
        done = np.zeros(n_envs, dtype=np.int32)
        reward_agent_list = []
        while done[0] !=1:

            observation_batched = batch_obs_combined_traj(observation)
            mask_batched = batch_mask_combined(mask)
            # Get the logprobs, action
            action, logprobability_t  = ppo_agent.sample_action_logits_trajectory(observation_batched, mask_batched)
            # Take one step in the environment
            next_observation, next_mask, reward, next_done = env.step(action.numpy())
            # Update the observation, mask and done
            final_arbitrary_pyzx = np.sum([np.all(angle == ARBITRARY) for angle in observation[0][1]])

            observation = next_observation
            done = next_done
            mask = next_mask

            reward_agent_list.append(reward)
        
        

        reward_list_agent.append(reward_agent_list)
        diff_arbitrary_agent.append([init_arbitrary - final_arbitrary_pyzx])

    end_time = time()
    print(end_time, flush=True)
    print(end_time-start_time, flush=True)

    add_save_final = add_save + f"_2_{size}_"
    with open('results_rl/reward_list_agent'+add_save_final+'.pkl', 'wb') as f:
        pickle.dump(reward_list_agent, f)
    with open('results_rl/diff_arbitrary_agent'+add_save_final+'.pkl', 'wb') as f:
        pickle.dump(diff_arbitrary_agent, f)