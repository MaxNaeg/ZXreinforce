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


from zxreinforce.rl_schemas import OBSERVATION_SCHEMA_ZX_final
from zxreinforce.VecAsyncEnvironment import VecZXCalculus
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.PPO_Agent_mult_GPU import PPOAgentPara
from zxreinforce.RL_Models import build_gnn_actor_model, build_gnn_critic_model
from zxreinforce.batch_utils import batch_mask_combined, batch_obs_combined_traj

graph_schema = text_format.Merge(OBSERVATION_SCHEMA_ZX_final, schema_pb2.GraphSchema())
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
n_average = 1000


# Load list of observations to evaluate the agent on
load_path_initial_obs_list= Path("../../saved_observations/initial_1000_obs_list_100_150.pkl")
with open(str(load_path_initial_obs_list), 'rb') as f:
    initial_obs_list = pickle.load(f)

# Load agent
load_path_agent = Path("../../saved_agents/all_1/saved_agent")
add_save="add save name here"   
load_idx = 400
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
                    check_consistencty=False,
                    dont_allow_stop=True)

reward_list_agent = []

start_time = time()
print(start_time, flush=True)
for n, inital_obs in enumerate(initial_obs_list):

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
        observation = next_observation
        done = next_done
        mask = next_mask

        reward_agent_list.append(reward)

    reward_list_agent.append(reward_agent_list)

end_time = time()
print(end_time, flush=True)
print(end_time-start_time, flush=True)


with open('results/reward_list_agent'+add_save+'.pkl', 'wb') as f:
    pickle.dump(reward_list_agent, f)