
# This scripts samples 1000 actions of each type by optimizing ZX-diagrams with a save RL agent.
# It saves the action indices, observations the actions where sampled in, the action rewards and action masks.
# The results are further evaluated in dist_vs_layer.py
import sys
sys.path.append("../../")
sys.path.append("../../../")


import pickle 
from pathlib import Path

import numpy as np
import functools

import tensorflow as tf
import tensorflow_gnn as tfgnn
import keras
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format


import zxreinforce.own_constants as oc
from zxreinforce.ZX_env import  ZXCalculus
from zxreinforce.rl_schemas import OBSERVATION_SCHEMA_ZX_final
from zxreinforce.VecAsyncEnvironment import AsyncVectorEnv
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.VecAsyncEnvironment import AsyncVectorEnv
from zxreinforce.PPO_Agent_mult_GPU import PPOAgentPara
from zxreinforce.RL_Models import build_gnn_actor_model, build_gnn_critic_model
from zxreinforce.batch_utils import batch_mask_combined, batch_obs_combined_traj
from zxreinforce.action_conversion_utils import get_action_type_idx



seed=0


# Load agent
graph_schema = text_format.Merge(OBSERVATION_SCHEMA_ZX_final, schema_pb2.GraphSchema())
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])

with strategy.scope():
    actor_model = build_gnn_actor_model(graph_tensor_spec=graph_tensor_spec)
    critic_model = build_gnn_critic_model(graph_tensor_spec=graph_tensor_spec)
    optimizer = keras.optimizers.Adam()


load_idx = 400
load_path = Path("../../saved_agents/all_1/saved_agent")
ppo_agent = PPOAgentPara.load_from_folder(load_path, actor_model, critic_model, optimizer, strategy, load_idx)


# Load env
n_envs = 90

max_steps = 200
add_reward_per_step = 0


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

env_gen = []
for idx in range(n_envs):
    resetterr = Resetter_ZERO_PI_PIHALF_ARB_hada(n_in_min,
                                n_in_max,
                                min_spiders,
                                max_spiders,
                                pi_fac,
                                pi_half_fac,
                                arb_fac,
                                p_hada,
                                min_mean_neighbours,
                                max_mean_neighbours,
                                np.random.default_rng(seed+idx))
    def get_env(reseter):
        return ZXCalculus(max_steps=max_steps, 
                            add_reward_per_step=add_reward_per_step,
                            resetter=reseter,
                            check_consistencty=False,
                            count_down_from=20,
                            dont_allow_stop=False)
    # Fuctools.partial makes sure that the right resetter is used
    env_gen.append(functools.partial(get_env, resetterr))
env = AsyncVectorEnv(env_gen)



# Sample n_samples actions for each action type
n_samples=1000

# Set up lists 
obs_lists = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]
mask_lists = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]

reward_lists = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]
act_idcs_lists = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]

observation, mask  = env.reset()

# Sample actions
while np.any([len(obs_li)<n_samples for obs_li in obs_lists]):
    # Step env
    observation_batched = batch_obs_combined_traj(observation)
    mask_batched = batch_mask_combined(mask)

    action, _  = ppo_agent.sample_action_logits_trajectory(observation_batched, mask_batched)
    next_observation, next_mask, reward, _ = env.step(action.numpy())
    
    # Append actions that are still needed
    act_type_idcs = np.array([get_action_type_idx(obs[-3], obs[-2], act) for obs, act in zip(observation, action)])
    for act_index in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS):
        if len(obs_lists[act_index]) < n_samples:
            idcs_occur = np.where(act_type_idcs == act_index)[0]
            obs_lists[act_index] +=  [observation[i] for i in idcs_occur]
            mask_lists[act_index] +=  [mask[i] for i in idcs_occur]
            reward_lists[act_index] +=  [reward[i] for i in idcs_occur]
            act_idcs_lists[act_index] +=  [action.numpy()[i] for i in idcs_occur]
            # Save list when 1000 actions are sampled
            if len(obs_lists[act_index]) >= n_samples:
                with open(f"sampled/obs_list_{act_index}_{n_samples}", 'wb') as f:
                    pickle.dump(obs_lists[act_index], f)
                with open(f"sampled/reward_list_{act_index}_{n_samples}", 'wb') as f:
                    pickle.dump(reward_lists[act_index], f)
                with open(f"sampled/act_idcs_list{act_index}_{n_samples}", 'wb') as f:
                    pickle.dump(act_idcs_lists[act_index], f)
                with open(f"sampled/mask_list{act_index}_{n_samples}", 'wb') as f:
                    pickle.dump(mask_lists[act_index], f)


    # Update obs/mask
    observation = next_observation
    mask = next_mask
        

