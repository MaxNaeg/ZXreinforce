
import sys
sys.path.append("../../")
sys.path.append("../../../")

import os
import argparse
import time
import datetime
import shutil
from pathlib import Path

import functools
import random
import keras

import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from google.protobuf import text_format

import zxreinforce.own_constants as oc
from zxreinforce.ZX_env_max import  ZXCalculus
from zxreinforce.rl_schemas import OBSERVATION_SCHEMA_ZX_MAX
from zxreinforce.VecAsyncEnvironment import VecZXCalculus, AsyncVectorEnv
from zxreinforce.Resetters import Resetter_ZERO_PI_PIHALF_ARB_hada
from zxreinforce.VecAsyncEnvironment import AsyncVectorEnv
from zxreinforce.Buffer import TrajectoryBuffer
from zxreinforce.PPO_Agent_mult_GPU import PPOAgentPara
from zxreinforce.RL_Models_Max import build_gnn_actor_model, build_gnn_critic_model
from zxreinforce.batch_utils import batch_mask_combined, batch_obs_combined_traj
from zxreinforce.action_conversion_utils import get_action_name, get_action_type_idx


parser = argparse.ArgumentParser(description="Runner Parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--seed", default=0, type=int, 
                    help="Seed for env and network initialization")


parser.add_argument("--ent_coeff", default=0., type=float, 
                    help="Entropy coefficient in loss function")
parser.add_argument("--learning_rate", default=3e-4, type=float, 
                    help="Learning rate of ADAM optimizer")
parser.add_argument("--add_reward_per_step", default=0., type=float, 
                    help="Reward per step, should be non-positive")

parser.add_argument("--extra_state_info", default=False, type=str, 
                    help="extra_state_info")

parser.add_argument("--adapted_reward", default=False, type=str, 
                    help="adapted_reward")

args = vars(parser.parse_args())


COPY_Files = True


extra_state_info= ("True" == args["extra_state_info"])
adapted_reward= ("True" == args["adapted_reward"])

seed = args["seed"]



add_dir=f"no_entropy_{seed=}_"



# Directory to save results
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exp_dir =  Path('runs/' + add_dir + current_time)
exp_dir.mkdir(parents=True, exist_ok=True)

# Log file
log_dir = exp_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
summary_writer = tf.summary.create_file_writer(str(log_dir))
print(f"log_dir: {str(log_dir)}", flush=True)

# Save dir
save_dir = exp_dir / "saved_agent"
save_dir.mkdir(parents=True, exist_ok=True)

if COPY_Files:
    shutil.copyfile("runner.py", 
                    str(exp_dir / "runner.py"))
    shutil.copyfile("../../zxreinforce/PPO_Agent_mult_GPU.py", str(exp_dir / "PPO_Agent_mult_GPU.py"))
    shutil.copyfile("../../zxreinforce/ZX_env_max.py", str(exp_dir / "ZX_env_max.py"))
    shutil.copyfile("../../zxreinforce/RL_Models_Max.py", str(exp_dir / "RL_Models_Max.py"))
    shutil.copyfile("../../zxreinforce/own_constants.py", str(exp_dir / "own_constants.py"))


### Hyperparameters of the PPO algorithm-----------------------------------------------------------








# Total timesteps to take in the environment
total_timesteps = 36e6

# Amount of environments executed in different CPU processes
n_envs = 90

# Number of steps sampled per environement in one trajectory before training
max_sample_steps = 1000
# Steps until environement resets
max_steps = 200
# Give countdown to NN if close to finish trajectory
count_down_from = 20

# Absolute value gradients are clipped to
abs_grad_clip = 100
# Maximum number of gradient updates after one trajectoruy sampling period
train_iterations = 10
# Discount factor for returns
gamma=0.99
# Trade-off between bias and variance in return estimation: If big, high variance, low bias
lam=0.9
# Ratio determines how far action prob can diverge during training, if big high divergence possible
clip_ratio=0.2
# Linearly anneal clip range
anneal_clip_range=True
# KL determines how far policy can diverge during training before sampling new trajectories
target_kl = 0.01
# How many individual steps to use in one training update.
# Minibatchsize should be divisible by amount of parallel GPUs used
minibatch_size = 3000
# Coeff of entropy in loss function. If higher leads to more exploration
ent_coeff=args["ent_coeff"]
# Linearly anneal entropy coeff
anneal_entropy=True
# Coeff of value loss in loss function. If higher leads to bigger gradient update steps in critic model
value_coeff=0.5
# Rescale gradients of minibtach such that norm doesnt exceed this
grad_norm_clip=0.5

# Adam optimizer parameters
learning_rate=args["learning_rate"]
adam_epsilon=1e-7
adam_beta_1=0.9
# Linearly decrease learning rate during training (decreased after each set of trajetcory sampling)
anneal_lr=True

# Save models every save_every updates
save_every = 10


# Environment params----------------------------------------------------------------------------------
# Negative reward per step to encourage environment to cancel early
add_reward_per_step = args["add_reward_per_step"]
# Check that enviroament is valid ZX diagram after every step (for debugging)
check_consistencty = False
# allow stop action
dont_allow_stop = False


# Use multiple cpus for environment
multiprocess = True

# Copy pyh=thon files to result folder


# Log histogram of selected actions
log_action_hist = True
# Log reward of completed episodes
log_episode_reward = True




# Params for initial env observations
min_spiders=10
max_spiders=15
n_in_min=1
n_in_max=3
pi_fac=0.4
pi_half_fac=0.4
arb_fac=0.4
p_hada=0.2
min_mean_neighbours=2
max_mean_neighbours=4



# Set seeds, This makes environments deterministic, but not training
# see https://jackd.github.io/posts/deterministic-tf-part-1/.
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.experimental.numpy.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)



# Make sure experiences can be divided into minibatches
assert((n_envs * max_sample_steps) % minibatch_size == 0)



                    


# Strategy for distributed GPU training
strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])

# Schema of our Graph
graph_schema = text_format.Merge(OBSERVATION_SCHEMA_ZX_MAX, schema_pb2.GraphSchema())
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

initial_rng = np.random.default_rng(seed)
if multiprocess:
    env_gen = []
    for idx in range(n_envs):
        seed_env = initial_rng.integers(0, np.iinfo(np.int32).max)
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
                                    np.random.default_rng(seed_env))
        def get_env(reseter):
            return ZXCalculus(max_steps=max_steps, 
                              add_reward_per_step=add_reward_per_step,
                              resetter=reseter,
                              check_consistencty=check_consistencty,
                              count_down_from=count_down_from,
                              dont_allow_stop=dont_allow_stop,
                              extra_state_info=extra_state_info,
                              adapted_reward=adapted_reward)
        # Fuctools.partial makes sure that the right resetter is used
        env_gen.append(functools.partial(get_env, resetterr))
    env = AsyncVectorEnv(env_gen)
else:
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
                                    np.random.default_rng(initial_rng.integers(0, np.iinfo(np.int32).max))) for idx in range(n_envs)]
    # This is a dummy vector env, that executes the individual envs after each other
    env = VecZXCalculus(resetter_list,
                        n_envs=n_envs, 
                        max_steps=max_steps, 
                        add_reward_per_step=add_reward_per_step,
                        check_consistencty=check_consistencty,
                        dont_allow_stop=dont_allow_stop,
                        extra_state_info=extra_state_info,
                        adapted_reward=adapted_reward)

# Buffer for storing trajectories of one sampling stage
buffer = TrajectoryBuffer(n_envs, max_sample_steps)

# The strategy automatically copies our model on many GPUs and takes care of same gradient updates
with strategy.scope():
    actor_model = build_gnn_actor_model(graph_tensor_spec=graph_tensor_spec,
                                        kernel_ini_hidden=tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=seed),
                                        kernel_ini_out=tf.keras.initializers.Orthogonal(gain=0.01, seed=seed))
    critic_model = build_gnn_critic_model(graph_tensor_spec=graph_tensor_spec, 
                                          kernel_ini_hidden=tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=seed),
                                          kernel_ini_out=tf.keras.initializers.Orthogonal(gain=1., seed=seed))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, epsilon=adam_epsilon, beta_1=adam_beta_1)


# The RL Agent
ppo_agent = PPOAgentPara(strategy=strategy,
                     actor_model=actor_model, 
                     critic_model=critic_model, 
                     optimizer=optimizer, 
                     train_iterations=train_iterations, 
                     gamma=gamma, 
                     lam=lam, 
                     clip_ratio=clip_ratio, 
                     target_kl=target_kl, 
                     minibatch_size=minibatch_size, 
                     ent_coeff=ent_coeff, 
                     value_coeff=value_coeff, 
                     grad_norm_clip=grad_norm_clip,
                     normalize_advantages=True,
                     abs_grad_clip=abs_grad_clip,
                     operation_seed=seed)




batch_size = n_envs * max_sample_steps

def train():

    global_step = 0    
    done = np.zeros(n_envs, dtype=np.int32)
    num_updates = int(total_timesteps // batch_size)
    print(f"Number of updates: {num_updates}", flush=True)


    observation, mask  = env.reset()

    sum_reward = 0

    for update in range(1, num_updates + 1):
        start_time = time.time()
        buffer.reset()
        
        # Annealing
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            ppo_agent.optimizer.learning_rate.assign(lrnow)
        else:
            lrnow=learning_rate

        if anneal_entropy:
            frac = 1.0 - (update - 1.0) / num_updates
            ent_now = frac * ent_coeff
            ppo_agent.ent_coeff = ent_now
        else:
            ent_now=ent_coeff

        if anneal_clip_range:
            frac = 1.0 - (update - 1.0) / num_updates
            clip_now = frac * clip_ratio
            ppo_agent.clip_ratio = clip_now
        else:
            clip_now=clip_ratio
            

        # Iterate over the steps of each epoch
        for step in range(0, max_sample_steps):

            global_step += 1 * n_envs

            # Batch graph observ ation to graph tensor and mask to tensor
            observation_batched = batch_obs_combined_traj(observation)
            mask_batched = batch_mask_combined(mask)
            # Get the logprobs, action
            action, logprobability_t  = ppo_agent.sample_action_logits_trajectory(observation_batched, mask_batched)
            # take one step in the environment
            next_observation, next_mask, reward, next_done = env.step(action.numpy())

            # Get the value of critic model
            value_t = ppo_agent.state_value(observation_batched)

            # Store results of time step
            buffer.add_step(observation, action, reward, value_t, logprobability_t, done, mask)

            # Update the observation, mask and done
            observation = next_observation
            done = next_done
            mask = next_mask


        # Next value for bootstrapping
        observation_batched = batch_obs_combined_traj(observation)
        next_value = ppo_agent.state_value(observation_batched)

        buffer.add_value_done(next_value, done)

        print("start train", flush=True)
        kl, pol_grad_loss, v_loss, entropy_loss, n_clipped_ratio, update_steps = ppo_agent.train(*buffer.get_episode())


        # Print mean return and length for each epoch
        mean_reward_update = np.mean(buffer.reward)
        sum_reward += mean_reward_update
        end_time = time.time()

        print(
            f" Epoch: {update}. Return: {mean_reward_update}, Mean Return: {sum_reward / update}. Time for step:{end_time-start_time}", flush=True
        )
        print(f"global_step={global_step}, episodic_return={np.sum(buffer.reward)/n_envs}", flush=True)


        # Log stuff for tensorboard
        with summary_writer.as_default():
            tf.summary.scalar("losses/value_loss", v_loss, step=global_step)
            tf.summary.scalar("losses/pol_grad_loss", pol_grad_loss, step=global_step)
            tf.summary.scalar("losses/entropy_loss", entropy_loss, step=global_step)

            tf.summary.scalar("kl", kl, step=global_step)
            tf.summary.scalar("n_clipped_ratio", n_clipped_ratio, step=global_step)
            tf.summary.scalar("update_steps", update_steps, step=global_step)

            tf.summary.scalar("mean_reward_update", mean_reward_update, step=global_step)
            tf.summary.scalar("mean_reward_all", sum_reward / update, step=global_step)

            tf.summary.scalar("ent_coeff", ent_now, step=global_step)
            tf.summary.scalar("learning_rate", lrnow, step=global_step)
            tf.summary.scalar("clip_range", clip_now, step=global_step)

        
            if log_episode_reward:

                dones = buffer.dones.T
                rewards= buffer.reward.T

                ep_rew_list=[]
                for don, rew in zip(dones, rewards):
                    start_idx = np.argmax(don)
                    don[start_idx] = 0
                    while np.max(don > 0):
                        end_idx = np.argmax(don)
                        ep_rew_list.append(rew[start_idx: end_idx])
                        don[end_idx] = 0
                        start_idx = end_idx
           
                episodes_averaged_over = len(ep_rew_list)
                if episodes_averaged_over != 0:
                    rewards_ep_all = tf.ragged.stack(ep_rew_list)
                    
                    ep_rewards_mean = tf.reduce_mean(tf.reduce_sum(rewards_ep_all, axis=-1))
                    ep_rewards_max = tf.reduce_max(tf.reduce_sum(rewards_ep_all, axis=-1))
                    ep_rewards_min = tf.reduce_min(tf.reduce_sum(rewards_ep_all, axis=-1))
                    ep_rewards_std = tf.math.reduce_std(tf.reduce_sum(rewards_ep_all, axis=-1))

                    ep_mean_length = tf.reduce_mean([len(ep) for ep in rewards_ep_all])

                    ep_rewards_max_any_mean = tf.reduce_mean(tf.reduce_max([tf.constant([0.,]*episodes_averaged_over), tf.reduce_max(
                        tf.math.cumsum(rewards_ep_all, axis=-1), axis=-1)], axis=0))
                else:
                    ep_rewards_mean = tf.constant(np.nan)
                    ep_rewards_max = tf.constant(np.nan)
                    ep_rewards_min = tf.constant(np.nan)
                    ep_rewards_std = tf.constant(np.nan)

                    ep_rewards_max_any_mean = tf.constant(np.nan)

                    ep_mean_length = tf.constant(np.nan)
                
                tf.summary.scalar("ep_rewards_mean", ep_rewards_mean, step=global_step)
                tf.summary.scalar("ep_rewards_max", ep_rewards_max, step=global_step)
                tf.summary.scalar("ep_rewards_min", ep_rewards_min, step=global_step)
                tf.summary.scalar("ep_rewards_std", ep_rewards_std, step=global_step)
                tf.summary.scalar("ep_mean_length", ep_mean_length, step=global_step)
                tf.summary.scalar("episodes_averaged", episodes_averaged_over, step=global_step)

                tf.summary.scalar("ep_rewards_max_any_mean", ep_rewards_max_any_mean, step=global_step)

            if log_action_hist:
                act_list = []
                for obs, act in zip(buffer.observation.ravel(), np.ravel(buffer.action)):
                    act_list.append(get_action_type_idx(obs[-3], obs[-2], act))

                act_hist = np.bincount(act_list, minlength=oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS + 1)

                for ac_idx, n_act in enumerate(act_hist):
                    tf.summary.scalar(get_action_name(ac_idx), n_act, step=global_step)

        # Save models
        if update % save_every == 0:
            ppo_agent.save(save_dir, index=update)
train()



