import json
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pathlib import Path


from tensorflow_gnn import GraphTensor
from keras import Model
from keras.optimizers import Optimizer

from . import own_constants as oc
from .batch_utils import batch_obs_combined_train, batch_mask_combined



class PPOAgentPara():
    """Proximal Policy Optimization agent thet can handle graph neural 
    networks as policy and value function."""
    def __init__(self, strategy:tf.distribute.MirroredStrategy, actor_model:Model, critic_model:Model, 
                 optimizer:Optimizer, train_iterations:int, 
                 gamma:float, lam:float, clip_ratio: float, target_kl:float, 
                 minibatch_size:int, ent_coeff:float, value_coeff:float, grad_norm_clip:float,
                 normalize_advantages:bool=True, operation_seed:int=42, 
                 abs_grad_clip:float=tf.float32.max):
        """strategy: tf.distribute.Strategy used to create the models and optimizer,
        actor_model: keras.Model, policy network,
        critic_model: keras.Model, value function,
        optimizer: keras.Optimizer, optimizer used to train the models,
        train_iterations: int, maximum number of training iterations per epoch,
        gamma: float, discount factor,
        lam: float, lambda parameter for generalized advantage estimate,
        clip_ratio: float, clipping parameter for PPO,
        target_kl: float, maximum kl divergence between old and new policy,
        minibatch_size: int, size of minibatches used for training,
        ent_coeff: float, coefficient for entropy loss,
        value_coeff: float, coefficient for value function loss,
        grad_norm_clip: float, maximum norm of gradients to clip to,
        normalize_advantages: bool, if True normalize advantages,
        operation_seed: int, random seed used for sampling actions,
        abs_grad_clip: float, maximum absolute value of gradients,
        """
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.optimizer = optimizer
        self.train_iterations = train_iterations
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.minibatch_size = minibatch_size
        self.ent_coeff = ent_coeff
        self.value_coeff = value_coeff
        self.grad_norm_clip = grad_norm_clip
        self.normalize_advantages = normalize_advantages
        self.operation_seed = operation_seed
        self.abs_grad_clip = abs_grad_clip
        self.strategy = strategy

    @classmethod
    def load_from_folder(cls, path:Path, actor_model:Model, critic_model:Model, 
                         optimizer:Optimizer, strategy:tf.distribute.MirroredStrategy, index:str=""):
        """path: Path, folder where agent weights are saved,
        actor_model: keras.Model, policy network,
        critic_model: keras.Model, value function,
        optimizer: keras.Optimizer, optimizer used to train the moddels,
        strategy: tf.distribute.Strategy used to create the models and optimizer,
        index: str, index of agent to load,
        returns: PPOAgentPara, loaded agent

        Loads agent from files saved by self.save method. 
        Needs actor_model, critic_model and optimizer of same structure as in original agent 
        (but arbitrary weights)"""

        with open(str(path / f"paras{index}.json"), "r") as infile:
            para_dict = json.load(infile)

        actor_model.load_weights(str(path / f"actor{index}.keras"))
        critic_model.load_weights(str(path / f"critic{index}.keras"))
        
        return cls(strategy, actor_model, critic_model, optimizer, **para_dict)


    def save(self, path:Path, index:str=""):
        """Saves params as json, actor weights, and critic weights"""

        para_dict = self.paras_to_dict()
        with open(str(path / f"paras{index}.json"), "w") as outfile:
            json.dump(para_dict, outfile)

        self.actor_model.save_weights(str(path / f"actor{index}.keras"))
        self.critic_model.save_weights(str(path / f"critic{index}.keras"))


    def paras_to_dict(self):
        return {"train_iterations": self.train_iterations,
                "gamma": self.gamma,
                "lam": self.lam,
                "clip_ratio": self.clip_ratio,
                "target_kl" : self.target_kl,
                "minibatch_size": self.minibatch_size,
                "ent_coeff": self.ent_coeff,
                "value_coeff": self.value_coeff,
                "grad_norm_clip": self.grad_norm_clip,
                "normalize_advantages": self.normalize_advantages,
                "operation_seed": self.operation_seed,
                "abs_grad_clip": self.abs_grad_clip}
    

    def _get_distribution(self, observation:GraphTensor, mask:tf.constant)->tfp.distributions.Categorical:
        """observation: batched GraphTensor,
        mask: batched tf.bool, mask of valid actions,
        returns: tfp.distributions.Categorical, categorical distribution of logits"""
        edge_dims = observation.edge_sets["edges"].sizes
        node_dims = observation.node_sets["spiders"].sizes
        n_batched = len(edge_dims)

        # Get log probabilities of actions for each observation
        logits = self.actor_model(observation)
        # If not batched need to expand dimension of stop action
        if n_batched == 1:
            logits[2] = tf.expand_dims(logits[2], axis=-1)
        stop_actions = tf.expand_dims(logits[2], axis=-1)

        # Split logist odf node and edge actions 
        node_act_split = tf.split(tf.reshape(logits[0], [-1]), 
                                  node_dims*tf.constant(oc.N_NODE_ACTIONS, dtype=tf.int32))
        edge_act_split = tf.split(tf.reshape(logits[1], [-1]), 
                                  edge_dims*tf.constant(oc.N_EDGE_ACTIONS, dtype=tf.int32))

        # This should be replaced with a tensorarray but works and can handle ragged tensors more easily
        reshaped_logits = []
        # Concatenate node, edge and stop actions for each observation
        for i in range(n_batched):
            reshaped_logits.append(tf.concat([node_act_split[i], edge_act_split[i], stop_actions[i]], axis=0))
        logits = tf.ragged.stack(reshaped_logits)

        # Due to bug in tf.map_fn with ragged tensors we need to pad logits with minimum values
        # see https://github.com/keras-team/keras/issues/16354
        logits = logits.to_tensor(default_value=tf.constant(logits.dtype.min, dtype=logits.dtype))

        # Apply mask
        logits = tf.where(mask, logits, logits.dtype.min)
        
        # Build Categorical distibution from logits of all observations
        distr = tfp.distributions.Categorical(logits=logits, validate_args=True, 
                                              allow_nan_stats=False)
        return distr
    
        
    
    
    
    @tf.function(reduce_retracing=True)
    def sample_action_logits_trajectory(self, observation:GraphTensor, mask:tf.constant)->tuple:
        """ observation: batched GraphTensor,
        mask: batched tf.bool, mask of valid actions,
        returns: tuple of (Tensor, Tensor),

        Computes Sampled action and corresponding logit for each batched observation"""

        distr = self._get_distribution(observation, mask)
        # Calculate normalized log probs
        actions, log_probs = distr.experimental_sample_and_log_prob(sample_shape=(1), seed=self.operation_seed)

        return tf.squeeze(actions, axis=[0]), tf.squeeze(log_probs, axis=[0])
    
    @tf.function(reduce_retracing=True)
    def state_value(self, observation:GraphTensor)->tf.constant:
        """Observation: batched GraphTensor
            returns: Tensor, float32 [batchsize]"""
        return self.critic_model(observation)
    
    def train(self, observations:list, actions:tf.constant, rewards:tf.constant, 
              values:tf.constant, logprobabilities:tf.constant, dones:tf.constant, masks:list)->tuple:
        """observations: list of ZX env observations,
        actions: tf.constant, int32 [n_steps, n_envs], actions taken
        rewards: tf.constant, float32 [n_steps, n_envs], rewards received
        values: tf.constant, float32 [n_steps + 1, n_envs], values of states
        logprobabilities: tf.constant, float32 [n_steps, n_envs], log probabilities of actions
        dones: tf.constant, int32 [n_steps + 1, n_envs], 0 if not done, 1 if done
        masks: list of np.ndarrays, bool [n_steps, n_envs], masks of valid actions,
        returns: tuple of (average KL divergence, policy grad loss, value loss, 
                        ratio of clipped actions, training epochs),

        Trains the agent for one training phase.
        """

        # Number of GPUs used for training
        n_parallel = self.strategy.num_replicas_in_sync

        n_steps = len(observations)
        n_envs = len(observations[0])
        batchsize = n_steps * n_envs
        # Caclulate generalized advantage estimate returns
        next_non_terminal = tf.cast(1 - dones, tf.float32)
        deltas = rewards + self.gamma * values[1:] * next_non_terminal[1:] - values[:-1]
        # Advantages are R_t(approximated) - V_t
        # See supplement of https://arxiv.org/pdf/1804.02717.pdf
        advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        lastgaelam =  np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(n_steps)):
            advantages[t] = lastgaelam = deltas[t] + self.gamma * self.lam * next_non_terminal[t+1] * lastgaelam
        # Returns is used to train critic, advantages to train actor.
        # returns = (R_t - V_t) + V_t = R_t
        # Q(s,a) - V(t) + V(t)
        returns = advantages + values[:1]


        # Batch_indices
        b_inds = np.arange(batchsize)
        # Flatten everything
        flat_obs = observations.ravel()
        flat_masks = masks.ravel()
        flat_logprobabilities = tf.reshape(logprobabilities, -1)
        flat_actions = tf.reshape(actions, -1)
        flat_advantages = tf.reshape(advantages, -1)
        flat_returns = tf.reshape(returns, -1)

        # Train agent for an epoch over all minibatches
        for epoch in range(self.train_iterations):
            # print(f"Start epoch {epoch}")
            # Pick random experiences
            np.random.shuffle(b_inds)

            # Initialize arrays to store results of each minibatch
            kl_av = np.zeros(batchsize//self.minibatch_size, dtype=np.float32)
            pol_grad_loss_av = np.zeros(batchsize//self.minibatch_size, dtype=np.float32)
            v_loss_av = np.zeros(batchsize//self.minibatch_size, dtype=np.float32)
            entropy_loss_av = np.zeros(batchsize//self.minibatch_size, dtype=np.float32)
            n_clipped_av = np.zeros(batchsize//self.minibatch_size, dtype=np.int32)

            # Train one step wrt. to only a minibatch of experiences
            for idx, start in enumerate(range(0, batchsize, self.minibatch_size)):
                # print(f"Start epoch {epoch}, minibatch {idx}")
                end = start + self.minibatch_size
                # Minibatchindices
                mb_inds = b_inds[start:end]
                # Split minibatches to distribute on GPUs
                mb_inds_split = np.split(mb_inds, n_parallel)
                split_list = []
                for idcs in mb_inds_split:
                    split_list.append(
                            [batch_obs_combined_train(flat_obs[idcs]),
                            tf.gather(flat_actions, idcs),
                            tf.gather(flat_logprobabilities, idcs),
                            tf.gather(flat_advantages, idcs),
                            tf.gather(flat_returns, idcs),
                            batch_mask_combined(list(flat_masks[idcs])),
                            tf.constant(self.value_coeff, dtype=tf.float32),
                            tf.constant(self.ent_coeff, dtype=tf.float32),
                            tf.constant(self.clip_ratio, dtype=tf.float32),
                            tf.constant(self.grad_norm_clip, dtype=tf.float32),
                            tf.constant(self.abs_grad_clip, dtype=tf.float32),
                            tf.constant(n_parallel, dtype=tf.float32)
                            ]
                            )
    
                def value_fn(value_context):
                    return split_list[value_context.replica_id_in_sync_group]
                
                distributed_values = (
                    self.strategy.experimental_distribute_values_from_function(
                    value_fn))

                # Run training step on all GPUs
                result = self.strategy.run(self._train_step, args=(distributed_values, ))
                # Reduce results to one value per minibatch
                kl, pol_grad_loss, v_loss, entropy_loss, n_clipped = self.strategy.reduce(
                    tf.distribute.ReduceOp.SUM, result, axis=None)

                kl_av[idx] = kl
                pol_grad_loss_av[idx] = pol_grad_loss
                v_loss_av[idx] = v_loss
                entropy_loss_av[idx] = entropy_loss
                n_clipped_av[idx] = n_clipped

            # Stop early if KL of policy diverges too much
            if np.mean(kl_av) > self.target_kl:
                break
        # Return the results of the last training epoch:
        return (np.mean(kl_av), np.mean(pol_grad_loss_av), np.mean(v_loss_av),
                np.mean(entropy_loss_av), np.mean(n_clipped_av) / self.minibatch_size, epoch + 1)

            

    @tf.function(reduce_retracing=True)
    def _train_step(self, split_batch:list)->tuple:  
        """split_batch: list,
        returns: tuple of (kl, pol_grad_loss, v_loss, entropy_loss, n_clipped),

`       Train agent for one step wrt. to a minibatch of experiences.
        """ 
        (observation_minibatched, actions, log_probs, mb_advantages, 
                    returns, mask_minibatched, value_coeff, ent_coeff, 
                    clip_ratio, grad_norm_clip, abs_grad_clip, n_parallel) = split_batch
        
        with tf.GradientTape() as tape:
            # Get distribtion
            # We need to copy paste all code again here to make tf function compatible-----------------------------------
            edge_dims = observation_minibatched.edge_sets["edges"].sizes
            node_dims = observation_minibatched.node_sets["spiders"].sizes
            n_batched = len(edge_dims)

            # Get log probabilities of actions for each observation
            logits = self.actor_model(observation_minibatched)
            # If not batched need to expand dimension of stop action
            if n_batched == 1:
                logits[2] = tf.expand_dims(logits[2], axis=-1)
            stop_actions = tf.expand_dims(logits[2], axis=-1)

            # Split logist odf node and edge actions 
            node_act_split = tf.split(tf.reshape(logits[0], [-1]), 
                                    node_dims*tf.constant(oc.N_NODE_ACTIONS, dtype=tf.int32))
            edge_act_split = tf.split(tf.reshape(logits[1], [-1]), 
                                    edge_dims*tf.constant(oc.N_EDGE_ACTIONS, dtype=tf.int32))

            # This should be replaced with a tensorarray but works and can handle ragged tensors more easily
            reshaped_logits = []
            # Concatenate node, edge and stop actions for each observation
            for i in range(n_batched):
                reshaped_logits.append(tf.concat([node_act_split[i], edge_act_split[i], stop_actions[i]], axis=0))
            logits = tf.ragged.stack(reshaped_logits)

            # Due to bug in tf.map_fn with ragged tensors we need to pad logits with minimum values
            # see https://github.com/keras-team/keras/issues/16354
            logits = logits.to_tensor(default_value=tf.constant(logits.dtype.min, dtype=logits.dtype))

            # Apply mask
            logits = tf.where(mask_minibatched, logits, logits.dtype.min)
            
            # Build Categorical distibution from logits of all observations
            distr = tfp.distributions.Categorical(logits=logits, validate_args=True, 
                                                allow_nan_stats=False)
            #-------------------------------------------------------------------------------------------------------------
            

            new_log_prob = distr.log_prob(actions)
            entropy = distr.entropy()
    
            new_value = self.critic_model(observation_minibatched)

            # log(p_new/p_old)
            logratio = new_log_prob - log_probs
            # p_new/p_old
            ratio = tf.exp(logratio)

            #Normalize advantages on a minibtach_level
            if self.normalize_advantages:
                advantage_mean, advantage_std = (
                    tf.reduce_mean(mb_advantages),
                    tf.math.reduce_std(mb_advantages),
                )
                # Make sure we dont divide by zero
                advantage_std = tf.math.maximum(advantage_std, tf.constant(1e-6, dtype=tf.float32))
                mb_advantages = (mb_advantages - advantage_mean) / advantage_std

            # Simplified PPO clip objective, 
            # see https://spinningup.openai.com/en/latest/algorithms/ppo.html
            min_advantage = tf.where(
                    mb_advantages > 0,
                    (1 + clip_ratio) * mb_advantages,
                    (1 - clip_ratio) * mb_advantages,)
            
            pol_grad_loss = -tf.reduce_mean(tf.minimum(ratio * mb_advantages, min_advantage)) / n_parallel

            # Value function loss
            v_loss = 0.5 * tf.reduce_mean((new_value - returns)**2) / n_parallel
            
            # Entropy should be high to encourage exploration
            entropy_loss = tf.reduce_mean(entropy) / n_parallel

            loss = pol_grad_loss - ent_coeff * entropy_loss + value_coeff * v_loss 
        
        # Number of clipped actions in current
        n_clipped = tf.reduce_sum(tf.cast(ratio * mb_advantages > min_advantage, dtype=tf.int32))

        # KL between new and old policy
        kl = tf.reduce_mean((ratio - 1) - logratio)        
        
        grads = tape.gradient(loss, self.actor_model.trainable_variables + self.critic_model.trainable_variables)

        # Clip gradient absolute to avoid overflow in tf.norm
        grads = [None if gradient is None else tf.clip_by_value(gradient, -abs_grad_clip, abs_grad_clip)
                 for gradient in grads]

        # Clip gradient norm
        grads, _ = tf.clip_by_global_norm(grads, grad_norm_clip)

        self.optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables + self.critic_model.trainable_variables))

        return kl, pol_grad_loss, value_coeff * v_loss, - ent_coeff * entropy_loss, n_clipped



                    




