# The TrajectoryBuffer class is used to store the trajectories of the agents.
import tensorflow as tf
import numpy as np

class TrajectoryBuffer():
    """Stores the experiences of the agent during the sampling phase"""
    def __init__(self, n_envs, n_steps):
        # Amount of parallel environments
        self.n_envs = n_envs
        # Amount of steps sampled per environment
        self.n_steps = n_steps
        self.reset()

    def add_step(self, observations, actions: np.ndarray, rewards: np.ndarray, values: np.ndarray, 
                 logprobability_ts: np.ndarray, dones: np.ndarray, mask):
        """observations: list of observations of ZX env,
        mask: list of boolean np arrays,
        Saves one step of the trajectory"""
        self.observation[self.step] = observations
        self.action[self.step] = actions
        self.reward[self.step] = rewards
        self.value[self.step] = values
        self.logprobability_t[self.step] =logprobability_ts
        self.dones[self.step] = dones
        self.mask[self.step] = mask

        self.step +=1

    def reset(self):
        """Resets the buffer"""
        self.step = 0
        self.dones = np.zeros((self.n_steps + 1, self.n_envs), dtype=np.int32)
        self.action = np.zeros((self.n_steps, self.n_envs), dtype=np.int32)
        self.value = np.zeros((self.n_steps + 1, self.n_envs), dtype=np.float32)
        self.reward = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.logprobability_t = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)

        self.observation = np.empty((self.n_steps, self.n_envs), dtype=np.dtype("object"))
        self.mask = np.empty((self.n_steps, self.n_envs), dtype=np.dtype("object"))
    
    def add_value_done(self, next_value, done):
        """Adds the value and done of the last step of the training phase to the buffer"""
        assert(self.step==self.n_steps)
        self.dones[self.step] = done
        self.value[self.step] = next_value


    def get_episode(self) -> tuple:
        """returns the saved experiences"""
        return (self.observation, 
                tf.convert_to_tensor(self.action, dtype=tf.int32), 
                tf.convert_to_tensor(self.reward, dtype=tf.float32), 
                tf.convert_to_tensor(self.value, dtype=tf.float32),
                tf.convert_to_tensor(self.logprobability_t, dtype=tf.float32), 
                tf.convert_to_tensor(self.dones, dtype=tf.int32),
                self.mask 
                )