# This file holds an implementation of a greedy startegy and 
# simulated annealing for optimizing ZX diagrams.
import numpy as np
import copy
from . import action_conversion_utils as acu
from .VecAsyncEnvironment import VecZXCalculus 
from .ZX_env_max import ZXCalculus, get_neighbours, check_consistent_diagram
from .own_constants import INPUT, OUTPUT
        
class GreedyCleverAgent():
    """This agent simplifies a ZX diagram by always choosing the action with the highest reward
    as long as non-negative rewards are available."""

    def step_trajectory(self, env: VecZXCalculus, seed=0, reset=True)-> tuple:
        """env: ZX environement to optimize,
          seed: random seed,
          reset: if True, env is reset, else env is not reset and the current state is used as start state,
          returns: tuple of (obs_list, mask_list, reward_list, done_list, initial_spiders),

          Optimizes env for a whole trajectory.
          """
        self.rng = np.random.default_rng(seed)
        obs_list = []
        mask_list = []
        reward_list = []
        done_list = []
        if reset:
            observation, mask = env.reset()
        else:
            observation, mask = env.env_list[0].get_observation_mask()
            observation = [observation]
            mask = [mask]

        initial_spiders = observation[0][-3]
        done = 0
        # Step while trajectory is no finished
        i=0
        while done == 0:
            i +=1
            next_observation, next_mask, reward, next_done = self.step(env)
            obs_list.append(observation)
            mask_list.append(mask)
            reward_list.append(reward[0])
            done_list.append(done)
            observation = next_observation
            done = next_done
            mask = next_mask
        return obs_list, mask_list, np.array(reward_list), done_list, initial_spiders

    def step(self, env: VecZXCalculus)->tuple:
        "Do one optimization step in env"
        action = self.find_greedy_action(env)
        observation_new, mask, reward, done = env.step([action, ])
        return observation_new, mask, reward, done
    
    def find_greedy_action(self, env: VecZXCalculus)->int:
        """env: ZX environement to optimize,
        returns: index of next greedy action"""
        env = env.env_list[0]
        # Get mask to find valid actions
        mask, _ = env.get_mask_counts()
        valid_act_idcs = np.where(mask)[0]
        # Get rewards for all valid actions
        rewards = np.zeros(len(valid_act_idcs))
        for i, action in enumerate(valid_act_idcs):
            action_idx = acu.get_action_type_idx(len(env.colors), len(env.source), action)
            action_name = acu.get_action_name(action_idx)
            # Make sure that unmerge rule and stop action are not chosen
            if  (action_name == "start_unmerge_rule" or 
                 action_name == "stop_action" or
                 action_name == "stop_action"):
                rewards[i] = -1
            else:
                _, _, reward, _ = copy.deepcopy(env).step(action)
                rewards[i] = reward
        if np.max(rewards) < 0:
            # Return stop action
            return len(mask)-1
        elif np.max(rewards) == 0:
            # Return random action with reward 0
            zero_act_idcs = [act_id for rew, act_id in zip(rewards, valid_act_idcs) if rew == 0]
            return self.rng.choice(zero_act_idcs)
        else:
            # Return action with maximal reward
            action = np.argmax(rewards)
            return valid_act_idcs[action]
        
class AfterPYZXAgent():
    def simplify(self, env: ZXCalculus):
        # Get mask to find valid actions
        mask, _ = env.get_mask_counts()
        # Unfuse all Hadamards that are on the inside of the diagram
        found_unfuse = True
        while found_unfuse:
            found_unfuse = False
            valid_act_idcs = np.where(mask)[0]
            for i, action in enumerate(valid_act_idcs):
                action_idx = acu.get_action_type_idx(len(env.colors), len(env.source), action)
                action_name = acu.get_action_name(action_idx)
                # Make sure that unmerge rule and stop action are not chosen
                if (action_name == "split_hadamard"):
                    target = acu.get_action_target(action, len(env.colors), len(env.source))
                    # only unfuse middle hadamards
                    # check_consistent_diagram(env.colors, env.angles, env.selected_node, env.source, env.target, env.selected_edges)
                    neighbours = get_neighbours(target, env.source, env.target)
                    assert len(neighbours) == 2, "Hadamard has to have two neighbours"
                    # if len(neighbours) != 2:
                    #     check_consistent_diagram(env.colors, env.angles, env.selected_node, env.source, env.target, env.selected_edges)
                    if not (
                        np.all(env.colors[neighbours[0]]  == INPUT) or
                        np.all(env.colors[neighbours[0]]  == OUTPUT) or
                        np.all(env.colors[neighbours[1]]  == INPUT) or
                        np.all(env.colors[neighbours[1]]  == OUTPUT)
                    ):
                        observation_new, mask, reward, done = env.step(action)
                        #check_consistent_diagram(*observation_new[:6])
                        found_unfuse = True
                        break
        # merge all possible nodes
        found_merge = True
        while found_merge:
            found_merge = False
            valid_act_idcs = np.where(mask)[0]
            for i, action in enumerate(valid_act_idcs):
                action_idx = acu.get_action_type_idx(len(env.colors), len(env.source), action)
                action_name = acu.get_action_name(action_idx)
                # Make sure that unmerge rule and stop action are not chosen
                if (action_name == "merge_rule"):
                    observation_new, mask, reward, done = env.step(action)
                    found_merge = True
                    break
        return env
            


class AnnealingAgent():
    """This agent simplifies a ZX diagram using simulated annealing."""
     
    def optimize_env(self, env: VecZXCalculus, start_temp: float, n_steps: int, anneal_type: str, 
                     seed:int=0, allow_stop_action:bool=False, exp_factor:float=None, 
                     reset:bool=True, start_unm_neg_rew:bool=True)->tuple:
        """env: ZX environement to optimize,
        start_temp: starting temperature,
        n_steps: maximum number of steps in the optimization,
        anneal_type: "linear" or "exponential",
        seed: random seed,
        allow_stop_action: if True, stop action is allowed,
        exp_factor: factor for exponential temperature annealing,
        reset: if True, env is reset, else env is not reset and the current state is used as start state,
        start_unm_neg_rew: if True, start unmerge rule has reward -1, else 0,
        returns: tuple of (obs_list, mask_list, reward_list, done_list),

        Optimizes ZX env using simulated annealing for n_steps.
        """
        rng = np.random.default_rng(seed)
        obs_list = []
        mask_list = []
        reward_list = []
        done_list = []
        env = env.env_list[0]
        # Use current state as start state or reset env
        if reset:
            observation, mask = env.reset()
        else:
            observation, mask = env.get_observation_mask()
        # Don't allow agent to sample stop action
        if not allow_stop_action:
            mask[-1] = False
        done = 0
        # Optimize for n_steps
        for step in range(n_steps):
            # Get current temperature
            if anneal_type == "linear":
                temp = self.get_temp_linear(start_temp, n_steps, step)
            elif anneal_type == "exponential":
                temp = self.get_temp_exp(start_temp, step, exp_factor)
            env_cop = copy.deepcopy(env)

            # Choose random action and step
            valid_act_idcs = np.where(mask)[0]
            try:
                action = rng.choice(valid_act_idcs)
            except Exception as e:
                # This happens only if no action is available in the diagram
                return obs_list, mask_list, np.array(reward_list), done_list

            next_observation, next_mask, reward, next_done = env_cop.step(action)

            action_idx = acu.get_action_type_idx(len(env.colors), len(env.source), action)
            action_name = acu.get_action_name(action_idx)

            # Move -1 reward from stop_unmerge_rule to start_unmerge_rule
            if start_unm_neg_rew and action_name == "start_unmerge_rule":
                reward = reward - 1
            elif start_unm_neg_rew and action_name == "stop_unmerge_rule":
                reward = reward + 1
                
            # Accept action
            if reward >= 0:
                env = env_cop
                rew = reward

                obs_list.append(observation)
                mask_list.append(mask)
                reward_list.append(rew)
                done_list.append(done)
                observation = next_observation
                done = next_done
                mask = next_mask
                if not allow_stop_action:
                    mask[-1] = False
            else:
                if temp <= 0:
                    p_accept = 0
                else:
                    p_accept = np.exp(reward/temp)
                # Accept action with probability exp(reward/temp)
                if rng.random() < p_accept:
                    env = env_cop
                    rew = reward

                    obs_list.append(observation)
                    mask_list.append(mask)
                    reward_list.append(rew)
                    done_list.append(done)
                    observation = next_observation
                    done = next_done
                    mask = next_mask
                    if not allow_stop_action:
                        mask[-1] = False
                else:
                    rew = 0
                    obs_list.append(observation)
                    mask_list.append(mask)
                    reward_list.append(rew)
                    done_list.append(done)
            
        return obs_list, mask_list, np.array(reward_list), done_list
    

    def get_temp_linear(self, start_temp:float, n_steps:int, step:int)->float:
        """start_temp: starting temperature,
        n_steps: maximum number of steps in the optimization,
        step: current step,
        returns: current temperature,
        Calculates the current temperature in a linear annealing schedule."""
        return start_temp * (n_steps - step) / n_steps
    
    def get_temp_exp(self, start_temp:float, step:int, factor:float)->float:
        """start_temp: starting temperature,
        step: current step,
        returns: current temperature,
        Calculates the current temperature in an exponential annealing schedule."""
        return start_temp * np.exp(-factor*step)