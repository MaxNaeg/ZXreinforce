
# This file contains the ZXCalculus environment class and functions to apply actions to the diagram
import copy
import pickle

import numpy as np

from collections import deque

from .own_constants import (INPUT, OUTPUT, GREEN, RED, HADAMARD, 
                      ZERO, PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE,
                      ANGLE_LIST, N_NODE_ACTIONS, N_EDGE_ACTIONS)


class ZXCalculus():
    """Class for single ZX-calculus environment"""
    def __init__(self,
                 max_steps:int=1000, 
                 add_reward_per_step:float=-0.05,
                 resetter=None,
                 check_consistencty:bool=False,
                 count_down_from:int=20,
                 dont_allow_stop:bool=False,
                 extra_state_info:bool=True,
                 adapted_reward:bool=True):
        """max_steps: maximum number of steps per trajectory,
        add_reward_per_step: reward added per step,
        resetter: object that can reset the environment,
        check_consistencty: if True, checks consistency of diagram after each step,
        count_down_from: start stop counter from this number,
        dont_allow_stop: if True, stop action is only allowed if no other action is available,
        """
        
        self.add_reward_per_step = add_reward_per_step
        self.check_consistencty = check_consistencty
        self.max_steps = max_steps        
        self.resetter = resetter
        self.count_down_from = count_down_from
        self.dont_allow_stop = dont_allow_stop
        self.extra_state_info = extra_state_info
        self.adapted_reward = adapted_reward

 

    def load_observation(self, observation:tuple):
        """ observation: (colors, angles, selected_node, source, target, selected_edges)
        Loads observation from into environment"""

        (self.colors, self.angles, self.selected_node, self.source, self.target, 
         self.selected_edges, _, _, _)= observation
        
        self.previous_spiders = self.n_spiders
        self.current_spiders = self.n_spiders
        self.step_counter = 0
        self.max_diff = 0

    def reset(self)->tuple:
        '''
        returns: (observation, mask), 
        where observation is (colors, angles, selected_node, source, target, selected_edges)
        Resets the environment:
        1. Sample ZX_diagram.
        2. Set the step counter to zero.
        Return new initial observation
        '''
        colors, angles, selected_node, source, target, selected_edges = self.resetter.reset()


        if self.check_consistencty:
            check_consistent_diagram(colors, angles, selected_node, source, target, selected_edges)

        self.colors = colors
        self.angles = angles
        self.source = source
        self.target = target
        self.selected_node = selected_node
        self.selected_edges = selected_edges

        self.step_counter = 0
        self.max_diff = 0


        

        # For keep track of previous spiders for reward function
        self.previous_spiders = self.n_spiders
        self.current_spiders = self.n_spiders
        
        return self.get_observation_mask()
    
    @property
    def n_spiders(self)->int:
        """Number of nodes in diagram"""
        return len(self.colors)
    
    @property
    def n_edges(self):
        """Number of edges in diagram"""
        return len(self.source)
    
    def get_observation_mask(self)-> tuple:
        """returns observation and mask
        observation: (colors, angles, selected_node, source, target, selected_edges)"""

        # Get mask and number of allowed relevant actions
        mask, rel_action_counts = self.get_mask_counts()
        # Get other global features
        n_spider = self.n_spiders
        n_edges = len(self.source)
        n_red = np.count_nonzero(np.all(self.colors==RED, axis=1))
        n_green = np.count_nonzero(np.all(self.colors==GREEN, axis=1))
        n_hada = np.count_nonzero(np.all(self.colors==HADAMARD, axis=1))

        n_zero = np.count_nonzero(np.all(self.angles==ZERO, axis=1))
        n_pi = np.count_nonzero(np.all(self.angles==PI, axis=1))
        n_arb = np.count_nonzero(np.all(self.angles==ARBITRARY, axis=1))
        minigame = np.sum(self.selected_node)

        # Count down at end of trajectory only
        if self.max_steps - self.step_counter < self.count_down_from:
            count_down = self.max_steps - self.step_counter
        else:
            count_down = self.count_down_from

        if self.extra_state_info:
            info_state = self.max_diff
        else:
            info_state = 0.

        context_features = np.array([count_down, info_state, n_spider, n_red/n_spider, n_green/n_spider, n_hada/n_spider,
                                       n_zero/n_spider, n_pi/n_spider, n_arb/n_spider, n_edges, minigame], dtype=np.float32)
        context_features = np.append(context_features, rel_action_counts)


        observation = [np.array(self.colors, dtype=np.int32), 
                        np.array(self.angles, dtype=np.int32),
                        np.array(self.selected_node, dtype=np.int32),
                        np.array(self.source, dtype=np.int32),
                        np.array(self.target, dtype=np.int32),
                        np.array(self.selected_edges, dtype=np.int32),
                        np.array(len(self.colors), dtype=np.int32), 
                        np.array(len(self.source), dtype=np.int32),
                        np.array(context_features, dtype=np.float32)]

        return observation, mask

    def get_mask_counts(self):
        """returns mask and number of allowed relevant actions"""
        return get_mask_counts(self.colors, self.angles, self.selected_node, self.source, 
                        self.target, self.selected_edges, self.dont_allow_stop)
    
    def step(self, action: int) -> tuple[int, int]:
        '''action: int, action to be applied to the environment,
        returns: (observation, mask, reward, done)
        '''

        ## Add the step counter for stopping criteria
        self.step_counter += 1

        # Check if trajectory is over
        if (self.step_counter >= self.max_steps or
            (action == N_EDGE_ACTIONS * self.n_edges + N_NODE_ACTIONS * self.n_spiders and not self.dont_allow_stop)
            ): 
            # Return observation, mask and reward, end_episode
            observation, mask = self.reset()
            return observation, mask, 0, 1
        elif action == N_EDGE_ACTIONS * self.n_edges + N_NODE_ACTIONS * self.n_spiders and self.dont_allow_stop:
            observation, mask = self.get_observation_mask()
            return observation, mask, 0, 0
        else:
            # Applies the action
            (mod_colors, mod_angles, mod_selected_node, mod_source, mod_target,
            mod_selected_edges) = apply_action(copy.deepcopy(self.colors), 
                                                copy.deepcopy(self.angles), 
                                                copy.deepcopy(self.selected_node),
                                                copy.deepcopy(self.source),
                                                copy.deepcopy(self.target), 
                                                copy.deepcopy(self.selected_edges), 
                                                copy.deepcopy(action))

            if self.check_consistencty:
                try:
                    check_consistent_diagram(mod_colors, mod_angles, mod_selected_node,
                                             mod_source, mod_target, mod_selected_edges)
                except AssertionError as e:

                    save(self.colors, self.angles, self.selected_node, 
                         self.source, self.target, self.selected_edges, self.step_counter)

                    with open(f"action{self.step_counter}.pkl", 'wb') as f:
                        pickle.dump(action, f)

                    raise e


            # Save new state
            self.colors = mod_colors
            self.angles = mod_angles
            self.source = mod_source
            self.target = mod_target
            self.selected_node = mod_selected_node
            self.selected_edges = mod_selected_edges


            ## Calculate the reward
            self.previous_spiders = self.current_spiders
            self.current_spiders = self.n_spiders
            reward = self.delta_spiders()

            if reward + self.max_diff >= 0:
                reward_tilde = reward + self.max_diff
                self.max_diff = 0
            else:
                reward_tilde = 0
                self.max_diff = reward + self.max_diff

            # return observation, mask, reward, done
            observation, mask = self.get_observation_mask()
            if self.adapted_reward:
                rew_returned = reward_tilde + self.add_reward_per_step
            else:
                rew_returned = reward + self.add_reward_per_step

            return observation, mask, rew_returned, 0
    
    def delta_spiders(self)->int:
        """returns reward"""
        return self.previous_spiders - self.current_spiders
    

def save(colors:np.ndarray, angles:np.ndarray, selected_node:np.ndarray, 
         source:np.ndarray, target:np.ndarray, selected_edges:np.ndarray, idx:int):
    """saves the current state of the environment at step idx"""
    with open(f"colors{idx}.pkl", 'wb') as f:
        pickle.dump(colors, f)
    with open(f"angles{idx}.pkl", 'wb') as f:
        pickle.dump(angles, f)
    with open(f"source{idx}.pkl", 'wb') as f:
        pickle.dump(source, f)
    with open(f"target{idx}.pkl", 'wb') as f:
        pickle.dump(target, f)
    with open(f"selected_node{idx}.pkl", 'wb') as f:
        pickle.dump(selected_node, f)
    with open(f"selected_edges{idx}.pkl", 'wb') as f:
        pickle.dump(selected_edges, f)


# The following functions are stand-alone functions to potentially make them jit compatible in the future
#--------------------------------------------------------------------------------------------------------

def check_consistent_diagram(colors, angles, selected_node, source, target, selected_edges):
    """Checks if the diagram is consistent. Raises assertion error"""
    # Check consistencies of length
    assert len(colors) == len(angles), "colors and angles different size"
    assert len(source) == len(target), "source and target different size"
    assert len(selected_node) == len(colors), "selected_node and color size"
    assert len(selected_edges) == len(target), "selected_edges and target different size"
    # Check that no node that diesnt exist has an edge
    assert np.max(source) < len(colors), "nodes connected as source, that dont exist"
    assert np.max(target) < len(colors), "nodes connected as target, that dont exist"
    # Check selected node
    assert np.sum(selected_node) <= 1, "multiple selected nodes"

    if np.sum(selected_node) < 1:
        assert np.sum(selected_edges) == 0, "edges colored while not in minigame"


    # Check that there are no ids
    mask_remove_id_color = get_mask_id_color(colors, angles, source, target)
    mask_remove_id_hadamard = get_mask_id_hadamard(colors, angles, source, target)
    idcs_color = list(np.where(mask_remove_id_color)[0])
    idcs_hada = list(np.where(mask_remove_id_hadamard)[0])
    assert len(idcs_color) == 0, "contains color id"
    assert len(idcs_hada) == 0, "contains hadamard id"

    
    # Check that there are no self loops
    assert not np.any(source == target), "node with self loop"

    # Check that there are no redundant edges
    edge_sets_list = [{s, t} for s,t in zip(source, target)]
    flag_redundant_edges = True
    for edge in edge_sets_list:
        if edge_sets_list.count(edge) > 1:
            flag_redundant_edges = False
            break
    assert flag_redundant_edges, "redundant edges in diagram"

    # Check that input/outputs have only one connected edge
    # Check that hadamards have exactly two neighbours
    for idx, color in enumerate(colors):
        if np.all(color==np.array(INPUT)) or np.all(color==np.array(OUTPUT)):
            neigh = get_neighbours(idx, source, target)
            assert len(neigh == 1), "input/output with multiple connected neighbours"
        elif np.all(color==np.array(HADAMARD)):
            neigh = get_neighbours(idx, source, target)
            assert len(neigh == 2), "hadamard not with two neighbors"

    # Check that there are no disconnected nodes:
    input_idcs = np.where(np.all(colors == INPUT, axis=1))
    output_idcs = np.where(np.all(colors == OUTPUT, axis=1))
    search_from_idcs = np.append(input_idcs, output_idcs)
    connected_nodes = []
    # Start searching from each input/ouput node
    for src_idx in search_from_idcs:
        # if not already reached this node:
        if src_idx not in connected_nodes:
            # get all connected nodes
            conn_nodes = breadth_first_search(src_idx, source, target, len(colors))
            connected_nodes += conn_nodes
     # Flatten  and sort     
    connected_nodes = np.array(connected_nodes, dtype=np.int32)
    all_nodes = np.arange(len(colors), dtype=np.int32)
    not_connected = all_nodes[np.in1d(all_nodes,connected_nodes,invert=True)]
    assert len(not_connected) == 0, "disconnected nodes in diagram"

def get_mask_counts(colors, angles, selected_node, source, target, selected_edges, dont_allow_stop=False):
    """returns mask and number of allowed relevant actions"""
    n_spiders = len(colors)
    n_edges = len(source)

    relevant_action_count_idcs = [4, 5, 6, 8, 9, 10, 11]
    norm_by = np.array([n_spiders, n_spiders, n_edges, n_edges, n_edges, n_edges, n_edges])
    
    mask_list = get_mask_list(colors, angles, selected_node, source, target, selected_edges, dont_allow_stop)
    rel_action_amounts = np.array([len(np.where(mask)[0]) for idx, mask in enumerate(mask_list)
                          if idx in relevant_action_count_idcs], dtype=np.float32)
    if not np.any(norm_by == 0):
        rel_action_amounts = rel_action_amounts / norm_by
    else:
        rel_action_amounts = np.zeros_like(rel_action_amounts)
    
    # If start_Unmerge rule selected compute different mask
    if np.sum(selected_node) == 0:
        return create_reshaped_mask(mask_list), np.array(rel_action_amounts, dtype=np.float32)
    else:
        mask_list = get_mask_list_minigame(colors, angles, selected_node, source, target, selected_edges)
        return create_reshaped_mask(mask_list), np.array(rel_action_amounts, dtype=np.float32)

def create_reshaped_mask(mask_list):
    """Reshape mask to be compatible with output of tf_gnn"""
    node_mask = np.array(mask_list[:N_NODE_ACTIONS])
    node_mask = np.ravel(node_mask, order="F")

    edge_mask = np.array(mask_list[N_NODE_ACTIONS:N_NODE_ACTIONS+N_EDGE_ACTIONS])
    edge_mask = np.ravel(edge_mask, order="F")

    return np.concatenate([node_mask, edge_mask, mask_list[-1]])

def get_mask_list_minigame(colors, angles, selected_node, source, target, selected_edges)->list:
    """returns mask if start_Unmerge action was selected"""
    n_nodes = len(colors)
    n_edges= len(source)

    node_mask = np.full(n_nodes, False)
    edge_mask = np.full(n_edges, False)


    idx_unmerge = np.where(selected_node==1)[0][0]
    mask_stop_unmerge_rule = np.full(n_nodes, False)

    n_selected = np.sum(selected_edges)

    
    if n_selected >= 2:
        mask_stop_unmerge_rule[idx_unmerge] = True

    
    conn_edge_idcs = np.where(source==idx_unmerge)[0]
    conn_edge_idcs = np.append(conn_edge_idcs, np.where(target==idx_unmerge)[0])

    mask_color_edge = np.full(n_edges, False)
    n_neigh = len(conn_edge_idcs)
    if not (np.all(angles[idx_unmerge] == ZERO) and n_selected >= n_neigh-2):
        # Only connected edges allowed
        mask_color_edge[conn_edge_idcs] = True
        # If already colored not allowed
        mask_color_edge[np.where(selected_edges==1)[0]] = False
    
    # There is no reason why you would stop in the minigame
    mask_stop_action = np.array([False])

    return [copy.copy(node_mask), mask_stop_unmerge_rule, copy.copy(node_mask), copy.copy(node_mask), copy.copy(node_mask),
     copy.copy(node_mask), copy.copy(edge_mask), mask_color_edge, copy.copy(edge_mask), copy.copy(edge_mask), copy.copy(edge_mask), 
     copy.copy(edge_mask), mask_stop_action]


def get_mask_list(colors, angles, selected_node, source, target, 
                  selected_edges, dont_allow_stop=False, dont_allow_unmerge=False)->list:
    """dont_allow_stop: if True, stop action is only allowed if no other action is available,
    dont_allow_unmerge: if True, start_unmerge action is never allowed
    returns: list of mask for all actions"""
    
    n_nodes = len(colors)
    n_edges= len(source)

    ### NODE ACTIONS

    # Mask for only green and red spiders
    mask_red_or_green = np.full(n_nodes, False)
    mask_red_or_green[np.all(colors==GREEN, axis=1)] = True
    mask_red_or_green[np.all(colors==RED, axis=1)] = True

    # All green or red nodes allowed
    mask_start_unmerge_rule = copy.copy(mask_red_or_green)
    # Arbitrary nodes need at least two neighbors
    # Count how many edges a node has
    if len(np.bincount(source, minlength=n_nodes)) != len(np.bincount(target, minlength=n_nodes)):
        print("error")
    ocurrances = np.bincount(source, minlength=n_nodes) + np.bincount(target, minlength=n_nodes)
    # If only one neighbour not allowed
    mask_start_unmerge_rule[ocurrances<=1] = False
    # Zero nodes need to have at least 3 neighbors
    zero_idcs = np.where(np.all(angles == ZERO, axis=-1))[0]
    for idx in zero_idcs:
        if ocurrances[idx] <= 3:
            mask_start_unmerge_rule[idx] = False

    if dont_allow_unmerge:
        mask_start_unmerge_rule = np.full(n_nodes, False)


    # Never allowed
    mask_stop_unmerge_rule = np.full(n_nodes, False)

    # Mask for (h) sym: change color and put hadamards
    # Also allowed on all green and red nodes
    mask_color_change = copy.copy(mask_red_or_green)

    # Mask on Hadamard handlings splitting
    mask_hadamard_split = np.full(n_nodes, False)
    mask_hadamard_split[np.all(colors==HADAMARD, axis=1)] = True
    
    # Mask on Hadamard merging (compare p 26 LARGE ZX caculus)
    # Mask for colored spiders with exactly two edges
    mask_hadamard_merge = copy.copy(mask_red_or_green)
    # Count how many edges a node has
    # If exactly two edges, allowed
    mask_hadamard_merge[ocurrances!=2] = False

    mask_euler = copy.deepcopy(mask_hadamard_merge)

    mask_hadamard_merge[np.logical_not(np.all(angles==PI_half, axis=1))] = False
    # get list of both neighbour indices shape (np.where(mask_hadamard_merge == True), 2)
    idcs_valid = np.where(mask_hadamard_merge)[0]
    # There should be at most 2 neighbors
    # neighbour_idcs = np.zeros((len(idcs_valid), 2), dtype=np.int32)
    # Get neighbours of all still allowed actions
    for idx in idcs_valid:
        # These are garantied to be only two
        neighbour_idcs = get_neighbours(idx, source, target)
        for neigh in neighbour_idcs:
            # Check amount of neighbours of nearest neighbour
            if len(get_neighbours(neigh, source, target)) != 2:
                mask_hadamard_merge[idx] = False
                break
            # Check that opposite color
            if np.logical_not(np.all(colors[idx] + colors[neigh] == np.array(RED) + np.array(GREEN))):
                mask_hadamard_merge[idx] = False
                break
            # Check that angle correct
            if np.logical_not(np.all(angles[neigh]==PI_half, axis=-1)):
                mask_hadamard_merge[idx] = False
                break

    # Euler mask starting with only green/red nodes with two neighbors
    mask_euler[np.logical_not(np.logical_or(np.all(angles==ARBITRARY, axis=1),
                             np.logical_or(np.all(angles==PI_half, axis=1), 
                                           np.all(angles==PI_three_half, axis=1))))] = False
    # Get list of both neighbour indices
    idcs_valid = np.where(mask_euler)[0]
    # There should be at most 2 neighbors
    # Get neighbours of all still allowed actions
    for idx in idcs_valid:
        # These are garantied to be only two
        neighbour_idcs = get_neighbours(idx, source, target)
        for neigh in neighbour_idcs:
            # Check amount of neighbours of nearest neighbour
            if len(get_neighbours(neigh, source, target)) != 2:
                mask_euler[idx] = False
                break
            # Check that opposite color
            if not np.all(colors[idx] + colors[neigh] == np.array(RED) + np.array(GREEN)):
                mask_euler[idx] = False
                break
            # Check that angle correct
            if not (np.all(angles[neigh]==PI_half) or 
                    np.all(angles[neigh]==PI_three_half) or 
                    np.all(angles[neigh]==ARBITRARY)):
                mask_euler[idx] = False
                break

    ### EDGE ACTIONS

    # Used for more actions
    neighbor_angles = np.stack([angles[source], angles[target]], axis=1)
    neighbor_colors = np.stack([colors[source], colors[target]], axis=1)

    
    # Mask for action (f) right (and hh right). Merging two spiders together if same color
    mask_merge_rule = np.all(colors[source] == colors[target], axis=-1)
    # Make sure inputs/outputs are not merged, merging hadamards is allowed, but has other effect!
    mask_merge_rule[np.all(colors[source] == INPUT, axis=-1)] = False
    mask_merge_rule[np.all(colors[target] == INPUT, axis=-1)] = False
    mask_merge_rule[np.all(colors[source] == OUTPUT, axis=-1)] = False
    mask_merge_rule[np.all(colors[target] == OUTPUT, axis=-1)] = False

    # Mask for action (f) left coloring edges. This is allowed for all 
    # edges and there are two actions per edge
    mask_color_edge = np.full(n_edges, False)

    # Mask for (pi) right inserting two hadamards. 
    mask_pi_sym = np.full(n_edges, True)
    # Remove options where no neighbor has pi angle
    # np.sum(neighbor_angles, axis=1) sums angles of both neighbors together
    mask_pi_sym[np.sum(neighbor_angles, axis=1)[:, np.where(PI)[0]].flatten() == 0] = False
    # Check if neighbours have different colors and are both green or red
    mask_pi_sym[np.logical_not(np.all(
        np.sum(neighbor_colors, axis=1) == (np.array(GREEN) + np.array(RED)), axis=-1))] = False
    # Loop over still allowed actions
    idcs_valid = np.where(mask_pi_sym)[0]
    for idx in idcs_valid:
        # Take care: both could be PIs
        valid = False
        if np.all(angles[source[idx]] == PI):
            pi_idx = source[idx]
            arb_idx = target[idx]
            if len(get_neighbours(pi_idx, source, target)) == 2 and len(get_neighbours(arb_idx, source, target)) >= 2:
                valid =True
        if np.all(angles[target[idx]] == PI):
            pi_idx = target[idx]
            arb_idx = source[idx]
            if len(get_neighbours(pi_idx, source, target)) == 2 and len(get_neighbours(arb_idx, source, target)) >= 2:
                valid =True
        mask_pi_sym[idx] = valid
    
    # Mask for (c) right, zero angle
    mask_c_right = np.full(n_edges, True)
    # Remove options where no neighbor has angle 0
    mask_c_right[np.sum(neighbor_angles, axis=1)[:, np.where(ZERO)].flatten() == 0] = False
    # Check if neighbours have different colors and are both green or red
    mask_c_right[np.logical_not(np.all(
        np.sum(neighbor_colors, axis=1) == (np.array(GREEN) + np.array(RED)), axis=-1))] = False
    # Loop over still allowed actions
    idcs_valid = np.where(mask_c_right)[0]
    for idx in idcs_valid:
        # Take care. Both could be angle zero
        valid = False
        if np.all(angles[source[idx]] == ZERO):
            zero_idx = source[idx]
            if len(get_neighbours(zero_idx, source, target)) == 1:
                valid = True
        if np.all(angles[target[idx]] == ZERO):
            zero_idx = target[idx]
            if len(get_neighbours(zero_idx, source, target)) == 1:
                valid = True
        mask_c_right[idx] = valid

    # Mask for (c) right, zero angle
    mask_c_right_pi = np.full(n_edges, True)
    # Remove options where no neighbor has angle PI
    mask_c_right_pi[np.sum(neighbor_angles, axis=1)[:, np.where(PI)].flatten() == 0] = False
    # Check if neighbours have different colors and are both green or red
    mask_c_right_pi[np.logical_not(np.all(
        np.sum(neighbor_colors, axis=1) == (np.array(GREEN) + np.array(RED)), axis=-1))] = False
    # Loop over still allowed actions
    idcs_valid = np.where(mask_c_right_pi)[0]
    for idx in idcs_valid:
        # Take care. Both could be angle zero
        valid = False
        if np.all(angles[source[idx]] == PI):
            pi_idx = source[idx]
            if len(get_neighbours(pi_idx, source, target)) == 1:
                valid = True
        if np.all(angles[target[idx]] == PI):
            pi_idx = target[idx]
            if len(get_neighbours(pi_idx, source, target)) == 1:
                valid = True
        mask_c_right_pi[idx] = valid

    mask_c_right = np.logical_or(mask_c_right, mask_c_right_pi)

    # Mask for (b) right together with Mask for (b) left
    mask_b_right = np.full(n_edges, True)
    # Remove options where not both angles are zero
    mask_b_right[np.logical_not(np.sum(neighbor_angles, axis=1)[:, np.where(ZERO)].flatten() == 2)] = False
    # Check if neighbours have different colors and are both green or red
    mask_b_right[np.logical_not(np.all(
        np.sum(neighbor_colors, axis=1) == (np.array(GREEN) + np.array(RED)), axis=-1))] = False
    # Loop over still allowed actions
    idcs_valid = np.where(mask_b_right)[0]
    mask_b_left = copy.copy(mask_b_right)
    for idx in idcs_valid:
        if len(get_neighbours(source[idx], source, target)) != 3:
            mask_b_left[idx] = False
            mask_b_right[idx] = False
        elif len(get_neighbours(target[idx], source, target)) != 3:
            mask_b_left[idx] = False
            mask_b_right[idx] = False
        else:
            # Now go one step further for (b) left
            # Check neighbors of source_idx
            neighbors_source = get_neighbours(source[idx], source, target)
            neighbors_target = get_neighbours(target[idx], source, target)
            valid = False
            for sneigh in neighbors_source:
                if (np.all(angles[sneigh] == ZERO) and 
                    np.all(colors[source[idx]] + colors[sneigh] == np.array(GREEN) + np.array(RED))
                    and sneigh != target[idx]):
                    # If neighbour fulfills conditions in pricipal see if he connects to neighbour of target
                    nn_neighbours_source = get_neighbours(sneigh, source, target)
                    if len(nn_neighbours_source) == 3:
                        for t_neigh in neighbors_target:
                            # If connected to target neighbour with only three edges and of opposite color action is valid
                            if (t_neigh in nn_neighbours_source and 
                                np.all(colors[t_neigh] + colors[sneigh] == np.array(GREEN) + np.array(RED)) and
                                len(get_neighbours(t_neigh, source, target)) == 3 and
                                np.all(angles[t_neigh] == ZERO) and
                                t_neigh != source[idx]):
                                valid = True
            mask_b_left[idx] = valid

    mask_stop_action = np.array([True])

    if dont_allow_stop:
        if np.any(np.concatenate([mask_start_unmerge_rule, mask_stop_unmerge_rule, mask_color_change, mask_hadamard_split, mask_hadamard_merge, mask_euler,
            mask_merge_rule, mask_color_edge, mask_pi_sym, mask_c_right,
            mask_b_right, mask_b_left])):
            # There is another action allowed, so we set the stop action to false
            mask_stop_action = np.array([False])


    return [mask_start_unmerge_rule, mask_stop_unmerge_rule, mask_color_change, mask_hadamard_split, mask_hadamard_merge, mask_euler,
            mask_merge_rule, mask_color_edge, mask_pi_sym, mask_c_right,
            mask_b_right, mask_b_left, mask_stop_action]


def apply_action(colors, angles, selected_node, source, target, selected_edges, action)->tuple:
        """ returns: (colors, angles, selected_node, source, target, selected_edges),
        Applies an action to the diagram.
        Takes numpy arrays as input and gives modified numpy arrays as output"""


        n_nodes = len(colors)
        n_edges= len(source)

        node_actions_fn = [start_unmerge_rule, stop_unmerge_rule,
                            color_change, split_hadamard, merge_hadamard, euler_rule]
        edge_actions_fn = [merge_rule, color_edge,
                           pi_rule, copy_rule_right, bialgebra_right, bialgebra_left]

        if action < N_NODE_ACTIONS * n_nodes:
            # Action is node action
            action_idx = action//N_NODE_ACTIONS
            action_fn = node_actions_fn[action%N_NODE_ACTIONS]
        elif action < N_NODE_ACTIONS * n_nodes + N_EDGE_ACTIONS * n_edges:
            # Action is edge action
            action = action - (N_NODE_ACTIONS * n_nodes)
            action_fn = edge_actions_fn[action%N_EDGE_ACTIONS]
            action_idx = action // N_EDGE_ACTIONS
        else:
            raise Exception(f"Action index {action} out of range {N_NODE_ACTIONS * n_nodes + N_EDGE_ACTIONS * n_edges}")
        #Apply action
        return(action_fn(colors, angles, selected_node, source, target, selected_edges, action_idx))       

def get_mask_id_color(colors, angles, source, target)->np.ndarray:
    """Computes mask for Identity actions"""
    n_nodes = len(colors)
    # Mask for only green and red spiders
    mask_id_right = np.full(n_nodes, False)
    mask_id_right[np.all(colors==GREEN, axis=1)] = True
    mask_id_right[np.all(colors==RED, axis=1)] = True

    ocurrances = np.bincount(source, minlength=n_nodes) + np.bincount(target, minlength=n_nodes)
    # If exactly two edges, allowed
    mask_id_right[ocurrances!=2] = False
    # If angle zero, allowed
    mask_id_right[np.logical_not(np.all(angles==ZERO, axis=-1))] = False

    return mask_id_right

def get_mask_id_hadamard(colors, angles, source, target)->np.ndarray:
    """Computes mask for Hadamard Identity actions"""
    mask_had_right = np.logical_and(np.all(colors[source] == HADAMARD, axis=-1), 
                                    np.all(colors[target] == HADAMARD, axis=-1))
    return mask_had_right

def apply_auto_actions(colors, angles, source, target)->tuple:
    """Apply these actions after each step until no more possible:
    Remove multiple edges, Identities, and disconnected parts of the diagram"""
    acted = True
    while acted:
        n_nodes = len(colors)
        list_mod_nodes = np.arange(n_nodes)
        colors, angles, source, target = remove_multiple_edges(
            list_mod_nodes, colors, angles, source, target)
        colors, angles, source, target,  = remove_ids(
            colors, angles, source, target)
        if len(colors) == n_nodes:
            acted = False

    colors, angles, source, target = remove_disconnected_nodes(
        colors, angles, source, target)
        
    return colors, angles, source, target

def remove_ids(colors, angles, source, target):
    """Removes all ids (colored spiders and double hadamards) from the diagram"""
    mask_remove_id_color = get_mask_id_color(colors, angles, source, target)
    mask_remove_id_hadamard = get_mask_id_hadamard(colors, angles, source, target)
    idcs_color = list(np.where(mask_remove_id_color)[0])
    idcs_hada = list(np.where(mask_remove_id_hadamard)[0])
    # While Id actions possible:
    while len(idcs_color + idcs_hada) > 0:
        # Remove spider
        if len(idcs_color) > 0:
            colors, angles, _, source, target, _ = id_removal(
                colors, angles, None, source, target, None, idcs_color[0])
        # Remove two hadamards
        else:
            colors, angles, _, source, target, _ = merge_rule(
                colors, angles, None, source, target, None, idcs_hada[0])
            
        mask_remove_id_color = get_mask_id_color(colors, angles, source, target)
        mask_remove_id_hadamard = get_mask_id_hadamard(colors, angles, source, target)
        idcs_color = list(np.where(mask_remove_id_color)[0])
        idcs_hada = list(np.where(mask_remove_id_hadamard)[0])

    return colors, angles, source, target


# Actions--------------------------------------------------------------------------------------------
def start_unmerge_rule(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """start Unfuse action"""
    selected_node[action_idx] = 1
    return colors, angles, selected_node, source, target, selected_edges

def color_edge(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """mark edge action after start Unfuse"""
    selected_edges[action_idx] = 1
    return colors, angles, selected_node, source, target, selected_edges

def stop_unmerge_rule(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """stop Unfuse action"""
    color_parent = colors[action_idx]
    # Add node
    colors = np.row_stack((colors, color_parent))
    angles = np.row_stack((angles, ZERO))
    child_idx = len(colors) - 1
    
    # Move correct edges from parent to child:
    connected_edge_idcs_source = np.where(source==action_idx)[0]
    for edge_idx in connected_edge_idcs_source:
        if selected_edges[edge_idx] == 1:
            source[edge_idx] = child_idx
    connected_edge_idcs_target = np.where(target==action_idx)[0]
    for edge_idx in connected_edge_idcs_target:
        if selected_edges[edge_idx] == 1:
            target[edge_idx] = child_idx

    # Add edge between node and parent
    source, target = add_edge(
        child_idx, action_idx, source, target)

    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
    
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))


def merge_rule(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Fuse action"""
    # Get idcs of nodes to merge
    merge_idcs = np.array([source[action_idx],
                target[action_idx]])
    merge_into = np.min(merge_idcs)
    merge_from = np.max(merge_idcs)

    # Action applied to Green or red spiders
    if np.all(colors[merge_from] == RED) or np.all(colors[merge_from] == GREEN):

        # Add angles of merged nodes together
        angles[merge_into] = get_added_angle(angles[merge_into], angles[merge_from])

        # Delete edge between nodes to avoid self connections:
        source, target = remove_edge(
            action_idx, source, target)

        # Change source of edges
        source = np.where(source==merge_from, merge_into, source)
        target = np.where(target==merge_from, merge_into, target)

        colors, angles, source, target = remove_node_with_edges(
            merge_from, colors, angles, source, target)
        
    # This is an edge connecting to hadamards
    else:
        # Get indices of neighbours. There should be only two
        neighbour_1_idx = get_neighbours(merge_idcs[0], source, target)
        if neighbour_1_idx[0] != merge_idcs[1]:
            neighbour_1_idx = neighbour_1_idx[0]
        else:
            neighbour_1_idx = neighbour_1_idx[1]

        neighbour_2_idx = get_neighbours(merge_idcs[1], source, target)
        if neighbour_2_idx[0] != merge_idcs[0]:
            neighbour_2_idx = neighbour_2_idx[0]
        else:
            neighbour_2_idx = neighbour_2_idx[1]

        if neighbour_1_idx != neighbour_2_idx:
            # Add edge between neighbours
            source, target = add_edge(neighbour_1_idx, neighbour_2_idx, source, target)

            # adjust neighbour indices becasue hada nodes will be removed
            if merge_from < neighbour_1_idx:
                neighbour_1_idx -=1
            if merge_into < neighbour_1_idx:
                neighbour_1_idx -=1
            if merge_from < neighbour_2_idx:
                neighbour_2_idx -=1
            if merge_into < neighbour_2_idx:
                neighbour_2_idx -=1
    

        # Remove both hadamard nodes with connected edges, take care about order!:
        colors, angles, source, target = remove_node_with_edges(
            merge_from, colors, angles, source, target)
        colors, angles, source, target = remove_node_with_edges(
            merge_into, colors, angles, source, target)
        

        
    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
    return  (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def color_change(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Color change action"""
    # Change color
    if np.all(colors[action_idx] == RED):
        new_color = GREEN
    else:
        new_color = RED
    colors[action_idx] = new_color
    
    # Insert Hadamards, removes color from edges
    neighbours = get_neighbours(action_idx, source, target)
    for neigh_idx in neighbours:
        # Add ahdamard node
        colors = np.row_stack((colors, HADAMARD))
        angles = np.row_stack((angles, NO_ANGLE))
        idx_hada = len(colors) - 1
        # Connect hadamard node to action node
        source, target = add_edge(
            action_idx, idx_hada, source, target)
        # Connect hadamard node to neighbour node
        source, target = add_edge(
            neigh_idx, idx_hada, source, target)
        # Remove edge between neighbor node and action node
        edge_idx = get_edge_idx(action_idx, neigh_idx, source, target)
        source, target = remove_edge(
            edge_idx, source, target)
        
    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def id_removal(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """removes identity spider"""
    neighbours = get_neighbours(action_idx, source, target)
    # Add edge between neighbors
    source, target = add_edge(
        neighbours[0], neighbours[1], source, target)
    # Remove node
    colors, angles, source, target = remove_node_with_edges(
        action_idx, colors, angles, source, target)

    # Adjust neighbor node idcs for remove_multiple_edges
    mod_nodes = []
    for mod_n in neighbours:
        if mod_n < action_idx:
            mod_nodes.append(mod_n)
        else:
            mod_nodes.append(mod_n-1)
    
    # Remove unnescesarry edges
    colors, angles, source, target = remove_multiple_edges(
            mod_nodes, colors, angles, source, target)
    colors, angles, source, target = remove_disconnected_nodes(
        colors, angles, source, target)
    
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def split_hadamard(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Hadamard unfuse action"""
    # Change middle node
    colors[action_idx] = RED
    angles[action_idx] = PI_half
    idcs_neigh = get_neighbours(action_idx, source, target)

    # Add new nodes
    for neigh_idx in idcs_neigh:
        colors = np.row_stack((colors, GREEN))
        angles = np.row_stack((angles, PI_half))
        idx_new = len(colors) - 1
        # Connect new node to action node
        source, target = add_edge(
            action_idx, idx_new, source, target)
        # Connect new node to neighbour node
        source, target = add_edge(
            idx_new, neigh_idx, source, target)
        # Remove edge between neighbor node and action node
        edge_idx = get_edge_idx(action_idx, neigh_idx, source, target)
        source, target = remove_edge(
            edge_idx, source, target)
        
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def merge_hadamard(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Hadamard fuse action"""
    # Change middle node
    colors[action_idx] = HADAMARD
    angles[action_idx] = NO_ANGLE

    idcs_neigh = get_neighbours(action_idx, source, target)
    # Connect node to next nearest neighbours:
    for neigh_idx in idcs_neigh:
        nn_neigh_idcs = get_neighbours(neigh_idx, source, target)
        # Don't add self loop on action_node!
        nn_neigh_idx = nn_neigh_idcs[0] if not nn_neigh_idcs[0] == action_idx else nn_neigh_idcs[1]
        source, target = add_edge(
            nn_neigh_idx, action_idx, source, target)
        
    # Remove nearest neighbours
    colors, angles, source, target = remove_node_with_edges(
        np.max(idcs_neigh), colors, angles, source, target)    
    colors, angles, source, target = remove_node_with_edges(
        np.min(idcs_neigh), colors, angles, source, target)

    # Get new index of hadamard node
    if action_idx > np.max(idcs_neigh):
        action_idx -= 1
    if action_idx > np.min(idcs_neigh):
        action_idx -= 1

    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
    
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def pi_rule(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Pi action"""
    neigh_source = get_neighbours(source[action_idx], source, target)
    if len(neigh_source) == 2 and np.all(angles[source[action_idx]] == PI):
        pi_idx = source[action_idx]
        alpha_idx = target[action_idx]
    else:
        pi_idx = target[action_idx]
        alpha_idx = source[action_idx]
    color_pi = colors[pi_idx]
    # Change sign of alpha node:
    angle_alpha = angles[alpha_idx]
    if np.all(angle_alpha == ARBITRARY):
        angle_new = ARBITRARY
    else:
        idx = ANGLE_LIST.index(list(angle_alpha))
        angle_new = ANGLE_LIST[-idx]
    angles[alpha_idx] = angle_new
    # Add new nodes:
    neigh_alpha = get_neighbours(alpha_idx, source, target)
    for neigh in neigh_alpha:
        if neigh != pi_idx:
            edge_idx = get_edge_idx(neigh, alpha_idx, source, target)
            colors, angles, source, target = insert_node_on_edge(
                colors, angles, source, target, edge_idx, color_pi, PI)
    # Add edge over pi node
    # Get neighbor of pi that is not alpha node
    neigh_pi = get_neighbours(pi_idx, source, target)
    if neigh_pi[0] != alpha_idx:
        neigh_pi = neigh_pi[0]
    else:
        neigh_pi = neigh_pi[1]
    source, target = add_edge(
        alpha_idx, neigh_pi, source, target)
    # Remove pi node
    colors, angles, source, target = remove_node_with_edges(
        pi_idx, colors, angles, source, target)

    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def copy_rule_right(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Copy action"""
    # Find node with only one neighbor
    neigh_source = get_neighbours(source[action_idx], source, target)
    if len(neigh_source) == 1:
        # Spider that gets copied
        copy_spider = source[action_idx]
        # Spider that gets deleted
        through_spider = target[action_idx]
    else:
        copy_spider = target[action_idx]
        through_spider = source[action_idx]

    angle_copy = angles[copy_spider]
    # Add nodes on edges
    neighbours_trough = get_neighbours(through_spider, source, target)
    for neigh in neighbours_trough:
        if neigh != copy_spider:
            edge_idx = get_edge_idx(neigh, through_spider, source, target)
            colors, angles, source, target = insert_node_on_edge(
                colors, angles, source, target, edge_idx, colors[copy_spider], angle_copy)
    # Remove original nodes
    colors, angles, source, target = remove_node_with_edges(
        np.max([through_spider, copy_spider]), colors, angles, source, target)
    colors, angles, source, target = remove_node_with_edges(
        np.min([through_spider, copy_spider]), colors, angles, source, target)

    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
    
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def bialgebra_right(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Bialgebra right action"""
    idx_left = source[action_idx]
    color_left = colors[idx_left]
    neighbours_left = get_neighbours(idx_left, source, target)

    idx_right = target[action_idx]
    color_right = colors[idx_right]
    neighbours_right = get_neighbours(idx_right, source, target)

    # Add nodes of opposite color as neighbours to neighbours_left
    for neigh in neighbours_left:
        # Make sure not to insert node on selected edge:
        if neigh != idx_right:
            edge_idx = get_edge_idx(neigh, idx_left, source, target)
            colors, angles, source, target = insert_node_on_edge(
                colors, angles, source, target, edge_idx, color_right, ZERO)
    # Add nodes of opposite color as neighbours to neighbours_right
    for neigh in neighbours_right:
        if neigh != idx_left:
            edge_idx = get_edge_idx(neigh, idx_right, source, target)
            colors, angles, source, target = insert_node_on_edge(
                colors, angles, source, target, edge_idx, color_left, ZERO)
    # Add edges between new nodes:
    source, target = add_edge(
        len(colors)-1, len(colors)-3, source, target)
    source, target = add_edge(
        len(colors)-1, len(colors)-4, source, target)
    source, target = add_edge(
        len(colors)-2, len(colors)-3, source, target)
    source, target = add_edge(
        len(colors)-2, len(colors)-4, source, target)
    
    # Remove original nodes
    colors, angles, source, target = remove_node_with_edges(
        np.max([idx_left, idx_right]), colors, angles, source, target)
    colors, angles, source, target = remove_node_with_edges(
        np.min([idx_left, idx_right]), colors, angles, source, target)
    
    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
    
    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def bialgebra_left(colors, angles, selected_node, source, target, selected_edges, action_idx):
    """Bialgebra left action"""
    source_idx = source[action_idx]
    target_idx = target[action_idx]
    neighbours_source = get_neighbours(source_idx, source, target) 
    neighbours_target = get_neighbours(target_idx, source, target)
    # Get indices of nodes participating
    for neigh_s in neighbours_source:
        # Make sure its not the target node and different color then source node
        if np.all(colors[source_idx] + colors[neigh_s] == np.array(RED) + np.array(GREEN)) and neigh_s != target_idx:
                nn_source = get_neighbours(neigh_s, source, target)
                # If only three neighbours
                if len(nn_source) == 3:
                    for nn in nn_source:
                        if (np.all(colors[nn] + colors[neigh_s] == np.array(RED) + np.array(GREEN)) and 
                            nn != source_idx and 
                            nn in neighbours_target and
                            len(get_neighbours(neigh_s, source, target)) == 3 and
                            len(get_neighbours(nn, source, target)) == 3 
                            ):
                                part_neigh_source = neigh_s
                                part_neigh_target = nn
                                break
    # Get indices of input/output nodes:
    input_list = []
    for idx in neighbours_source:
        if idx != target_idx and idx != part_neigh_source:
            input_list.append(idx)
            break
    for idx in get_neighbours(part_neigh_target, source, target):
        if idx != target_idx and idx != part_neigh_source:
            input_list.append(idx)
            break

    assert len(input_list) ==2, "bug in bialgebra left"

    output_list = []
    for idx in neighbours_target:
        if idx != source_idx and idx != part_neigh_target:
            output_list.append(idx)
            break
    for idx in get_neighbours(part_neigh_source, source, target):
        if idx != source_idx and idx != part_neigh_target:
            output_list.append(idx)
            break

    assert len(output_list) ==2, "bug in bialgebra left 2"

    # Add new input node
    colors = np.row_stack((colors, colors[target_idx]))
    angles = np.row_stack((angles, ZERO))
    # Add new output node
    colors = np.row_stack((colors, colors[source_idx]))
    angles = np.row_stack((angles, ZERO))

    # Add edge between new input and output node
    source, target = add_edge(
                    len(colors)-1, len(colors)-2, source, target)
    # Add edges for input node
    for in_node in input_list:
        source, target = add_edge(
                in_node, len(colors)-2, source, target)
        
    # Add edges for output node
    for out_node in output_list:
        source, target = add_edge(
                out_node, len(colors)-1, source, target)
        
    # Remove old nodes:
    to_remove_list = np.array([target_idx, source_idx, part_neigh_source, part_neigh_target])
    to_remove_list[::-1].sort()
    for node_to_remove in to_remove_list:
        colors, angles, source, target = remove_node_with_edges(
            node_to_remove, colors, angles, source, target)

    # Remove unnescesarry edges
    colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)

    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))

def euler_rule(colors, angles, selected_node, source, target, selected_edges, action_idx):  
    """Euler action"""
    angle_mid = angles[action_idx]
    neigh_idcs = get_neighbours(action_idx, source, target)
    angle_neigh_1 = angles[neigh_idcs[0]]
    angle_neigh_2 = angles[neigh_idcs[1]]

    if np.all(colors[action_idx] == GREEN):
        color_mid = GREEN
        color_out = RED
    else:
        color_mid = RED
        color_out = GREEN

    # Invert colors
    colors[action_idx] = color_out
    colors[neigh_idcs[0]] = color_mid
    colors[neigh_idcs[1]] = color_mid

    # Change angles. Only pi/2, 3pi/2 and arbitrary allowed

    if (np.all(angle_mid == ARBITRARY) or 
        np.all(angle_neigh_1 == ARBITRARY) or 
        np.all(angle_neigh_2 == ARBITRARY)):
            angles[action_idx] = ARBITRARY
            angles[neigh_idcs[0]] = ARBITRARY
            angles[neigh_idcs[1]] = ARBITRARY
    # pi/2, pi/2, pi/2 do nothing
    # 3pi/2, pi/2, 3pi/2 do nothing
    elif np.all(angle_mid == PI_half):
        if np.all(angle_neigh_1 == PI_half):
            if np.all(angle_neigh_2 == PI_three_half):
                # pi/2, pi/2, 3pi/2
                angles[neigh_idcs[0]] = PI_three_half
                angles[neigh_idcs[1]] = PI_half
        elif np.all(angle_neigh_1 == PI_three_half):
            if np.all(angle_neigh_2 == PI_half):
                # 3pi/2, pi/2, pi/2
                angles[neigh_idcs[0]] = PI_half
                angles[neigh_idcs[1]] = PI_three_half
    elif np.all(angle_mid == PI_three_half):
        if np.all(angle_neigh_1 == PI_half):
            if np.all(angle_neigh_2 == PI_half):
                # pi/2, 3pi/2, pi/2
                angles[action_idx] = PI_half
                angles[neigh_idcs[0]] = PI_three_half
                angles[neigh_idcs[1]] = PI_three_half
            elif np.all(angle_neigh_2 == PI_three_half):
                # pi/2, 3pi/2, 3pi/2
                angles[action_idx] = PI_half
        elif np.all(angle_neigh_1 == PI_three_half):
            if np.all(angle_neigh_2 == PI_half):
                # 3pi/2, 3pi/2, pi/2
                angles[action_idx] = PI_half
    elif np.all(angle_mid == PI_three_half):
        if np.all(angle_neigh_1 == PI_three_half):
            if np.all(angle_neigh_2 == PI_three_half):
                # 3pi/2, 3pi/2, 3pi/2
                angles[action_idx] = PI_half
                angles[neigh_idcs[0]] = PI_half
                angles[neigh_idcs[1]] = PI_half               
            

    return (colors, angles, np.zeros(len(colors), dtype=np.int32), 
             source, target, np.zeros(len(target), dtype=np.int32))


# Helper functions--------------------------------------------------------------------------------------
def remove_disconnected_nodes(colors, angles, source, target):
    """Removes nodes that are not connected to inputs or outputs"""
    input_idcs = np.where(np.all(colors == INPUT, axis=1))
    output_idcs = np.where(np.all(colors == OUTPUT, axis=1))
    search_from_idcs = np.append(input_idcs, output_idcs)
    connected_nodes = []
    # Start searching from each input/ouput node
    for src_idx in search_from_idcs:
        # If not already reached this node:
        if src_idx not in connected_nodes:
            # Get all connected nodes
            conn_nodes = breadth_first_search(src_idx, source, target, len(colors))
            connected_nodes += conn_nodes
     # Flatten and sort     
    connected_nodes = np.array(connected_nodes, dtype=np.int32)
  
    all_nodes = np.arange(len(colors), dtype=np.int32)

    not_connected = all_nodes[np.in1d(all_nodes,connected_nodes,invert=True)]
    not_connected[::-1].sort()

    # Remove all unconnected
    for idx in not_connected:
        colors, angles, source, target = remove_node_with_edges(
            idx, colors, angles, source, target)
        
    return colors, angles, source, target

def breadth_first_search(source_idx, source, target, n_nodes)->list:
    """returns indices of all nodes connected to source_idx"""
    # save if node already visited
    visited = np.zeros(n_nodes, dtype=np.int32)
    visited[source_idx] = 1

    queue = deque()
    queue.append(source_idx)

    reachable_nodes = []

    while(len(queue) > 0):
        # Dequeue a vertex from queue
        u = queue.popleft()
 
        reachable_nodes.append(u)
 
        # Get all adjacent vertices of the dequeued
        # vertex u. If a adjacent has not been visited,
        # then mark it visited and enqueue it
        for itr in get_neighbours(u, source, target):
            if (visited[itr] == 0):
                visited[itr] = 1
                queue.append(itr)
 
    return reachable_nodes

def insert_node_on_edge(colors, angles, source, target, edge_idx, color, angle):
    """Insert node with color and angle on edge"""
    colors = np.row_stack((colors, color))
    angles = np.row_stack((angles, angle))
    idx_new_node = len(colors) - 1

    neighbors = (source[edge_idx], target[edge_idx])
    # add new edges
    for neigh in neighbors:
        source, target = add_edge(
            idx_new_node, neigh, source, target)
    # Remove original edge
    source, target = remove_edge(
        edge_idx, source, target)
    return colors, angles, source, target

def get_edge_idx(node_idx_1, node_idx_2, source, target):
    """Returns edge index between nodes, or None if edge doesn't exists"""
    edge_idx = np.where(np.logical_and(source==node_idx_1, target==node_idx_2))[0]
    if edge_idx.shape[0] == 0:
        edge_idx = np.where(np.logical_and(target==node_idx_1, source==node_idx_2))[0]
    return edge_idx
    


def remove_multiple_edges(
            list_mod_nodes, colors, angles, source, target):
    """Takes edges between two spiders with same color to only one edge and 
    edges between spiders with different colors mod 2"""
            
    hada_loop_idcs = []
    hada_loop_neigh_idcs = []
    for node_idx in list_mod_nodes:
        color = colors[node_idx]
        if list(color) in [GREEN, RED]:
            nodes_connected_idcs = get_neighbours(node_idx, source, target)
            # Count appearances
            unique, counts = np.unique(nodes_connected_idcs, return_counts=True)
            for neighbor_idx, counts in zip(unique, counts):
                if counts >= 2:
                    neighbor_color = colors[neighbor_idx]
                    if list(neighbor_color) in [GREEN, RED]:
                        edge_idcs = np.append(np.where(np.logical_and(source==node_idx, target==neighbor_idx)),
                                              np.where(np.logical_and(source==neighbor_idx, target==node_idx)))
                        edge_idcs[::-1].sort()
                        # If different color and mod 2 remove all
                        if not (np.all(color + neighbor_color == np.array(GREEN)+np.array(RED)) 
                            and len(edge_idcs)%2 == 0):
                            edge_idcs = edge_idcs[:-1]

                        for idx in edge_idcs:
                            source, target = remove_edge(
                                idx, source, target)
                    else:          
                        # Neighbour_index is a hadamard connected by two edges to current node
                        hada_loop_idcs.append(neighbor_idx)
                        hada_loop_neigh_idcs.append(node_idx)

                    
    # neighbour_index is a hadamard connected by two edges to current node
    # Remove hadamard if it was in a loop and add pi phase shift to neighbor
    # Eq 83 in ZX bible
    
    # Apply phase shift to neighbors
    for neighbor_idx in hada_loop_neigh_idcs:
        angles[neighbor_idx] = get_added_angle(angles[neighbor_idx], np.array(PI))
    # Remove Hadamard with edges
    hada_loop_idcs = np.array(hada_loop_idcs)
    hada_loop_idcs[::-1].sort()
    for hada_idx in hada_loop_idcs:
        colors, angles, source, target = remove_node_with_edges(hada_idx, colors, angles, source, target)
    return colors, angles, source, target                    

def remove_edge(edge_idx, source, target):
    """removes edge at given index"""
    source = np.delete(source, edge_idx, axis=0)
    target = np.delete(target, edge_idx, axis=0)
    return source, target
       

def remove_node_with_edges(node_idx, colors, angles, source, target):
    """Removes node at node_idx and all connected edges"""
    # Remove node color
    colors = np.delete(colors, node_idx, axis=0)
    # Remove node angle
    angles = np.delete(angles, node_idx, axis=0)

    # Important: This need to be done before the edge indices are decreased
    connected_edge_indices = np.append(np.where(source==node_idx), np.where(target==node_idx))
    # Sort in descending order
    connected_edge_indices[::-1].sort()

    for edge_idx in connected_edge_indices:
        source, target = remove_edge(
            edge_idx, source, target)

    # Decrease edge indcs over deleted node
    source[source > node_idx] = source[source > node_idx] - 1
    target[target > node_idx] = target[target > node_idx] - 1

    
    return colors, angles, source, target


def add_edge(source_idx, target_idx, source, target):
    """Adds edge between source_idx and target_idx nodes"""
    source = np.append(source, source_idx)
    target = np.append(target, target_idx)
    return source, target


def get_neighbours(index, source, target):
    """returns indices of all nodes connected to index"""
    idcs = target[np.where(source==index)[0]]
    idcs = np.append(idcs, source[np.where(target==index)[0]])
    return idcs

     
def get_added_angle(angle1:np.ndarray, angle2:np.ndarray)-> np.ndarray:
    """Takes one hot encoded angles and adds them up:
    0: 0pi
    1: pi/2
    2: Pi
    3: 3pi/2
    4: arbitrary
    5: no angle"""

    angle_idx1 = np.argmax(angle1)
    angle_idx2 = np.argmax(angle2)

    if angle_idx1 == 4 or angle_idx2 == 4:
        idx = 4
    elif angle_idx1 < 4 and angle_idx2 < 4:
        idx = (angle_idx1 + angle_idx2) % 4
    else:
        raise Exception(f"Adding angles {angle_idx1}, {angle_idx2}")
    return np.eye(len(ZERO), dtype=np.int32)[idx]       
