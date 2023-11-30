# This file contains functions to build up a diagram in layers around a node/edge.

import tensorflow as tf
import numpy as np
import copy
from keras import Model

from . import own_constants as oc
from .batch_utils import batch_obs_combined_traj


def build_small_obs(obs, rel_node_idcs, rel_edge_idcs):
    colors, angles, selected_node, source, target, selected_edges, n_nodes, n_edges, context = obs
    adapted_source = np.array([rel_node_idcs.index(node) for node in source[rel_edge_idcs]], dtype=np.int32)
    adapted_target = np.array([rel_node_idcs.index(node) for node in target[rel_edge_idcs]], dtype=np.int32)
    return (colors[rel_node_idcs], angles[rel_node_idcs], selected_node[rel_node_idcs],
            adapted_source, adapted_target, selected_edges[rel_edge_idcs],
            np.array(len(rel_node_idcs), dtype=np.int32), np.array(len(rel_edge_idcs), dtype=np.int32), context)


def add_layers_graph(obs:tuple, actor_model:Model, node_edge:str, 
                     idx_act:int, mask:np.ndarray, n_dist:int=8)->tuple:
    """obs: observation of the ZX environment,
    actor_model: gnn model,
    node_edge: "node" or "edge", build graph around nodie or edge
    idx_act: index of the node/edge to add layers around,
    mask: mask of the ZX environment,
    n_dist: number of layers to add,
    returns: tuple of (obs, diff_list, log_list, logs_all_masked, rel_node_idcs, rel_edge_idcs),
    
    Builds subgraph around the node/edge specified by idx_act in obs with n_dist layers.
    Computes logits for the node/edge in the original graph and for the subgraph of each layer
    """
    rel_action_index = 0
    colors, angles, selected_node, source, target, selected_edges, n_nodes, n_edges, context = obs
    
    if node_edge == "node":
        mask_small = mask[oc.N_NODE_ACTIONS*idx_act : oc.N_NODE_ACTIONS*(idx_act+1)]
        rel_node_idcs = [idx_act]
        rel_edge_idcs = []
        logs_idx=0
    elif node_edge == "edge":
        mask_small = mask[oc.N_NODE_ACTIONS*n_nodes +  oc.N_EDGE_ACTIONS*idx_act: oc.N_NODE_ACTIONS*n_nodes +  oc.N_EDGE_ACTIONS*(idx_act+1)]
        rel_node_idcs = [source[idx_act], target[idx_act]]
        rel_edge_idcs = [idx_act]
        logs_idx=1

    # Get logits in original graph
    logs_all = actor_model(batch_obs_combined_traj([obs]))[logs_idx][idx_act]
    logs_all_masked = tf.where(mask_small, logs_all, logs_all.dtype.min)

    # Build subgraph of layer 0
    log_list = []
    small_obs = build_small_obs(obs, rel_node_idcs, rel_edge_idcs)
    logs_small = actor_model(batch_obs_combined_traj([small_obs]))[logs_idx]
    logs_small_masked = tf.where(mask_small, logs_small, logs_small.dtype.min)
    log_list.append(logs_small_masked)

    # Save difference between logits of original graph and subgraph of layer 0
    diff_list = []
    diff = tf.reduce_sum(tf.abs(logs_small_masked - logs_all_masked))
    diff_list.append(diff)
    
    # Repeat for each layer
    for i in range(n_dist):
        
        non_rel_edge_idcs = [i for i in range(n_edges) if i not in rel_edge_idcs]
        rel_node_idcs_layer = copy.copy(rel_node_idcs)
        # Add new nodes with connection
        for edge_idx in non_rel_edge_idcs:
            source_idx, target_idx = source[edge_idx], target[edge_idx]
            if source_idx in rel_node_idcs_layer or target_idx in rel_node_idcs_layer:
                rel_edge_idcs.append(edge_idx)
                if source_idx not in rel_node_idcs:
                    rel_node_idcs.append(source_idx)
                if target_idx not in rel_node_idcs:
                    rel_node_idcs.append(target_idx)
        # Add inner layer edges
        non_rel_edge_idcs = [i for i in range(n_edges) if i not in rel_edge_idcs]
        for edge_idx in non_rel_edge_idcs:
            source_idx, target_idx = source[edge_idx], target[edge_idx]
            if source_idx in rel_node_idcs_layer and target_idx in rel_node_idcs_layer:
                rel_edge_idcs.append(edge_idx)

        # Build subgraph and get logits
        small_obs_try = build_small_obs(obs, rel_node_idcs, rel_edge_idcs)
        logs_small = actor_model(batch_obs_combined_traj([small_obs_try]))[logs_idx][rel_action_index]
        logs_small_masked = tf.where(mask_small, logs_small, logs_small.dtype.min)
        log_list.append(logs_small_masked)

        diff_try = tf.reduce_sum(tf.square(logs_small_masked - logs_all_masked))
        diff_list.append(diff_try)

    return build_small_obs(obs, rel_node_idcs, rel_edge_idcs), diff_list, log_list, logs_all_masked, rel_node_idcs, rel_edge_idcs