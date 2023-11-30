# This file contains fast functions to batch observations from the ZX environment into a single GraphTensor
# and to batch a list of masks into a single padded mask.

import numpy as np
import numba
import tensorflow as tf
from tensorflow_gnn.graph import adjacency as adj
from tensorflow_gnn.graph.graph_tensor import GraphTensor, NodeSet, EdgeSet, Context



# We need two batch functions to reduce retracing 
# (one for sample trajectory batch size and one for training batch size)
def batch_obs_combined_traj(observation: list) -> GraphTensor:
    """observation: List of ZX env observations,

    returns: GraphTensor,

    Use this function during trajectory sampling,
    Batches the individual observations into a single batched GraphTensor 
    compatibale with tensorflow_gnn"""
    # This function reorders the values of the individual observations. Fastyer in numpy than in tf
    reordered = xbatch_merge_list_of_observations_context_part1(observation)
    # Now batch the reordered values in a tf.fucntion
    return xbatch_merge_list_of_observations_context_part2(*reordered)

def batch_obs_combined_train(observation: list)-> GraphTensor:
    """observation: List of ZX env observations,

    returns: GraphTensor,

    Use this function during training,
    Batches the individual observations into a single batched GraphTensor 
    compatibale eiyh tensorflow_gnn"""
    reordered = xbatch_merge_list_of_observations_context_part1(observation)
    return xbatch_merge_list_of_observations_context_part2_train(*reordered)



def xbatch_merge_list_of_observations_context_part1(observations: list) -> tuple:
    """observation: List of ZX env observations,

    returns: tuple (colors, angles, selected_node, context, 
    selected_edges, source, target, node_siozes, edge_sizes, to_add),

    Reorders the values of the individual observations as a preparation to create a
    batched GraphTensor. Faster in numpy than in tf"""
   
    color_list = []
    angle_list = []
    selected_node_list = []
    source_list = []
    target_list = []
    selected_edges_list = []
    node_sizes = []
    edge_sizes = []
    context_list = []

    for obs in observations:
        color_list.append(obs[0])
        angle_list.append(obs[1])
        selected_node_list.append(obs[2])
        source_list.append(obs[3])
        target_list.append(obs[4])
        selected_edges_list.append(obs[5])
        node_sizes.append(obs[6])
        edge_sizes.append(obs[7])
        context_list.append(obs[8])

    # Nodes will be indexed continuosly in batched graph
    # We need to adjust the edge indices accordingly
    cum_node_sizes = np.cumsum([0] + node_sizes[:-1])
    to_add = np.repeat(cum_node_sizes, edge_sizes)


    return (tf.constant(np.concatenate(color_list)), 
            tf.constant(np.concatenate(angle_list)), 
            tf.constant(np.concatenate(selected_node_list)),
            tf.constant(np.stack(context_list), dtype=tf.float32), 
            tf.constant(np.concatenate(selected_edges_list)), 
            tf.constant(np.concatenate(source_list)), 
            tf.constant(np.concatenate(target_list)),
            tf.constant(node_sizes), 
            tf.constant(edge_sizes), 
            tf.constant(to_add, dtype=tf.int32))

@tf.function(reduce_retracing=True)
def xbatch_merge_list_of_observations_context_part2(color_list: tf.constant, angle_list: tf.constant, 
                                                    selected_node_list: tf.constant, context_list: tf.constant,
                                                   selected_edges_list: tf.constant, source_list: tf.constant, 
                                                   target_list: tf.constant, node_sizes: tf.constant, 
                                                   edge_sizes: tf.constant, to_add: tf.constant)-> GraphTensor:
    
    """Input: Graph features as tf.constants,

    returns: GraphTensor,
    
    Batches the Graph features into one GraphTensor"""
    
    selected_node_list = tf.expand_dims(selected_node_list,-1)

    source_list = source_list + to_add
    target_list = target_list + to_add

    selected_edges_list = tf.expand_dims(selected_edges_list,-1)

    return GraphTensor.from_pieces(
        node_sets={
            "spiders":
                NodeSet.from_fields(
                    sizes=node_sizes,
                    features={"color": color_list,
                              "angle": angle_list,
                              "selected_node": selected_node_list},
                )
        },
        edge_sets={
            "edges":
                EdgeSet.from_fields(
                    sizes=edge_sizes,
                    adjacency=adj.Adjacency.from_indices(
                        source=("spiders", source_list),
                        target=("spiders", target_list),
                    ),
                    features={"selected_edges": selected_edges_list},
                )
        },
        context=Context.from_fields(
            features={"context_feat": context_list},
            sizes=tf.ones((len(node_sizes),1)),
            shape=(len(node_sizes),)
        ),
    )

# Same as xbatch_merge_list_of_observations_context_part2 but needed to reduce retracing
@tf.function(reduce_retracing=True)
def xbatch_merge_list_of_observations_context_part2_train(color_list: tf.constant, angle_list: tf.constant, 
                                                    selected_node_list: tf.constant, context_list: tf.constant,
                                                   selected_edges_list: tf.constant, source_list: tf.constant, 
                                                   target_list: tf.constant, node_sizes: tf.constant, 
                                                   edge_sizes: tf.constant, to_add: tf.constant)-> GraphTensor:
    
    """Input: Graph features as tf.constants,

    returns: GraphTensor,
    
    Batches the Graph features into one GraphTensor"""
    
    selected_node_list = tf.expand_dims(selected_node_list,-1)

    source_list = source_list + to_add
    target_list = target_list + to_add

    selected_edges_list = tf.expand_dims(selected_edges_list,-1)

    return GraphTensor.from_pieces(
        node_sets={
            "spiders":
                NodeSet.from_fields(
                    sizes=node_sizes,
                    features={"color": color_list,
                              "angle": angle_list,
                              "selected_node": selected_node_list},
                )
        },
        edge_sets={
            "edges":
                EdgeSet.from_fields(
                    sizes=edge_sizes,
                    adjacency=adj.Adjacency.from_indices(
                        source=("spiders", source_list),
                        target=("spiders", target_list),
                    ),
                    features={"selected_edges": selected_edges_list},
                )
        },
        context=Context.from_fields(
            features={"context_feat": context_list},
            sizes=tf.ones((len(node_sizes),1)),
            shape=(len(node_sizes),)
        ),
    )

def batch_mask_combined(mask_list:list[np.ndarray]) -> tf.constant:
    """mask_list: list of boolean np arrays,
    
    returns tf. constant,
    
    Batches the individual masks into one by padding them all to the same size"""
    return tf.constant(batch_mask_np_jit(mask_list))

@numba.njit
def batch_mask_np_jit(mask_list:list)-> np.ndarray:
    """Same as batch_mask_combined, but without tf to make numba compatible"""
    # Find max len to pad to
    len_list = np.array([len(mask) for mask in mask_list])
    max_len = np.max(len_list)
    mask_padded = np.full((len(mask_list), max_len), False)
    # Pad each mask with False values
    for i, mask in enumerate(mask_list):
        pad_list = np.full(max_len - len(mask), False)
        mask_padded[i] = np.append(mask, pad_list)
    return mask_padded





