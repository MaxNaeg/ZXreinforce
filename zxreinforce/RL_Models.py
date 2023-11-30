import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn

from keras.initializers import Initializer
from tensorflow_gnn.graph.graph_tensor import GraphTensorSpec

from . import own_constants as own_constants
from .own_graph_layers.own_graph_update import ListNodeSetUpdate
from .own_graph_layers.own_next_state import ListNextStateFromConcat

def dense(units, l2_regularization:float, dropout_rate:float, activation:str, 
          kernel_initializer:Initializer, bias_initializer:Initializer):
    """A Dense layer with regularization (L2 and Dropout)."""
    regularizer = tf.keras.regularizers.l2(l2_regularization)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            kernel_initializer=kernel_initializer, 
            bias_initializer=bias_initializer),
        tf.keras.layers.Dropout(dropout_rate)
    ])



fac = 1
def build_gnn_actor_model(
    graph_tensor_spec:GraphTensorSpec,
    # Dimensions for message passing.
    message_dim:int=128*fac,
    next_node_state_dim:int=128*fac,
    next_edge_state_dim:int=128*fac,
    # Dimension of hidden layer in action selection
    hidden_action_dim:int=128*fac,
    # Number of message passing steps.
    num_message_passing:int=6,
    # Hyperparameters for dense layers.
    l2_regularization:float=0.,
    dropout_rate:float=0.,
    activation:str="tanh",
    kernel_ini_hidden:Initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=0),
    kernel_ini_out:Initializer=tf.keras.initializers.Orthogonal(gain=0.01, seed=0),
    bias_ini_hidden:str="zeros",
    bias_ini_out:str="zeros",
)->tf.keras.Model:
    """graph_tensor_spec: GraphTensorSpec describing the input graph.
    message_dim: Dimension of message computing networks,
    next_node_state_dim: Dimension of node state computing networks,
    next_edge_state_dim: Dimension of edge state computing networks,
    hidden_action_dim: Dimension of hidden layers in action selection,
    num_message_passing: Number of message passing layers,
    l2_regularization: L2 regularization strength,
    dropout_rate: Dropout rate,
    activation: Activation function of nn layers,
    kernel_ini_hidden: Initializer for hidden layers,
    kernel_ini_out: Initializer for output layers,
    bias_ini_hidden: Initializer for bias of hidden layers,
    bias_ini_out: Initializer for bias of output layers,
    return: Keras Model that takes a GraphTensor and returns logits for node, edge and stop actions.
    """
    # Input Object for GNNS
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

    # This merges all graphs in the batch to a single graph with shape [elements per batch]
    graph = input_graph.merge_batch_to_components()


    # Just concatenate initial features
    def set_initial_node_state(node_set, *, node_set_name):
        features = node_set.features
        return tf.cast(tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())]), 
                       dtype=tf.float32) # now feauture is tfgnn.HIDDEN_STATE
    def set_initial_edge_state(edge_set, *, edge_set_name):
        features = edge_set.features
        return tf.cast(tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())]), 
                       dtype=tf.float32) # now feauture is tfgnn.HIDDEN_STATE
    
    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
            graph)
    
    # Do message pasing between nodes <-> nodes and edges (from neighboring nodes) 
    for _ in range(num_message_passing):
        # Define here to use the same network for message passing in both directions
        message_fn = dense(message_dim, l2_regularization, dropout_rate, 
                           activation, kernel_ini_hidden, bias_ini_hidden)
        # This layer computes the new node state from target node and message
        new_node_fn = dense(next_node_state_dim, l2_regularization, dropout_rate, 
                            activation, kernel_ini_hidden, bias_ini_hidden)
        # This layer computes the new edge state form edge state and neighboring node states
        new_edge_fn = dense(next_edge_state_dim, l2_regularization, dropout_rate, 
                            activation, kernel_ini_hidden, bias_ini_hidden)   


        # Updates the whole graph
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "spiders": ListNodeSetUpdate(
                    [("edges", tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn,
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET,
                        sender_edge_feature=tfgnn.HIDDEN_STATE),),
                    ("edges", tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn,
                        reduce_type="sum",
                        receiver_tag=tfgnn.SOURCE,
                        sender_edge_feature=tfgnn.HIDDEN_STATE),)
                    ],
                    ListNextStateFromConcat(new_node_fn))},
            edge_sets = {'edges': tfgnn.keras.layers.EdgeSetUpdate(
                next_state = tfgnn.keras.layers.NextStateFromConcat(
                    new_edge_fn))},
        )(graph)

    # Stop counter
    count_down = tf.gather(graph.context.features["context_feat"], [0,], axis=1)

    # List holding logits of node, edge and stop action
    logit_list = []
    # Unnormalized logits for node actions
    # Add countdown to selection
    count_down_nodes = tf.repeat(count_down, graph.node_sets['spiders'].sizes, axis=0)
    logits_node_act = tf.keras.layers.Concatenate(axis=1)([graph.node_sets['spiders'][tfgnn.HIDDEN_STATE], 
                                                           count_down_nodes])
    logits_node_act = dense(hidden_action_dim, l2_regularization, dropout_rate, activation, 
                            kernel_ini_hidden, bias_ini_hidden) (
        logits_node_act)
    logits_node_act = dense(own_constants.N_NODE_ACTIONS, l2_regularization, 0, 'linear', 
                            kernel_ini_out, bias_ini_out)(logits_node_act)
    logits_node_act = tf.squeeze(logits_node_act)
    logit_list.append(logits_node_act)

    # Unnormalized logits for edge actions
    # Add countdown to selection
    count_down_edges = tf.repeat(count_down, graph.edge_sets['edges'].sizes, axis=0)

    logits_edge_act = tf.keras.layers.Concatenate(axis=1)([graph.edge_sets['edges'][tfgnn.HIDDEN_STATE], 
                                                           count_down_edges])
    logits_edge_act = dense(hidden_action_dim, l2_regularization, dropout_rate, activation, 
                            kernel_ini_hidden, bias_ini_hidden) (
        logits_edge_act)
    logits_edge_act = dense(own_constants.N_EDGE_ACTIONS, l2_regularization, 0, 'linear', 
                            kernel_ini_out, bias_ini_out)(logits_edge_act)
    logits_edge_act = tf.squeeze(logits_edge_act)
    logit_list.append(logits_edge_act)


    # Pool all node features to the context, aggregate by mean
    pooled_features_nodes = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="spiders")(graph)
    
    # Pool all edge features to the context, aggregate by mean
    pooled_features_edges = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", edge_set_name="edges")(graph)
    
    node_context_features = tf.keras.layers.Concatenate(axis=1)([graph.context.features["context_feat"],
                                                                  pooled_features_nodes, 
                                                                  pooled_features_edges])

    logits_stop_act = dense(hidden_action_dim, l2_regularization, dropout_rate, activation, 
                            kernel_ini_hidden, bias_ini_hidden)(
        node_context_features)
    logits_stop_act = dense(hidden_action_dim, l2_regularization, dropout_rate, activation, 
                            kernel_ini_hidden, bias_ini_hidden)(
        logits_stop_act)
    logits_stop_act = dense(1, l2_regularization, 0, 'linear', kernel_ini_out, bias_ini_out)(logits_stop_act)
    logits_stop_act = tf.squeeze(logits_stop_act)
    logit_list.append(logits_stop_act)

    # Build a Keras Model for the transformation from input_graph to logits.
    return tf.keras.Model(inputs=[input_graph], outputs=logit_list)



def build_gnn_critic_model(
    graph_tensor_spec:GraphTensorSpec,
    # Dimensions for message passing.
    message_dim:int=128*fac,
    next_node_state_dim:int=128*fac,
    next_edge_state_dim:int=128*fac,
    # Size of readout layer
    readout_dim:int=128*fac,
    # Number of message passing steps.
    num_message_passing:int=6,
    # Hyperparameters for dense layers.
    l2_regularization:float=0.,
    dropout_rate:float=0.,
    activation:str="tanh",
    kernel_ini_hidden:Initializer=tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=0),
    kernel_ini_out:Initializer=tf.keras.initializers.Orthogonal(gain=1, seed=0),
    bias_ini_hidden:str="zeros",
    bias_ini_out:str="zeros",
)->tf.keras.Model:
    """graph_tensor_spec: GraphTensorSpec describing the input graph.
    message_dim: Dimension of message computing networks,
    next_node_state_dim: Dimension of node state computing networks,
    next_edge_state_dim: Dimension of edge state computing networks,
    readout_dim: Dimension of hidden layers in readout,
    num_message_passing: Number of message passing layers,
    l2_regularization: L2 regularization strength,
    dropout_rate: Dropout rate,
    activation: Activation function of nn layers,
    kernel_ini_hidden: Initializer for hidden layers,
    kernel_ini_out: Initializer for output layers,
    bias_ini_hidden: Initializer for bias of hidden layers,
    bias_ini_out: Initializer for bias of output layers,
    return: Keras Model that takes a GraphTensor and returns value function."""
    # Input Object for GNNS
    input_graph = tf.keras.layers.Input(type_spec=graph_tensor_spec)

    # This merges all graphs in the batch to a single graph with shape [elements per batch]
    graph = input_graph.merge_batch_to_components()

    # Just concatenate initial embedding
    def set_initial_node_state(node_set, *, node_set_name):
        features = node_set.features
        return tf.cast(tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())]), dtype=tf.float32) # now feauture is tfgnn.HIDDEN_STATE
    def set_initial_edge_state(edge_set, *, edge_set_name):
        features = edge_set.features
        return tf.cast(tf.keras.layers.Concatenate()([v for _, v in sorted(features.items())]), dtype=tf.float32) # now feauture is tfgnn.HIDDEN_STATE
    

    graph = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_initial_node_state, edge_sets_fn=set_initial_edge_state)(
            graph)
    
    # Do message pasing between nodes <-> nodes and edges (from neighboring nodes) 
    for _ in range(num_message_passing-1):
        # Define here to use the same network for message passing in both directions
        message_fn = dense(message_dim, l2_regularization, dropout_rate, 
                           activation, kernel_ini_hidden, bias_ini_hidden)
        # This layer computes the new node state from target node and message
        new_node_fn = dense(next_node_state_dim, l2_regularization, dropout_rate, 
                            activation, kernel_ini_hidden, bias_ini_hidden)
        # This layer computes the new edge state form edge state and neighboring node states
        new_edge_fn = dense(next_edge_state_dim, l2_regularization, dropout_rate, 
                            activation, kernel_ini_hidden, bias_ini_hidden)   

        # Updates the whole graph
        graph = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "spiders": ListNodeSetUpdate(
                    [("edges", tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn,
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET,
                        sender_edge_feature=tfgnn.HIDDEN_STATE),),
                    ("edges", tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn,
                        reduce_type="sum",
                        receiver_tag=tfgnn.SOURCE,
                        sender_edge_feature=tfgnn.HIDDEN_STATE),)
                    ],
                    ListNextStateFromConcat(new_node_fn))},
            edge_sets = {'edges': tfgnn.keras.layers.EdgeSetUpdate(
                next_state = tfgnn.keras.layers.NextStateFromConcat(
                    new_edge_fn))},
        )(graph)


    
    # Pool all node features to the context, aggregate by mean
    pooled_features_nodes = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="spiders")(graph)
    
    # Pool all edge features to the context, aggregate by mean
    pooled_features_edges = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", edge_set_name="edges")(graph)
    
    node_context_features = tf.keras.layers.Concatenate(axis=1)([graph.context.features["context_feat"], 
                                                                 pooled_features_nodes, 
                                                                 pooled_features_edges])
    
    # Hidden layer for readout,
    value = dense(readout_dim, l2_regularization, dropout_rate, activation, 
                  kernel_ini_hidden, bias_ini_hidden)(node_context_features)
    # Hidden layer for readout,
    value = dense(readout_dim, l2_regularization, dropout_rate, activation, 
                  kernel_ini_hidden, bias_ini_hidden)(node_context_features)
    # Classify state value based on this information
    value = dense(1, l2_regularization, 0, "linear", 
                  kernel_ini_out, bias_ini_out)(value)
    value = tf.squeeze(value)
    # Build a Keras Model for the transformation from input_graph to logits.
    return tf.keras.Model(inputs=[input_graph], outputs=[value])