# This file holds slightly modified versions of the original tensorflow_gnn files to enable
# undirected message passing
from typing import Optional
import tensorflow as tf
from tensorflow_gnn.graph import graph_tensor_ops as broadcast_ops
from tensorflow_gnn.graph import dict_utils as du
from tensorflow_gnn.graph import graph_constants as const
from tensorflow_gnn.graph import graph_tensor as gt
from tensorflow_gnn.keras.layers import next_state as next_state_lib
from tensorflow_gnn.keras.layers.graph_update import EdgesToNodePoolingLayer, _check_is_layer, _copy_if_sequence, _get_feature_or_features

@tf.keras.utils.register_keras_serializable(package="GNN")
class ListNodeSetUpdate(tf.keras.layers.Layer):

  def __init__(self,
               edge_set_inputs: list[tuple[const.EdgeSetName, EdgesToNodePoolingLayer]],
               next_state: next_state_lib.NextStateForNodeSet,
               *,
               node_input_feature: Optional[const.FieldNameOrNames]
               = const.HIDDEN_STATE,
               context_input_feature: Optional[const.FieldNameOrNames] = None,
               **kwargs):
    super().__init__(**kwargs)
    self._edge_set_inputs = [
        (key, _check_is_layer (value,
                             f"NodeSetUpdate(edge_set_inputs={{{key}: ...}}"))
        for key, value in edge_set_inputs]
    self._next_state = _check_is_layer(next_state,
                                       "NodeSetUpdate(next_state=...")
    self._node_input_feature = _copy_if_sequence(node_input_feature)
    self._context_input_feature = _copy_if_sequence(context_input_feature)

  def get_config(self):
    return dict(
        # Sublayers need to be top-level objects in the config (b/209560043).
        **du.with_key_prefix(self._edge_set_inputs, "edge_set_inputs/"),
        next_state=self._next_state,
        node_input_feature=self._node_input_feature,
        context_input_feature=self._context_input_feature,
        **super().get_config())

  @classmethod
  def from_config(cls, config):
    config["edge_set_inputs"] = du.pop_by_prefix(config, "edge_set_inputs/")
    return cls(**config)

  def call(self, graph: gt.GraphTensor,
           node_set_name: const.NodeSetName) -> gt.GraphTensor:
    gt.check_scalar_graph_tensor(graph, "NodeSetUpdate")

    next_state_inputs = []
    # Input from the nodes themselves.
    next_state_inputs.append(
        _get_feature_or_features(graph.node_sets[node_set_name],
                                 self._node_input_feature))
    # Input from edge sets.
    input_from_edge_sets = []
    for edge_set_name, input_fn in self._edge_set_inputs:
      input_from_edge_sets.append(input_fn(
          graph, edge_set_name=edge_set_name))
    next_state_inputs.append(input_from_edge_sets)
    # Input from context.
    next_state_inputs.append(tf.nest.map_structure(
        lambda value: broadcast_ops.broadcast_context_to_nodes(  # pylint: disable=g-long-lambda
            graph, node_set_name, feature_value=value),
        _get_feature_or_features(graph.context, self._context_input_feature)))

    next_state_inputs = tuple(next_state_inputs)
    assert len(next_state_inputs) == 3, "Internal error"
    return self._next_state(next_state_inputs)