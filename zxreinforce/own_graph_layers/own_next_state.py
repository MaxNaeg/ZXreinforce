# This file holds slightly modified versions of the original tensorflow_gnn files to enable
# undirected message passing

from typing import Tuple

import tensorflow as tf

from tensorflow_gnn.graph import graph_constants as const

@tf.keras.utils.register_keras_serializable(package="GNN")
class ListNextStateFromConcat(tf.keras.layers.Layer):
  """Computes a new state by summing up messages from edges and concatening result with node and context feature  
    and applying a user defined Keras Layer. Should be used together with own_graph_update.ListNodeSetUpdate

  Init args:
    transformation: Required. A Keras Layer to transform the combined inputs
      into the new state.

  Call returns:
    The result of transformation.
  """

  def __init__(self,
               transformation: tf.keras.layers.Layer,
               **kwargs):
    super().__init__(**kwargs)
    self._transformation = transformation

  def get_config(self):
    return dict(transformation=self._transformation,
                **super().get_config())

  def call(
      self, inputs: Tuple[
          const.FieldOrFields, const.FieldsNest, const.FieldsNest
      ]) -> const.FieldOrFields:
    summed = tf.keras.layers.Add()(inputs[1])
    flatened = tf.nest.flatten([inputs[0], summed, inputs[2]])
    net = tf.concat(flatened, axis=-1)
    net = self._transformation(net)
    return net