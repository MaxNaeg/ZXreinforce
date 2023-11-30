# This script uses the data save by action_sampler.py to evaluate the dependence of actions
# on their local environment. It does so by building the ZX-diagram in layers and evaluating
# the logits of the actions at each layer. The results are saved in dist_lay/
import sys
sys.path.append("../../")
sys.path.append("../../../")

import pickle
import numpy as np
import tensorflow_gnn as tfgnn
import tensorflow_gnn.proto.graph_schema_pb2 as schema_pb2
from pathlib import Path
from google.protobuf import text_format


import zxreinforce.own_constants as oc
from zxreinforce import interpret_utils
from zxreinforce.RL_Models import build_gnn_actor_model
from zxreinforce.rl_schemas import OBSERVATION_SCHEMA_ZX_final
from zxreinforce.action_conversion_utils import get_action_target, get_action_name


# Total amount of action types
idcs_act = np.arange(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS)
# Amount of actions sampled per type
n_samples = 1000

# Load data saved by action_sampler.py
obs_lists = []
act_lists = []
rew_lists = []
mask_lists = []
for idx_act in idcs_act:
    with open(f"sampled/obs_list_{idx_act}_{n_samples}", 'rb') as f:
        obs_lists.append(pickle.load(f))
    with open(f"sampled/reward_list_{idx_act}_{n_samples}", 'rb') as f:
        rew_lists.append(pickle.load(f))
    with open(f"sampled/act_idcs_list{idx_act}_{n_samples}", 'rb') as f:
        act_lists.append(pickle.load(f))
    with open(f"sampled/mask_list{idx_act}_{n_samples}", 'rb') as f:
        mask_lists.append(pickle.load(f))

# Load actor model
agent_path = Path("../../saved_agents/all_1/saved_agent")
index = 400

graph_schema = text_format.Merge(OBSERVATION_SCHEMA_ZX_final, schema_pb2.GraphSchema())
graph_tensor_spec = tfgnn.create_graph_spec_from_schema_pb(graph_schema)

actor_model = build_gnn_actor_model(graph_tensor_spec=graph_tensor_spec)
actor_model.load_weights(str(agent_path / f"actor{index}.keras"), by_name=True)


# Build observation in layers
n_layers = 6
res_lists_dist = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]
res_lists_logs = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]
res_lists_logs_all = [ [] for _ in range(oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS) ]

# Add to saved results
add="_1000_"
# Loop over the action types
for i, (actl, obsl, rewl, maskl) in enumerate(zip(act_lists, obs_lists, rew_lists, mask_lists)):
    print(f"Eval action {i}: {get_action_name(i)}")
    # Node or edeg action?
    if i < oc.N_NODE_ACTIONS:
        type="node"
    else:
        type="edge"
    # Go over all actions sampled for specific action type
    for act, obs, rew, mask in zip(actl, obsl, rewl, maskl):
        # Get target node/edge index
        index_acting = get_action_target(act, obs[-3], obs[-2])
        # Build diagram in layers and evaluate logits
        (obs_red_lay, diff_list_lay, log_list_lay, logs_all_masked_lay, 
        red_node_idcs_lay, red_edge_idcs_lay
        ) = interpret_utils.add_layers_graph(obs, actor_model, type, index_acting , mask, n_dist=n_layers) 
        res_lists_dist[i].append(diff_list_lay)
        res_lists_logs[i].append(log_list_lay)
        res_lists_logs_all[i].append(logs_all_masked_lay)

with open(f"dist_lay/res_lists_dist_{add}", 'wb') as f:
    pickle.dump(res_lists_dist, f)
with open(f"dist_lay/res_lists_logs{add}", 'wb') as f:
    pickle.dump(res_lists_logs, f)
with open(f"dist_lay/res_lists_logs_all{add}", 'wb') as f:
    pickle.dump(res_lists_logs_all, f)