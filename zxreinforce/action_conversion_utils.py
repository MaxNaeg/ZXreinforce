from . import own_constants as oc

node_actions_fn = ["start_unmerge_rule", "stop_unmerge_rule",
                        "color_change", "split_hadamard", "merge_hadamard", "euler_rule"]

edge_actions_fn = ["merge_rule", "color_edge",
                        "pi_rule", "copy_rule_right", "bialgebra_right", "bialgebra_left"]
action_list = node_actions_fn + edge_actions_fn + ["stop_action"]

def get_action_type_idx(n_nodes: int, n_edges: int, action: int) -> int:
    """returns the index specifying the action type of action"""
    if action < oc.N_NODE_ACTIONS * n_nodes:
        action_fn_idx = action % oc.N_NODE_ACTIONS
    elif action < oc.N_NODE_ACTIONS * n_nodes + oc.N_EDGE_ACTIONS * n_edges:
        action = action - (oc.N_NODE_ACTIONS * n_nodes)
        action_fn_idx = action % oc.N_EDGE_ACTIONS + oc.N_NODE_ACTIONS
    else:
        action_fn_idx = oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS 
    return action_fn_idx

def get_action_name(ac_idx: int) -> str:
    """returns the action name given the action type index"""
    return action_list[ac_idx]


def get_action_index(n_nodes: int, n_edges: int, action_fn: str, act_taget: int) -> int:
    """returns the original action index given the action type and action target (which node/edge)"""
    if action_fn in node_actions_fn:
        assert act_taget < n_nodes
        return act_taget * oc.N_NODE_ACTIONS + node_actions_fn.index(action_fn)
    elif action_fn in edge_actions_fn:
        assert act_taget < n_edges
        return oc.N_NODE_ACTIONS * n_nodes + act_taget * oc.N_EDGE_ACTIONS + edge_actions_fn.index(action_fn)
    elif action_fn == "stop_action":
        return oc.N_NODE_ACTIONS * n_nodes + oc.N_EDGE_ACTIONS * n_edges
    else:
        raise Exception("action_fn not known")
    
def get_action_target(action: int, n_nodes: int, n_edges: int) -> int:
    """returns the node/edge index the action is acting on"""
    if action < oc.N_NODE_ACTIONS * n_nodes:
        action_idx = action // oc.N_NODE_ACTIONS
    elif action < oc.N_NODE_ACTIONS * n_nodes + oc.N_EDGE_ACTIONS * n_edges:
        action = action - (oc.N_NODE_ACTIONS * n_nodes)
        action_idx = action // oc.N_EDGE_ACTIONS
    else:
        action_idx = 0
    return action_idx
    