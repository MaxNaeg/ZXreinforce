import networkx as nx
import numpy as np
import itertools
import cycler
from collections import deque
import matplotlib.pyplot as plt

from . import own_constants as oc
from .ZX_env_max import get_neighbours
from .action_conversion_utils import get_action_name, get_action_type_idx, get_action_target

#Networkx utils------------------------------------------------------------------------------------

color_palette_ibm = ['#648fff', '#ffb000', '#dc267f', '#785ef0', '#fe6100']
cycler_color_palette_ibm = cycler.cycler(color=color_palette_ibm)

def get_angle_networkx(angle:list)->str:
    """return angle for plotting in networkx graph"""
    if np.all(angle==oc.ZERO):
        return ""
    elif np.all(angle==oc.PI_half):
        return r"$\pi/2$"

    elif np.all(angle==oc.PI):
        return r"$\pi$"

    elif np.all(angle==oc.PI_three_half):
        return r"$3\pi/2$"
  
    elif np.all(angle==oc.ARBITRARY):
        return r"$\alpha$"
    
    elif np.all(angle==oc.NO_ANGLE):
        return ""
    

def get_color_networkx(color:list)->str:
    """get color of node for netwrokx graph"""
    if np.all(color==oc.INPUT):
        return '#ffb000'
    elif np.all(color==oc.OUTPUT):
        return '#ffb000'
    elif np.all(color==oc.RED):
        return '#dc267f'
    elif np.all(color==oc.GREEN):
        return '#648fff'
    elif np.all(color==oc.HADAMARD):
        return "black"
    else:
        raise Exception("COLOR NOT RECOGNISED")

def get_color_edge_networkx(selected_edges:int)->str:
    """get color of edge for networkx graph"""
    if selected_edges == 1:
        return '#785ef0'
    else:
        return "grey"
    
def observation_to_networkx(observation:tuple)->nx.DiGraph:
    """return networkx graph from observation as returned by environment"""
    colors, angles, selected_node, source, target, selected_edges, n_nodes, n_edges, context = observation
    G = nx.DiGraph()
    for idx in range(n_nodes):
        G.add_node(idx, angle=get_angle_networkx(angles[idx]), color=get_color_networkx(colors[idx]))
    for idx in range(n_edges):
        G.add_edge(source[idx], target[idx], color=get_color_edge_networkx(selected_edges[idx]))
    return G


#Observation potting utils------------------------------------------------------------------------------------

def plot_action(observation:tuple, observation_new:tuple, action:int, figsize=(8,7)):
    """observation: observation before action was applied,
        observation_new: observation after action was applied,
        action: action that was applied,
        figsize: size of figure,

        Plots the two observations over each other.
    """
    
    action_name = get_action_name(get_action_type_idx(int(observation[-3]), int(observation[-2]), int(action)))
    ac_idx = get_action_target(int(action), int(observation[-3]), int(observation[-2]))

    plt.figure(figsize=figsize)
    plt.subplot(211)
    plot_observation(observation)
    plt.subplot(212)
    plt.title(f"{action_name, ac_idx}")
    plot_observation(observation_new)
    plt.tight_layout()
    plt.show()


def plot_observation(observation:tuple):
    """Plots observation using networkx"""

    colors, angles, selected_node, source, target, selected_edges, n_nodes, n_edges, context = observation
    # Build graph
    net_graph = observation_to_networkx(observation)
    # node and edge colors
    edge_colors = nx.get_edge_attributes(net_graph,'color').values()
    node_colors = [node[1]['color'] for node in net_graph.nodes(data=True)]

   
    # Node labels are index and angle
    node_labels = nx.get_node_attributes(net_graph, "angle")
    for node, label in node_labels.items():
        node_labels[node] = str(node) + "\n" + label
   


    # Sort graph into layers
    layer_list = sort_graph_into_layers(colors, source, target)
    layer_dict = layer_list_to_node_attrs(layer_list)
    nx.set_node_attributes(net_graph, layer_dict)   
    # Get positons using layers
    pos = nx.multipartite_layout(net_graph, subset_key="layer")

    # Node size and edge width
    options = {
        'node_size': 100,
        'width': 3,
        'alpha': 0.8,
    }

    # Edge labels are indices
    edge_labels = {}
    for i, (s, t) in enumerate(zip(source, target)):
        edge_labels[(s,t)] = i


    nx.draw(net_graph, pos = pos, labels=node_labels, edge_color=edge_colors, node_color=node_colors, **options)
    nx.draw_networkx_edge_labels(net_graph, pos, edge_labels=edge_labels, alpha=0.8, 
                                 bbox={"boxstyle":'round', 
                                       "ec":(1.0, 1.0, 1.0), 
                                       "fc":(1.0, 1.0, 1.0), 
                                       "alpha":0.0,})


def layer_list_to_node_attrs(lay_list:list)->dict:
    """Transforms layer list to dict that can be used as attribute for networkx graph"""
    attr = {}
    for idx, lay in enumerate(lay_list):
        for node in lay:
            attr[node] = {'layer': idx}
    return attr


def sort_graph_into_layers(colors:np.ndarray, source:np.ndarray, target:np.ndarray)->list:
    """Sort graph into layers. Each layer has no connection between nodes. First layer holds input nodes."""
    n_nodes = len(colors)
    input_nodes = [idx for idx, color in enumerate(colors) if np.all(colors[idx] == np.array(oc.INPUT))]
    output_nodes = [idx for idx, color in enumerate(colors) if np.all(colors[idx] == np.array(oc.OUTPUT))]

    # For sorting subgraphs
    if len(input_nodes) + len(output_nodes) ==0:
        input_nodes = [n_nodes - 1]


    layers = deque()
    layers.append(input_nodes)
    leftover_nodes = deque()

    # To make sure algorithm doesnt run forever with wrong input
    max_layers = 1000

    # Add layer starting from input nodes
    while len(list(itertools.chain(*layers))) < n_nodes and len(layers) < max_layers:
        connected_to_last_layer = []
        for node in layers[-1]:
            # Get neighbours of node in last layer
            connected_to_node = list(get_neighbours(node, source, target))
            for con_node in connected_to_node:
                # If neighbour not yet asigned to layer add to connected nodes
                if (not con_node in list(itertools.chain(*layers)) and 
                    not con_node in leftover_nodes and
                    not con_node in connected_to_last_layer):
                    connected_to_last_layer.append(con_node)

        if len(leftover_nodes) == 0 and len(connected_to_last_layer) == 0:
             break
        
        layers.append([])

        # Iterate through leftover nodes:
        for _ in range(len(leftover_nodes)):
            con_node = leftover_nodes.popleft()
            neighs_con = list(get_neighbours(con_node, source, target))
            add_to_layer = True
            for neigh_con in neighs_con:
                    if neigh_con in layers[-1]:
                        add_to_layer = False
                        break
            if add_to_layer:
                layers[-1].append(con_node)
            else:
                leftover_nodes.appendleft(con_node)

        # Iterate through connected nodes
        for con_node in connected_to_last_layer:
            neighs_con = list(get_neighbours(con_node, source, target))
            # Add con node if it is not neighbour of other node already in layer
            add_to_layer = True
            for neigh_con in neighs_con:
                if neigh_con in layers[-1]:
                    add_to_layer = False
                    break
            if add_to_layer:
                layers[-1].append(con_node)
            else:
                leftover_nodes.append(con_node)



    disconnected_output_nodes = [idx for idx in output_nodes if not idx in list(itertools.chain(*layers))]
    
    layers_d = deque()
    layers_d.append(disconnected_output_nodes)
    leftover_nodes = deque()

    # To make sure algorithm doesnt run forever with wrtong input
    max_layers = 1000

    # Add layer starting from disconnected output nodes
    while len(list(itertools.chain(*layers_d))) < n_nodes and len(layers_d) < max_layers:
        connected_to_last_layer = []
        for node in layers_d[-1]:
            # Get neighbours of node in last layer
            connected_to_node = list(get_neighbours(node, source, target))
            for con_node in connected_to_node:
                # If neighbour not yet asigned to layer add to connected nodes
                if (not con_node in list(itertools.chain(*layers_d)) and 
                    not con_node in leftover_nodes and
                    not con_node in connected_to_last_layer):
                    connected_to_last_layer.append(con_node)

        if len(leftover_nodes) == 0 and len(connected_to_last_layer) == 0:
             break
        
        layers_d.append([])

        # Iterate through leftover nodes:
        for _ in range(len(leftover_nodes)):
            con_node = leftover_nodes.popleft()
            neighs_con = list(get_neighbours(con_node, source, target))
            add_to_layer = True
            for neigh_con in neighs_con:
                    if neigh_con in layers_d[-1]:
                        add_to_layer = False
                        break
            if add_to_layer:
                layers_d[-1].append(con_node)
            else:
                leftover_nodes.appendleft(con_node)

        # Iterate through connected nodes
        for con_node in connected_to_last_layer:
            neighs_con = list(get_neighbours(con_node, source, target))
            # Add con node if it is not neighbour of other node aklready in layer
            add_to_layer = True
            for neigh_con in neighs_con:
                if neigh_con in layers_d[-1]:
                    add_to_layer = False
                    break
            if add_to_layer:
                layers_d[-1].append(con_node)
            else:
                leftover_nodes.append(con_node)


    n_input_lay = len(layers)
    n_output_lay = len(layers_d)
    layers = list(layers)
    layers_d = list(layers_d)

    final_layers = []
    if n_input_lay > n_output_lay:
        final_layers = layers
        for idx, output_lay in enumerate(layers_d):
            final_layers[-idx-1] += output_lay
    else:
        final_layers = [layers[0]] + layers_d[::-1]
        for idx, lay in enumerate(layers[1:]):
            final_layers[1+idx] += lay

    
    return final_layers


def plot_action_hist(value_dict:dict, step_idx_list:list=[-1], norm:bool=True, figsize=(10,2), 
                     savename:str=None, color_cycler:cycler=cycler_color_palette_ibm):
    """Plot histogram of actions taken in training"""

    n_actions=oc.N_NODE_ACTIONS + oc.N_EDGE_ACTIONS +1
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_prop_cycle(color_cycler)
    for step_idx in step_idx_list:
        action_counts=[]
        labels=[]
        for idx in range(n_actions):
            action_counts.append(value_dict[get_action_name(idx)][step_idx])
            labels.append(get_action_name(idx))
        if norm:
            action_counts = action_counts/np.max(action_counts)
        ax.scatter(range(n_actions), action_counts, label=f"train step: {step_idx}")

    ax.set_xticks(range(n_actions))
    ax.set_xticklabels(labels, rotation=90)
    ax.grid()
    ax.tick_params(direction="in", right=True, top=True)
    if savename:
        plt.savefig(savename)
    plt.legend()
    plt.show()
