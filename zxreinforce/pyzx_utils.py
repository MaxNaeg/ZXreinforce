import numpy as np

from fractions import Fraction

import pyzx as zx
from pyzx.graph.base import BaseGraph
from pyzx.utils import VertexType as vt
from pyzx.utils import EdgeType as et



from zxreinforce.own_constants import (INPUT, OUTPUT, HADAMARD, GREEN, RED, ZERO, 
                                       PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE)


def obs_to_pzx(obs):
    """Convert an observation of the ZXenv to a pyzx graph."""
    # pyzx can't handle arbitrary angles, so we pick a very small angle instead
    max_arb=100
    g=zx.Graph()
    for color, angle in zip(obs[0], obs[1]):
        
        if np.all(angle == ZERO):
            zx_angle = 0
        elif np.all(angle == PI):
            zx_angle = 1
        elif np.all(angle == PI_half):
            zx_angle = Fraction(1,2)
        elif np.all(angle == PI_three_half):
            zx_angle = Fraction(3,2)
        elif np.all(angle == ARBITRARY):
            zx_angle = Fraction(1, max_arb * 4)

        if np.all(color == INPUT):
            g.add_vertex(vt.BOUNDARY, row=0)
        elif np.all(color == OUTPUT):
            g.add_vertex(vt.BOUNDARY, row=2)
        elif np.all(color == GREEN):
            g.add_vertex(vt.Z, phase=zx_angle, row=1)
        elif np.all(color == RED):
            g.add_vertex(vt.X, phase=zx_angle, row=1)
        elif np.all(color == HADAMARD):
            g.add_vertex(vt.H_BOX, row=1)
        else:
            raise ValueError("Color not recognized")
        
    assert sorted(g.vertices()) == list(range(len(obs[0]))), "Vertices are not in order"
        
    for i, j in zip(obs[3], obs[4]):
        assert i != j, "Self loops are not allowed"
        g.add_edge((i,j))
    return g

def pyzx_to_obs(graph: BaseGraph):
    """Convert a pyzx graph to an observation of the ZXenv without context feratures."""
    colors = []
    angles = []
    source = []
    target = []
    min_row = min(graph.rows().values())
    max_row = max(graph.rows().values())
    
    sorted_vertices = sorted(graph.vertices())
    for node_idx in sorted_vertices:
        zxcol, zxang, zxrow = graph.types()[node_idx], graph.phases()[node_idx], graph.rows()[node_idx]

        if zxang == 0:
            angle = ZERO
        elif zxang == 1:
            angle = PI
        elif zxang == Fraction(1,2):
            angle = PI_half
        elif zxang == Fraction(3,2):
            angle = PI_three_half
        else:
            angle = ARBITRARY
        
        if zxcol == vt.BOUNDARY and zxrow == min_row:
            colors.append(INPUT)
            angles.append(NO_ANGLE)
        elif zxcol == vt.BOUNDARY and zxrow == max_row:
            colors.append(OUTPUT)
            angles.append(NO_ANGLE)
        elif zxcol == vt.Z:
            colors.append(GREEN)
            angles.append(angle)
        elif zxcol == vt.X:
            colors.append(RED)
            angles.append(angle)
        elif zxcol == vt.H_BOX:
            colors.append(HADAMARD)
            angles.append(NO_ANGLE)
        else:
            raise ValueError("Type not recognized")


    for edge in graph.edges():
        edge_type = graph.edge_type(edge)
        i,j = edge
        if edge_type == et.SIMPLE:
            # raise ValueError("Simple edges are not allowed coming from pyzx optimizer")
            source.append(sorted_vertices.index(i))
            target.append(sorted_vertices.index(j))
        elif edge_type == et.HADAMARD:
            colors.append(HADAMARD)
            angles.append(NO_ANGLE)
            source.append(sorted_vertices.index(i))
            target.append(len(colors)-1)
            source.append(len(colors)-1)
            target.append(sorted_vertices.index(j))
        else:
            raise ValueError("Edge type not recognized")
    
    return (np.array(colors), 
            np.array(angles),
            np.zeros(len(colors)), 
            np.array(source), 
            np.array(target),
            np.zeros(len(source)),
            np.array(len(colors)),
            np.array(len(source)),
            np.array([])
            )   
        
            