# Holds resetters for the ZX env returning 
# (colors, angles, selected_node, source, target, selected_edges)

import numpy as np
from .own_constants import (INPUT, OUTPUT, GREEN, RED, HADAMARD, 
                      ZERO, PI_half, PI, PI_three_half, ARBITRARY, NO_ANGLE,
                      ANGLE_LIST)
from .ZX_env_max import apply_auto_actions, add_edge


class Resetter_ZERO_PI_PIHALF_ARB_hada():
    def __init__(self,
                 n_in_min:int,
                 n_in_max:int,
                 min_spiders:int,
                 max_spiders:int,
                 pi_fac:float,
                 pi_half_fac:float,
                 arb_fac:float,
                 p_hada:float,
                 min_mean_neighbours:int,
                 max_mean_neighbours:int,
                 rng:np.random.Generator):
        """n_in_min: minimum number of input spiders,
        n_in_max: maximum number of input spiders,
        min_spiders: minimum number of spiders in total,
        max_spiders: maximum number of spiders in total,
        pi_fac: factor by which to reduce probability of pi angle,
        pi_half_fac: factor by which to reduce probability of pi/2 angle,
        arb_fac: factor by which to reduce probability of arbitrary angle,
        p_hada: factor by which to reduce probability of hadamard node,
        min_mean_neighbours: minimum number of neighbours per node,
        max_mean_neighbours: maximum number of neighbours per node,
        rng: numpy random generator"""
        self.n_in_min = n_in_min
        self.n_in_max = n_in_max
        self.min_spiders = min_spiders
        self.max_spiders = max_spiders
        self.pi_fac = pi_fac
        self.pi_half_fac = pi_half_fac
        self.arb_fac = arb_fac
        self.p_hada = p_hada
        self.min_mean_neighbours = min_mean_neighbours
        self.max_mean_neighbours = max_mean_neighbours
        self.rng = rng

    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds random ZX diagrams"""
        # Sample inout and output number unifomrly
        n_input = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        n_output = self.rng.integers(low=self.n_in_min, high=self.n_in_max+1)
        # Sample number of spiders uniformly
        n_init_spiders = self.rng.integers(low=self.min_spiders, high=self.max_spiders+1)
        # Sample number of hadamards
        n_hada  = self.rng.integers(low=0, high=(n_init_spiders * self.p_hada))
        # Make sure there is at least one spider
        n_init_spiders = np.max([n_init_spiders, 1])

        # Sample neighbour number uniformly
        mean_neighbours = self.rng.integers(low=self.min_mean_neighbours, high=self.max_mean_neighbours+1)
        # Calculate probability of each edge such that self.mean_neighbours 
        # is expected value of neighbours per node
        p_edge = (mean_neighbours - (n_input + n_output) / n_init_spiders) / (n_init_spiders+1)
        if p_edge < 0:
            p_edge = 0

        # Sample probabilities for angles uniformly, reduce, and normalize
        p_zero, p_pi, p_pi_half, p_arb = self.rng.uniform(size=4)
        p_pi *= self.pi_fac
        p_arb *= self.arb_fac
        p_pi_half *= self.pi_half_fac

        norm = p_zero + p_pi + p_arb + p_pi_half
        p_zero /= norm
        p_pi /= norm
        p_arb /= norm
        p_pi_half /= norm

        ps_angle = [0]*(len(ZERO)-1)
        ps_angle[np.where(PI)[0][0]] = p_pi
        ps_angle[np.where(PI_half)[0][0]] = p_pi_half
        ps_angle[np.where(ZERO)[0][0]] = p_zero
        ps_angle[np.where(ARBITRARY)[0][0]] = p_arb

        # Sample color of each spider randomly
        red_spiders= self.rng.binomial(n_init_spiders, 0.5)
        # Hackky way to make 1d numpy array out of angle list
        angle_arr = np.empty(len(ANGLE_LIST)+1, dtype="O")
        angle_arr[:] = (ANGLE_LIST+[ARBITRARY])[:]

        # Create angle list for all spiders
        node_angle_list = self.rng.choice(angle_arr, size=n_init_spiders, p=ps_angle) 
        node_angle_list = node_angle_list

        # Create color list for all spiders
        node_color_list = np.array([RED] * red_spiders + [GREEN] * (n_init_spiders - red_spiders))
        self.rng.shuffle(node_color_list, axis=0)
        node_color_list = node_color_list.tolist()

        colors = np.array([INPUT] * n_input + [OUTPUT] * n_output + node_color_list)
        angles = np.array([NO_ANGLE] * (n_input + n_output) + list(node_angle_list))

        # Create edges
        edge_source, edge_target = np.where(np.triu(np.reshape(
            self.rng.choice(2, size=int(n_init_spiders**2), p=(1-p_edge, p_edge)), (n_init_spiders, n_init_spiders)), k=1))
        
        source = np.array(list(np.arange(n_input + n_output)) + list(edge_source + n_input + n_output))
        target = np.array(list(np.arange(n_input + n_output, 2 * n_input + n_output)) + 
                       list(np.arange(2 * n_input + n_output, 2 * n_input + 2 * n_output)) + 
                       list(edge_target + n_input + n_output))

        # Apply automatic actions
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        # Add hadamards
        idcs_to_connect = np.arange(n_input + n_output, len(colors))

        if len(idcs_to_connect) >= 2:
            for _ in range(n_hada):

                idx_new_node = len(colors)
                colors = np.row_stack((colors, HADAMARD))
                angles = np.row_stack((angles, NO_ANGLE))


                connected_idx1 = self.rng.choice(idcs_to_connect, 1)[0]
                new_to_connect = np.delete(idcs_to_connect, np.where(idcs_to_connect==connected_idx1)[0])
                connected_idx2 = self.rng.choice(new_to_connect, 1)[0]

                source, target = add_edge(idx_new_node, connected_idx1, source, target)
                source, target = add_edge(idx_new_node, connected_idx2, source, target)

        # Apply automatic actions
        colors, angles, source, target = apply_auto_actions(
            colors, angles, source, target)
        
        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges

class Resetter_Test_COPY():
    def __init__(self, 
                 n_out:int,
                 n_extra_node:int, 
                 angle_copy:list=ZERO, 
                 color_zero:list=GREEN, 
                 angles_out:list=ARBITRARY, 
                 angle_other:list=ARBITRARY):
        """n_out: number of output nodes,
        n_extra_node: number of extra nodes inserted on outputs,
        angle_copy: angle of copy node,
        color_zero: color of zero node,
        angles_out: angle of output nodes,
        angle_other: angle of extra nodes"""
        self.n_out = n_out
        self.n_extra_node = n_extra_node
        self.color_zero = color_zero
        self.angle_copy=angle_copy
        # Spiders need oposite color
        if np.all(self.color_zero == GREEN):
            self.other_col = RED
        else:
            self.other_col = GREEN
        self.angles_out=angles_out
        self.angle_other=angle_other
    
    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Copy test ZX diagram"""
        colors = np.array([self.color_zero] * 1 + [self.other_col] * 1 
                            + [self.color_zero] * self.n_extra_node 
                            + [OUTPUT]* self.n_out)
        angles = np.array([self.angle_copy] * 1 + [self.angle_other] * 1 
                            + [self.angles_out] * self.n_extra_node
                            + [NO_ANGLE]* self.n_out)

        source = [0,]
        target = [1,]
        for i in range(self.n_extra_node):

            source.append(1)
            target.append(2 + i)

            source.append(2 + i)
            target.append(2 + i + self.n_extra_node)

        for i in range(self.n_extra_node, self.n_out):
            source.append(1)
            target.append(2 + i + self.n_extra_node)

        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    

class ResetterBialgUnmerge():
    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Bialgabra diagram with one ectra output"""
        colors=[]
        angles=[]

        colors.append(INPUT)
        colors.append(INPUT)

        colors.append(GREEN)
        colors.append(GREEN)
        colors.append(RED)
        colors.append(RED)

        colors.append(OUTPUT)
        colors.append(OUTPUT)
        colors.append(OUTPUT)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        source = [0, 1, 2, 2, 3, 3, 4, 5, 5,]
        target = [2, 3, 4, 5, 4, 5, 6, 7, 8,]

        selected_node = [0]*len(colors)
        selected_edges = [0]*len(source)

        return (np.array(colors), np.array(angles), np.array(selected_node),
                np.array(source), np.array(target), np.array(selected_edges))
    

class ResetterBialgLvl2():
    def reset(self)->tuple:
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Bialgabra diagram level 2"""
        colors=[]
        angles=[]

        colors.append(INPUT)
        colors.append(INPUT)

        colors.append(GREEN)
        colors.append(GREEN)
        colors.append(RED)
        colors.append(RED)
        colors.append(RED)

        colors.append(OUTPUT)
        colors.append(OUTPUT)
        colors.append(OUTPUT)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)
        angles.append(ZERO)

        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)
        angles.append(NO_ANGLE)

        source = [0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6,]
        target = [2, 3, 4, 5, 6, 4, 5, 6, 7, 8, 9,]

        selected_node = [0]*len(colors)
        selected_edges = [0]*len(source)

        return (np.array(colors), np.array(angles), np.array(selected_node),
                np.array(source), np.array(target), np.array(selected_edges))




        

class Bilagebra_lvl_n_m():
    def __init__(self, 
                 n_min:int,
                 n_max:int,
                 rng:np.random.Generator):
        """n_min: minimum number of nodes on one side,
        n_max: maximum number of nodes on one side
        rng: numpy random generator"""
        self.n_min = n_min
        self.n_max = n_max
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra with n_min to n_max nodes on each side"""
        n_green = self.rng.integers(low=self.n_min, high=self.n_max+1)
        n_red = self.rng.integers(low=self.n_min, high=self.n_max+1)

        colors = np.array(
            [INPUT] * n_green + [GREEN, ] * n_green + [RED,] * n_red + [OUTPUT,] * n_red, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_green + [ZERO, ] * n_green + [ZERO,] * n_red + [NO_ANGLE,] * n_red, 
            dtype=np.int32
        )

        source = []
        target = []
        # Edges between input and green
        for i in range(n_green):
            source.append(i)
            target.append(n_green + i)
        # Edges between red and output
        for i in range(2*n_green, 2*n_green + n_red):
            source.append(i)
            target.append(n_red + i)
        # Edges between green and red
        for i in range(n_green, 2*n_green):
            for j in range(2*n_green, 2*n_green + n_red):
                source.append(i)
                target.append(j)

        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    

class Bilagebra_add_edges():
    def __init__(self, 
                 add_min:int,
                 add_max:int,
                 rng:np.random.Generator):
        """add_min: minimum number of edges to add,
        add_max: maximum number of edges to add
        rng: numpy random generator"""
        self.add_min = add_min
        self.add_max = add_max
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra multiple outputs added to one node"""
        add_max = self.rng.integers(low=self.add_min, high=self.add_max+1)
        index_conn = self.rng.integers(low=0, high=4)
        print(index_conn)

        source = []
        target = []

        if index_conn < 2:
            n_input = 2 + add_max
            n_output = 2
            for i in range(2, 2 + add_max):
                source.append(index_conn+n_input)
                target.append(i)
        else:
            n_input = 2
            n_output = 2 + add_max
            for i in range(n_input + 6, n_input + 6 + add_max):
                source.append(index_conn+n_input)
                target.append(i)

        # Edges between green and red
        for i in range(n_input, n_input+2):
            source.append(i)
            target.append(n_input+2)
            source.append(i)
            target.append(n_input+3)

        # At least on edge between input and green
        for i in range(0, 2):
            source.append(i)
            target.append(n_input + i)

        # At least on edge between output and red
        for i in range(n_input + 2, n_input + 4):
            source.append(i)
            target.append(2 + i)

        colors = np.array(
            [INPUT] * n_input + [GREEN, ] * 2 + [RED,] * 2 + [OUTPUT,] * n_output, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_input + [ZERO, ] * 2 + [ZERO,] * 2 + [NO_ANGLE,] * n_output, 
            dtype=np.int32
        )


        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges
    


class Bilagebra_add_angle():
    def __init__(self, 
                 rng:np.random.Generator):
        """rng: numpy random generator"""
        self.rng=rng


    def reset(self):
        """returns (colors, angles, selected_node, source, target, selected_edges)
        Builds Bialgebra with random angle added to one node"""
        index_angle = self.rng.integers(low=0, high=4)
        angle_type = self.rng.choice([ZERO, PI_half, PI, PI_three_half, ARBITRARY])

        source = []
        target = []

        if index_angle < 2 :
            n_input = 1
            n_output = 2

            source.append(0)
            target.append(2 - index_angle)

            source.append(3)
            target.append(5)
            source.append(4)
            target.append(6)
        else:
            n_input = 2
            n_output = 1

            source.append(0)
            target.append(2)
            source.append(1)
            target.append(3)

            source.append(6)
            target.append(7 - index_angle)


        # Edges between green and red
        for i in range(n_input, n_input+2):
            source.append(i)
            target.append(n_input+2)
            source.append(i)
            target.append(n_input+3)

        colors = np.array(
            [INPUT] * n_input + [GREEN, ] * 2 + [RED,] * 2 + [OUTPUT,] * n_output, 
            dtype=np.int32
        )
        angles = np.array(
            [NO_ANGLE] * n_input + [ZERO, ] * 2 + [ZERO,] * 2 + [NO_ANGLE,] * n_output, 
            dtype=np.int32
        )

        angles[n_input+index_angle] = angle_type


        source = np.array(source, dtype=np.int32)
        target = np.array(target, dtype=np.int32)


        selected_edges = np.zeros(len(source), dtype=np.int32)
        selected_node = np.zeros(len(colors), dtype=np.int32)

        return colors, angles, selected_node, source, target, selected_edges


