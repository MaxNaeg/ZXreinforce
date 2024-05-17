# One hot encoding of colors
INPUT = [1,0,0,0,0]
OUTPUT = [0,1,0,0,0]   
GREEN = [0,0,1,0,0]   
RED = [0,0,0,1,0]   
HADAMARD = [0,0,0,0,1]    

# One hot encoding of angles
ZERO = [1,0,0,0,0,0] 
PI_half = [0,1,0,0,0,0]  
PI = [0,0,1,0,0,0] 
PI_three_half = [0,0,0,1,0,0] 
ARBITRARY = [0,0,0,0,1,0] 
NO_ANGLE = [0,0,0,0,0,1] 

# List of available angles
ANGLE_LIST = [ZERO, PI_half, PI, PI_three_half]


# Number of node/edge actions
N_NODE_ACTIONS = 6
N_EDGE_ACTIONS = 6
