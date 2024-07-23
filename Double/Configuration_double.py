import numpy as np
np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60


## ==> Time definition 
T    = 0.5
dt   = 0.01
N    = int(T/dt)
iter = 1000     


## ==> Bounds and weights for both  
lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi
w_q1 = 1e3
w_q2 = 1e3

lowerVelocityLimit1 = -10
upperVelocityLimit1 = 10
lowerVelocityLimit2 = -10
upperVelocityLimit2 = 10
w_v1 = 1e-3
w_v2 = 1e-3

lowerControlBound1 = -9.81
upperControlBound1 = 9.81
lowerControlBound2 = -9.81
upperControlBound2 = 9.81
w_u2 = 1e-3
w_u1 = 1e-3

# Target postion 
q1_target = 3/4*np.pi
q2_target = 5/4*np.pi

# Multiprocess
processor = 4

# Number of state and control 
ns = 4
nu = 2



############# TILL HERE check 
# ======= NN
train_size = 0.8    # Ratio of dataset in train set
epochs     = 300    # Total epoches 
patience   = 10     # Epochs to wait before early_stop
L_rate     = 0.001  # Learing rate, default value for both Adam && Nadam



# ======= MCP
TC_on         = 1     # flag for terminal constrains
initial_state = np.array([np.pi, 1, np.pi, 0])   
mpc_step      = 200             

## ==> Noise (To use?)
noise = 0
mean = 0.001
std = 0.001


# ====== DEF
# Grid sample of the state
def grid(pos1, vel1 ,pos2, vel2):    # as input the grid wanted dimension
    n_ics = pos1 * vel1 * pos2 * vel2
    state_array = np.zeros((n_ics,(ns)))
    lin_p1      = np.linspace(lowerPositionLimit1, upperPositionLimit1, num=pos1)
    lin_p2      = np.linspace(lowerPositionLimit2, upperPositionLimit2, num=pos2)
    lin_v1       = np.linspace(lowerVelocityLimit1, upperVelocityLimit1, num=vel1)
    lin_v2       = np.linspace(lowerVelocityLimit2, upperVelocityLimit2, num=vel2)
    
    ######## indexing TO CHECK for reference index defition (work also without)
    grid = np.meshgrid(lin_p1, lin_v1, lin_p2, lin_v2, indexing='ij')
    state_array = np.column_stack((grid[0].flatten(), grid[1].flatten(), grid[2].flatten(), grid[3].flatten()))
    return state_array, n_ics

    
# Print time
def print_time(start, end):  # as input the timestamp in UNIX
    diff   = end - start
    h, r   = divmod(diff, 3600)
    m, sec = divmod(r, 60)
    
    print(f'Computation time: {h}h  {m}min  {int(sec)}s')