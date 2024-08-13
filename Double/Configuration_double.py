import numpy as np
from random import uniform


## ==> Time definition 
T    = 0.5
dt   = 0.01
N    = int(T/dt)    
iter = 5000    

## ==> Random
rnd_augment   = 0
n_rnd         = 295
# Reducing the velcoity bound according to the single pendulumn viabile set
# In order to increase the possibility to found viable points
rnd_vel = 10


## ==> Bounds and weights for both  
lowerPositionLimit1 = 3/4*np.pi
upperPositionLimit1 = 5/4*np.pi
lowerPositionLimit2 = 3/4*np.pi
upperPositionLimit2 = 5/4*np.pi
w_q1 = 1e3
w_q2 = 1e3

lowerVelocityLimit1 = -10
upperVelocityLimit1 =  10
lowerVelocityLimit2 = -10
upperVelocityLimit2 =  10
w_v1 = 1e-3
w_v2 = 1e-3 

# Double seems to be enough
lowerControlBound1 = -9.81*5
upperControlBound1 =  9.81*5
lowerControlBound2 = -9.81*4
upperControlBound2 =  9.81*4
w_u1 = 1e-3
w_u2 = 1e-3

# Target postion 
q1_target = np.pi
q2_target = np.pi


# Multiprocess
processor = 4

# Number of state and control 
ns = 4
nu = 2


# ======= NN
train_size = 0.8    # Ratio of dataset in train set
epochs     = 300    # Total epoches 
patience   = 15     # Epochs to wait before early_stop
L_rate     = 0.0005 # Learing rate, default value for both Adam && Nadam



# ======= MCP
TC_on         = 1    # flag for terminal constrains
TC_limit      = 0.5
initial_state = np.array([5/4*np.pi, -1.0, np.pi, 0.0])    
mpc_step      = 200             


# ====== DEF
# Grid sample of the state
def grid(pos1, vel1 ,pos2, vel2):    # as input the grid wanted dimension
    n_ics = pos1 * vel1 * pos2 * vel2
    state_array = np.zeros((n_ics,(ns)))
    lin_p1      = np.linspace(lowerPositionLimit1, upperPositionLimit1, num=pos1)
    lin_p2      = np.linspace(lowerPositionLimit2, upperPositionLimit2, num=pos2)
    lin_v1      = np.linspace(lowerVelocityLimit1, upperVelocityLimit1, num=vel1)
    lin_v2      = np.linspace(lowerVelocityLimit2, upperVelocityLimit2, num=vel2)
    
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

# Random sample 
def random(n_ics):
    state_array = np.zeros((n_ics,ns))
    for i in range(n_ics):
        state_array[i][0] = uniform(lowerPositionLimit1, upperPositionLimit1)
        state_array[i][1] = uniform(-rnd_vel, rnd_vel)
        state_array[i][2] = uniform(lowerPositionLimit2, upperPositionLimit2)
        state_array[i][3] = uniform(-rnd_vel, rnd_vel)
    
    return state_array     

