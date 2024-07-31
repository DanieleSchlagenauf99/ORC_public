import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60


## ==> Time definition 
T    = 0.5
dt   = 0.01
N    = int(T/dt)
iter = 1000    # Max number of iteration 


## ==> Bounds from A2  
#  Weights chosen to be tight on the position and lighter on velcoty and torque
lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
w_q = 1e2

lowerVelocityLimit = -10
upperVelocityLimit = 10
w_v = 1e-1

lowerControlBound = -9.81
upperControlBound = 9.81
w_u = 1e-4

# Target postion 
q_target = 5/4 * np.pi
# Multiprocess

processor = 3
# Number of state and control 
ns = 2
nu = 1


# ======= NN
train_size = 0.8    # Ratio of dataset in train set
epochs     = 300    # Total epoches 
patience   = 10     # Epochs to wait before early_stop
L_rate     = 0.001  # Learing rate, default value for both Adam && Nadam



# ======= MCP
TC_on         =  1                     # flag for terminal constrains
initial_state = np.array([np.pi, 1])   # I.C.
mpc_step      = 100             

## ==> Noise (To use?)
noise = 0
mean = 0.001
std = 0.001


# ====== DEF
# Grid sample of the state
def grid(pos, vel):    # as input the grid wanted dimension
    n_ics = pos * vel
    state_array = np.zeros((n_ics,(ns)))
    lin_p       = np.linspace(lowerPositionLimit, upperPositionLimit, num=pos)
    lin_v       = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=vel)
    grid        = np.meshgrid(lin_p, lin_v)
    state_array = np.column_stack((grid[0].flatten(),grid[1].flatten()))
    
    return state_array, n_ics

# Print time
def print_time(start, end):  # as input the timestamp in UNIX
    diff   = end - start
    h, r   = divmod(diff, 3600)
    m, sec = divmod(r, 60)
    
    print(f'Computation time: {h}h  {m}min  {int(sec)}s')


