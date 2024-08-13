import numpy as np


T    = 0.5         # Time horizon
dt   = 0.01        # Time step 
N    = int(T/dt)   # Number of step
iter = 1000        # Max number of iteration 


## ==> LIMITS (same of A2)  
#  Weights are set higher for position while lower for velocity and actuation
lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
w_q = 1e3

lowerVelocityLimit = -10
upperVelocityLimit = 10
w_v = 1e-3

lowerControlBound = -9.81
upperControlBound = 9.81
w_u = 1e-3


## ==> OTHERS  
# Multiprocess
processor = 4

# Number of state and control 
ns = 2
nu = 1


# ==> NN
train_size = 0.8    # Ratio of dataset in train set
epochs     = 300    # Total epoches 
patience   = 10     # Epochs to wait before early stoping
L_rate     = 0.001  # Learing rate (default value for Adam)


# ==> MCP
TC_on         = 1                        # flag for terminal constrains
TC_limit      = 1 - 1e-5                 # Set close to 1 to enforce the viability. Conservative if closer (1e-7) 5e-6
initial_state = np.array([np.pi, 2.5])  # I.C.
mpc_step      = 200                    
q_target      = 3/4*np.pi                # Target postion


# ==> FUNCTIONS
# Grid sample of the state
def grid(pos, vel):    
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


