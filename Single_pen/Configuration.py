import numpy as np

np.set_printoptions(precision=3, linewidth=200, suppress=True)
LINE_WIDTH = 60

# Multiprocess
processor = 2

# Time definition 
T = 1.0
dt = 0.01
N = int(T/dt)

# MAx number of iteration 
iter = 1000  

# Bounds: severe on position, lighter in velcoty and torque 
# ADJUST THE WEIGHT!
lowerPositionLimit = 3/4*np.pi
upperPositionLimit = 5/4*np.pi
w_q = 1e3

lowerVelocityLimit = -10
upperVelocityLimit = 10
w_v = 1e-3

lowerControlBound = -9.81
upperControlBound = 9.81
w_u = 1e-3

# Number of state and control 
nq = 1
nv = 1
nu = 1

def grid(pos, vel):
    n_ics = pos * vel
    state_array = np.zeros((n_ics,(nq+nv)))
    lin_p = np.linspace(lowerPositionLimit, upperPositionLimit, num=pos)
    lin_v = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=vel)
    grid = np.meshgrid(lin_p, lin_v)
    state_array = np.column_stack((grid[0].flatten(),grid[1].flatten()))
    return state_array, n_ics

def grid_marco(n_pos, n_vel):
    n_ics = n_pos * n_vel   
    possible_q = np.linspace(lowerPositionLimit, upperPositionLimit, num=n_pos)
    possible_v = np.linspace(lowerVelocityLimit, upperVelocityLimit, num=n_vel)
    state_array = np.zeros((n_ics, 2))  

    i = 0
    for q in possible_q:
        for v in possible_v:
            state_array[i, :] = np.array([q, v])
            i += 1
    return state_array

####### SINCE HERE ONLY THE OCP DONE ####### 

# MCP