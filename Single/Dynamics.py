import numpy as np 
import casadi as cas 
import Configuration as conf

def f_single(x,u):
    dt = conf.dt  # Time step
    l  = 1        # Length 
    m  = 1        # Mass
    g  = 9.81     # Gravity 
    
    # State as (q, v) 
    q, dq = x[0], x[1]
    
    # Acceleration 
    ddq = -g / l * cas.sin(q) + u / (m * l**2) 
    
    # Next step
    x_next = x + dt * (np.array([dq,ddq]))

    return x_next



