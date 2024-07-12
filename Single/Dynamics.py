import numpy as np 
from casadi import sin 

def f_single(x,u):
    dt = 0.01
    l  = 1
    m  = 1
    g  = 9.81
    
    # State as postion and velocity 
    q, dq = x[0], x[1]
    
    # Acceleration 
    ddq = -g / l * sin(q) + u / (m * l**2) 
    
    # Next step
    x_next = x + dt * (np.array([dq,ddq]))
    
    return x_next