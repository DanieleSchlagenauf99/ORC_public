from casadi import sin, cos
import numpy as np

def f_double(x, u):
    dt = 0.01 
    l1 = 1 
    l2 = 1 
    m1 = 1 
    m2 = 1 
    g = 9.81 

    # State (q1, v1, q2, v2) and input 
    q1, dq1, q2, dq2 = x[0], x[1], x[2], x[3]
    u1, u2 = u[0], u[1]

    # Accelerations
    ddq1 = (l1**2 * l2 * m2 * dq1**2 * sin(-2 * q2 + 2 * q1)
            + 2 * u2 * cos(-q2 + q1) * l1
            + 2 * ( g * sin(-2 * q2 + q1) * l1 * m2 / 2
                    + sin(-q2 + q1) * dq2**2 * l1 * l2 * m2
                    + g * l1 * (m1 + m2 / 2) * sin(q1)
                    - u1 ) * l2
            ) / l1**2 / l2 / (m2 * cos(-2 * q2 + 2 * q1) - 2 * m1 - m2)
    
    ddq2 = (-g * l1 * l2 * m2 * (m1 + m2) * sin(-q2 + 2 * q1)
            - l1 * l2**2 * m2**2 * dq2**2 * sin(-2 * q2 + 2 * q1)
            - 2 * dq1**2 * l1**2 * l2 * m2 * (m1 + m2) * sin(-q2 + q1)
            + 2 * u1 * cos(-q2 + q1) * l2 * m2
            + l1 * (m1 + m2) * (sin(q2) * g * l2 * m2 - 2 * u2)
            ) / l2**2 / l1 / m2 / (m2 * cos(-2 * q2 + 2 * q1) - 2 * m1 - m2)
    
    # Next step 
    x_next = x + dt*np.array([dq1, ddq1, dq2, ddq2])
    return x_next