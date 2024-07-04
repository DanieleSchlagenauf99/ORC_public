#from ocp_conf import grid 
import Configuration as conf
import numpy as np 
import ocp

N = conf.N
p = ocp.OcpSinglePendulum()
x_init = [-0.8, 0.7]
sol = p.solve(x_init, N)