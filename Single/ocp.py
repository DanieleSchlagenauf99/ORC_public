import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import Dynamics
import multiprocessing
import Configuration as conf

import time



class OcpSinglePendulum:
    def __init__(self):
        # Time 
        self.T   = conf.T                
        self.dt  = conf.dt    
        # Weights           
        self.w_q = conf.w_q                           
        self.w_v = conf.w_v   
        self.w_u = conf.w_u 
        
    def solve(self, x_init, N ,X_guess = None, U_guess = None): 
        self.opti = cas.Opti()
        
        # Casadi variables
        self.q = self.opti.variable(N)       
        self.v = self.opti.variable(N)
        self.u = self.opti.variable(N-1)
        q = self.q
        v = self.v
        u = self.u
        
        # Initialization 
        if (X_guess is not None):                           # State
            for i in range(N):
                self.opti.set_initial(q[i], X_guess[i,0])
                self.opti.set_initial(v[i], X_guess[i,1]) 
        else:
            for i in range(N):
                self.opti.set_initial(q[i], x_init[0])
                self.opti.set_initial(v[i], x_init[1])
        
        if (U_guess is not None):                           # Input
            for i in range(N-1):
                self.opti.set_initial(u[i], U_guess[i,:])
        
        
        ## ==> Cost function 
        self.cost = 0
        self.running_costs = [None,]*(N)      
        for i in range(N):
            self.running_costs[i] =  self.w_q * (q[i] - conf.q_target)**2       # Position 
            self.running_costs[i] += self.w_v * v[i]**2                         # Velocity
            if (i<N-1):            
                self.running_costs[i] += self.w_u * u[i]**2                     # Input 
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)
        
        ## ==> Constrains
        # Initial state 
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])
        
        for i in range(N):
            # Bounded state
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
            
            if (i<N-1):
                # Dynamics
                x_plus = Dynamics.f_single(np.array([q[i], v[i]]), u[i])
                self.opti.subject_to(q[i+1] == x_plus[0])
                self.opti.subject_to(v[i+1] == x_plus[1])
                # Bounded Torque
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))
               
        
        ## ==> Chosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter)}
        self.opti.solver("ipopt", opts, s_opst)
        return self.opti.solve()
    
 



if __name__ == "__main__":

    # OCP problem
    ocp = OcpSinglePendulum()
    
    # Inital state grid
    npos = 30
    nvel = 30
    state_array, n_ics = conf.grid(npos, nvel)
    
    # State computation function
    def ocp_function_single_pendulum(index):
        viable = []
        no_viable = []
        
        for i in range(index[0], index[1]):
            x = state_array[i, :]
            try:
                #sol = ocp.solve(x, conf.N)   sol will not be used!
                ocp.solve(x, conf.N)
                viable.append([x[0], x[1]])
                print("Feasible initial state found: [{:.3f}   {:.3f}]".format(*x))
            except RuntimeError as e:                     
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found: [{:.3f}   {:.3f}]".format(*x))
                    no_viable.append([x[0], x[1]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable
    
    
    ## ==> Multi process  
    # state equal division 
    indexes = np.linspace(0, n_ics, num=conf.processor+1)
    args = []
    for i in range(conf.processor):
        args.append((int(indexes[i]), int(indexes[i+1])))
        
    # Pool logic application 
    pool = multiprocessing.Pool(processes=conf.processor)
    start = time.time()
    results = pool.map(ocp_function_single_pendulum, args)
    pool.close()
    pool.join()
    end = time.time()
    
    # Print duration 
    conf.print_time(start, end)
    
    
    '''
    # old
    viable_states_list = []
    no_viable_states_list = []
    for i in range(conf.processor):
        viable_states_list.append(results[i][0])
        no_viable_states_list.append(results[i][1])
    '''
    ##################### Updated 
    # List state from each processor
    viable_states    = [results[i][0] for i in range(conf.processor)]
    no_viable_states = [results[i][1] for i in range(conf.processor)]
    
    viable_states    = np.concatenate(viable_states)
    no_viable_states = np.concatenate(no_viable_states)



    ## ==> Plot state space
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='c', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='m', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()



    ## ==> Dataset creation 
    viable_states    = np.column_stack((viable_states, np.ones(len(viable_states), dtype=int)))
    no_viable_states = np.column_stack((no_viable_states, np.zeros(len(no_viable_states), dtype=int)))
    dataset          = np.concatenate((viable_states, no_viable_states))

    # Export data in csv file
    columns = ['q', 'v', 'viable']
    with open('data_single.csv', 'w') as f:
        f.write(','.join(columns) + '\n')
        np.savetxt(f, dataset, delimiter=',', fmt='%s')



    