import numpy as np
import matplotlib.pyplot as plt
import casadi as cas
import Dynamics_double as dyn
import multiprocessing
import Configuration_double as conf

import time



class OcpDoublePendulum:
    def __init__(self):
        # Time 
        self.T   = conf.T                
        self.dt  = conf.dt    
        self.N   = conf.N 
        # Weights           
        self.w_q1 = conf.w_q1                          
        self.w_v1 = conf.w_v1  
        self.w_u1 = conf.w_u1
        self.w_q2 = conf.w_q2                           
        self.w_v2 = conf.w_v2   
        self.w_u2 = conf.w_u2 
        
    def solve(self, x_init, N ,X_guess = None, U_guess = None): 
        self.opti = cas.Opti()
        
        # Casadi variables
        self.q1 = self.opti.variable(N)       
        self.v1 = self.opti.variable(N)
        self.u1 = self.opti.variable(N-1)
        self.q2 = self.opti.variable(N)       
        self.v2 = self.opti.variable(N)
        self.u2 = self.opti.variable(N-1)
        
        q1 = self.q1
        v1 = self.v1
        u1 = self.u1
        q2 = self.q2
        v2 = self.v2
        u2 = self.u2
        
        # Initialization 
        if (X_guess is not None):                           # State
            for i in range(N):  # Checjing defintion of inital state q1,v1,q2,v2
                self.opti.set_initial(q1[i], X_guess[i,0])
                self.opti.set_initial(v1[i], X_guess[i,1]) 
                self.opti.set_initial(q2[i], X_guess[i,2])
                self.opti.set_initial(v2[i], X_guess[i,3]) 
        else:
            for i in range(N):
                self.opti.set_initial(q1[i], x_init[0])
                self.opti.set_initial(v1[i], x_init[1])
                self.opti.set_initial(q2[i], x_init[2])
                self.opti.set_initial(v2[i], x_init[3])
        
        if (U_guess is not None):                           # Input
            for i in range(N-1):
                self.opti.set_initial(u1[i], U_guess[i,0])
                self.opti.set_initial(u2[i], U_guess[i,1])
        
        
        ## ==> Cost function 
        self.cost = 0
        self.running_costs = [None,]*(N)      
        for i in range(N):
            self.running_costs[i] =  self.w_q1 * (q1[i] - conf.q1_target)**2      # On q1 
            self.running_costs[i] += self.w_v1 * v1[i]**2                         
            
            self.running_costs[i] += self.w_q2 * (q2[i] - conf.q2_target)**2      # On q2
            self.running_costs[i] += self.w_v2 * v2[i]**2                         
            
            if (i<N-1):            
                self.running_costs[i] += self.w_u1 * u1[i]**2                     # Torque 1
                self.running_costs[i] += self.w_u2 * u2[i]**2                     # Torque 2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)
        
        ## ==> Constrains
        # Initial state 
        self.opti.subject_to(q1[0] == x_init[0])
        self.opti.subject_to(v1[0] == x_init[1])
        self.opti.subject_to(q2[0] == x_init[2])
        self.opti.subject_to(v2[0] == x_init[3])
        
        for i in range(N):
            # Bounded state
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit1, q1[i], conf.upperPositionLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit1, v1[i], conf.upperVelocityLimit1))
            self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit2, q2[i], conf.upperPositionLimit2))
            self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit2, v2[i], conf.upperVelocityLimit2))
            
            if (i<N-1):
                # Dynamics
                x_plus = dyn.f_double(np.array([q1[i], v1[i], q2[i], v2[i]]), np.array([u1[i], u2[i]]))
                self.opti.subject_to(q1[i+1] == x_plus[0])
                self.opti.subject_to(v1[i+1] == x_plus[1])
                self.opti.subject_to(q2[i+1] == x_plus[2])
                self.opti.subject_to(v2[i+1] == x_plus[3])
                # Bounded Torque
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound1, u1[i], conf.upperControlBound1))
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound2, u2[i], conf.upperControlBound2))
        
        ## ==> Chosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter)}
        self.opti.solver("ipopt", opts, s_opst)
        return self.opti.solve()
    
 



if __name__ == "__main__":
    ocp = OcpDoublePendulum()
    
    # Initial state grid
    nq1 = 10
    nv1 = 10
    nq2 = 10
    nv2 = 10
    state_array, n_ics = conf.grid(nq1, nv1, nq2, nv2)
    
    # State computation function
    def ocp_function_double_pendulum(index):
        viable = []
        no_viable = []
        
        for i in range(index[0], index[1]):
            ######################################################### MISS
            x = state_array[i, :]
            try:
                #sol = ocp.solve(x, conf.N)   sol will not be used!
                ocp.solve(x, conf.N)
                viable.append([x[0], x[1], x[2], x[3]])
                print("Feasible initial state found: [{:.4f}   {:.2f}   {:.4f}   {:.2f}]".format(*x))
            except RuntimeError as e:                     
                if "Infeasible_Problem_Detected" in str(e):
                    print("Non feasible initial state found: [{:.4f}   {:.2f}   {:.4f}   {:.2f}]".format(*x))
                    no_viable.append([x[0], x[1], x[2], x[3]])
                else:
                    print("Runtime error:", e)
        return viable, no_viable
    
    '''
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
    
    
    
    # old
    viable_states_list = []
    no_viable_states_list = []
    for i in range(conf.processor):
        viable_states_list.append(results[i][0])
        no_viable_states_list.append(results[i][1])
    
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

    # Save data in .csv 
    columns = ['q', 'v', 'viable']
    with open('data_single.csv', 'w') as f:
        f.write(','.join(columns) + '\n')
        np.savetxt(f, dataset, delimiter=',', fmt='%s')

    '''

    