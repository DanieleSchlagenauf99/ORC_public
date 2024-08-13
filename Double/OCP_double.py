import numpy as np
import casadi as cas
import Dynamics_double as dyn
import Configuration_double as conf

import multiprocessing
import time
import os              # for file interaction 

# Set print options
np.set_printoptions(precision=3, linewidth=200, suppress=True)

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
        
        # Initialization: x = [q1, v1, q2, v2] && u = [u1, u2]
        if (X_guess is not None):                           # State
            for i in range(N):  
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
        
        
        ## ==> COST FUNCTION 
        self.cost = 0
        self.running_costs = [None,]*(N)      
        for i in range(N):
            self.running_costs[i] = self.w_v1 * v1[i]**2                         
            self.running_costs[i] += self.w_v2 * v2[i]**2                         
            if (i<N-1):            
                self.running_costs[i] += self.w_u1 * u1[i]**2                     # Torque 1
                self.running_costs[i] += self.w_u2 * u2[i]**2                     # Torque 2
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)
        
        
        ## ==> CONSTRAINS
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
        
        
        ## ==> SOLVER
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter)}
        self.opti.solver("ipopt", opts, s_opst)
        
        return self.opti.solve()
    
 



if __name__ == "__main__":
    ocp = OcpDoublePendulum()
    
    # Selection of sampling method: random allows a better way to populate the 
    # dataset, making it harder to take the same data multiple times     
    if (conf.rnd_augment == 1):
        n_ics = conf.n_rnd
        state_array = conf.random(n_ics)
    else: 
        # Initial state grid
        nq1, nv1 = 10, 10
        nq2, nv2 = 10, 10
        state_array, n_ics = conf.grid(nq1, nv1, nq2, nv2)
    
    ## ==> AUGMENTATION OF THE DATASET:
    # If dataset is already build, avoid to repeat same data and add only new I.C
    data_path = "data_11.csv"     # Select the file from the current direcotry  
    if (os.path.exists(data_path)):  
        data = np.genfromtxt(data_path, delimiter=",", skip_header=1) 
        old_data     = data[:,:-1]                # Extract all known points exept for the last column (viability) 
        known_points = set(map(tuple,old_data))   # Set as tuple to be unchangeable
        
        # New state array: created by insert all data which are not in known_points
        state_array = np.array([point for point in state_array if tuple(point) not in known_points])
        n_ics = state_array.shape[0]   # new value of n_ics
    
    
    # State computation function
    def ocp_function_double_pendulum(index):
        viable,no_viable = [],[]
        k = 0                            # Actual step for single process
        l = int(n_ics / conf.processor)  # Total number of step for single process
        # Select data 
        for i in range(index[0], index[1]):
            k = k + 1
            x = state_array[i, :]
            try:  
                sol = ocp.solve(x, conf.N)
                viable.append([x[0], x[1], x[2], x[3]])
                print(f'Step: {k} / {l}, Feasible initial state found: {x}')
            except RuntimeError as e:                     
                if "Infeasible_Problem_Detected" in str(e):
                    print(f'Step: {k} / {l}, Non feasible initial state found: {x}')
                    no_viable.append([x[0], x[1], x[2], x[3]])
                else:
                    print(f'Runtime error: {e}')
        return viable, no_viable
    
    
    ## ==> MULTIPROCESS  
    # Split same number of operation in each processor 
    indexes = np.linspace(0, n_ics, num=conf.processor+1)
    args = []
    for i in range(conf.processor):
        args.append((int(indexes[i]), int(indexes[i+1])))
        
    # Pool logic application 
    pool    = multiprocessing.Pool(processes=conf.processor)
    start   = time.time()
    results = pool.map(ocp_function_double_pendulum, args)  # first colum viable, second colum non viable
    pool.close()
    pool.join()
    
    # Print duration 
    end = time.time()
    conf.print_time(start, end)
    
    # Join the result from each processor
    viable_states    = [results[i][0] for i in range(conf.processor)]
    no_viable_states = [results[i][1] for i in range(conf.processor)]
    viable_states    = np.concatenate(viable_states)
    no_viable_states = np.concatenate(no_viable_states)

    
    ## ==> DATASET CREATION 
    # The dataset is created by concatenate all the state marking with 1 the viable and 0 the non viable 
    viable_states    = np.column_stack((viable_states, np.ones(len(viable_states), dtype=int)))
    no_viable_states = np.column_stack((no_viable_states, np.zeros(len(no_viable_states), dtype=int)))
    
    if (viable_states.shape[0]==0):
        dataset = no_viable_states    
    else:
        dataset = np.concatenate((viable_states, no_viable_states))


    # Open the file with the a (append) options allow to append data if file exist or
    # create a new file if it doesn't   
    file_exist = os.path.exists(data_path)
    columns    = ['q1', 'v1', 'q2', 'v2', 'viable']
    with open(data_path, 'a') as f:
        if not file_exist:                      # If not exist insert columns header
            f.write(','.join(columns) + '\n')
        np.savetxt(f, dataset, delimiter=',', fmt='%s')



