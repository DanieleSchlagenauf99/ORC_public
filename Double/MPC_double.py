import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import Configuration as conf
import Dynamics 
import joblib
from NN import model_creation  

## ==> NB before run ensure the PLOT flag in NN is set  =0 
PLOT = 0

class MpcSinglePendulum:
    def __init__(self):
        # Time && Weights
        self.T   = conf.T                   
        self.dt  = conf.dt               

        self.w_q = conf.w_q                 
        self.w_u = conf.w_u                 
        self.w_v = conf.w_v         
                
        # NN
        self.model   = model_creation(conf.ns)        
        self.model.load_weights("single_w.weights.h5")
        self.weights = self.model.get_weights()
        
        # Scaler import   
        #self.scaler = joblib.load('scalerA100.pkl')
        #self.scaler_mean = self.scaler.mean_
        #self.scaler_scale = self.scaler.scale_
    

    ## ==> NN from TensorFlow to casADi 
    def NN_with_sigmoid(self,params, x):
        # param = weights && x = state
        out = np.array(x)
        it = 0
        for param in params:
            param = np.array(param.tolist())
            if it % 2 == 0:
                out = np.transpose(param) @ out  
            else:
                out = param + out  
                if it == (len(params)-1):  
                    out = 1 / (1 + cas.exp(-out))       # Sigmoid function on otput layer 
                else:
                    out = cas.fmax(0., cas.MX(out[0]))  # ReLU function 
            it += 1
        return out

    
    def solve(self, x_init, N, X_guess = None, U_guess = None): 
        self.opti = cas.Opti()
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
                self.opti.set_initial(u[i], U_guess[i])
                
             
        
        ## ==> Running cost 
        self.cost = 0
        self.running_costs = [None,]*(N)      
        for i in range(N):
            self.running_costs[i] =  self.w_q * (q[i] - conf.q_target)**2       # Position 
            self.running_costs[i] += self.w_v * v[i]**2                         # Velocity
            if (i<N-1):            
                self.running_costs[i] += self.w_u * u[i]**2                     # Input 
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)     


        ## ==> Terminal cost 
        state = [q[N-1], v[N-1]]
        #stateNorm = (state - self.scaler_mean) / self.scaler_scale
        aa = [state[0],state[1]]
        
        if conf.TC_on:
            self.opti.subject_to(self.NN_with_sigmoid(self.weights, aa) > 0.0)



        ## ==> Constrains
        # Initial state 
        self.opti.subject_to(q[0] == x_init[0])
        self.opti.subject_to(v[0] == x_init[1])
        
        for i in range(N):
            # Bounded state
            self.opti.subject_to(q[i] <= conf.upperPositionLimit)
            self.opti.subject_to(q[i] >= conf.lowerPositionLimit)
            #self.opti.subject_to(self.opti.bounded(conf.lowerPositionLimit, q[i], conf.upperPositionLimit))
            self.opti.subject_to(v[i] <= conf.upperVelocityLimit)
            self.opti.subject_to(v[i] >= conf.lowerVelocityLimit)
            #self.opti.subject_to(self.opti.bounded(conf.lowerVelocityLimit, v[i], conf.upperVelocityLimit))
            
            if (i<N-1):
                # Dynamics
                x_plus = Dynamics.f_single(np.array([q[i], v[i]]), u[i])
                self.opti.subject_to(q[i+1] == x_plus[0])
                self.opti.subject_to(v[i+1] == x_plus[1])
                # Bounded Torque
                self.opti.subject_to(self.opti.bounded(conf.lowerControlBound, u[i], conf.upperControlBound))
               
        
        ## ==> Chosing solver
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter), "tol": 1e-3}
        self.opti.solver("ipopt", opts, s_opst)
        return self.opti.solve()




if __name__ == "__main__":

    # Instance of OCP solver
    mpc = MpcSinglePendulum()
    
    # First solution 
    initial_state     = conf.initial_state   
    sol= mpc.solve(initial_state, conf.N)
    
    # Empty list to collect each first step 
    actual_trajectory = []                    
    actual_inputs     = []   
    
    # First step computed                 
    actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
    actual_inputs.append(sol.value(mpc.u[0]))
    

    ## ==> Guess 
    new_state_guess = np.zeros((conf.N,2))
    new_input_guess = np.zeros(((conf.N)-1))
    
    new_state_guess[:-1, 0] = sol.value(mpc.q[1:]).reshape(-1)
    new_state_guess[:-1, 1] = sol.value(mpc.v[1:]).reshape(-1)
    new_input_guess[:-1]    = sol.value(mpc.u[1:]).reshape(-1)
    
    # Last values 
    last_state = np.array([sol.value(mpc.q[-1]), sol.value(mpc.v[-1])])
    last_input = sol.value(mpc.u[-1])
    next_state = Dynamics.f_single(last_state, last_input) 
    new_state_guess[-1, 0] = next_state[0]
    new_state_guess[-1, 1] = next_state[1]  
    new_input_guess[-1]    = last_input
       
       
       
    # Update the state and input guesses for the next MPC iteration
    for i in range(conf.mpc_step):
        
        '''
        TO BE CHECK IF IMPLEMNT OR NOT (POSSBILE DISCUSSION IN THE REPORT)
        noise = np.random.normal(conf.mean, conf.std, actual_trajectory[i].shape)
        if conf.noise:
            new_init_state = actual_trajectory[i] + noise
        else:
            new_init_state = actual_trajectory[i]
            
        COULD TRY TO MERGE THE ABOVE CODE 
        '''
        
        # Solve new optimization  
        new_init_state = actual_trajectory[i]
        try:
            sol = mpc.solve(new_init_state, conf.N ,new_state_guess, new_input_guess)
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("\n########################################")
                print(f'At the step {i}')
                print(mpc.opti.debug)
                break
            else:
                print("Runtime error:", e)
                
        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))

        # New guess
        new_state_guess[:-1, 0] = sol.value(mpc.q[1:]).reshape(-1)
        new_state_guess[:-1, 1] = sol.value(mpc.v[1:]).reshape(-1)
        new_input_guess[:-1]    = sol.value(mpc.u[1:]).reshape(-1)
        last_state = np.array([sol.value(mpc.q[-1]), sol.value(mpc.v[-1])])
        last_input = sol.value(mpc.u[-1])
        next_state = Dynamics.f_single(last_state, last_input) 
        new_state_guess[-1, 0] = next_state[0]
        new_state_guess[-1, 1] = next_state[1]  
        new_input_guess[-1]    = last_input
        
        print("Step", i+1, "out of", conf.mpc_step, "done")
    
    
    # Extract positions and velocities from the actual trajectory
    positions = []
    velocities = []
    for i, state in enumerate(actual_trajectory):
        positions.append(actual_trajectory[i][0])
        velocities.append(actual_trajectory[i][1])



   
    ## ==> Compute the viable set using the NN  
    state_array = conf.grid(100,100)[0]
    #to_test     = mpc.scaler.transform(state_array)
    label_pred  = mpc.model.predict(state_array)
    label_pred  = np.round(label_pred)
    
    viable_states = []
    no_viable_states = []
    for i, label in enumerate(label_pred):
        if label:
            viable_states.append(state_array[i,:])
        else:
            no_viable_states.append(state_array[i,:])
    
    viable_states    = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)
    '''
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    ax.scatter(viable_states[:,0], viable_states[:,1], c='r')
    ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='b')
    ax.scatter(positions, velocities, color=(0, 1, 1), s=30)
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()
    '''    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='c', label='viable')
        ax.legend()

    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='m', label='non-viable')
        ax.legend()
    ax.scatter(positions, velocities, color='darkblue', s=30, label='MPC')
    ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    # Torque plot
    if (PLOT == 1):
        fig = plt.figure(figsize=(12,8))
        plt.plot(actual_inputs)
        plt.xlabel('mpc step')
        plt.ylabel('u [N/m]')
        plt.title('Torque')
        plt.show()

        positions = []
        velocities = []

        for element in actual_trajectory:
            positions.append(element[0])
            velocities.append(element[1])

        # Position plot
        fig = plt.figure(figsize=(12,8))
        plt.plot(positions)
        plt.xlabel('mpc step')
        plt.ylabel('q [rad]')
        plt.title('Position')
        plt.show()

        # Velocity plot
        fig = plt.figure(figsize=(12,8))
        plt.plot(velocities)
        plt.xlabel('mpc step')
        plt.ylabel('v [rad/s]')
        plt.title('Velocity')
        plt.show()
    