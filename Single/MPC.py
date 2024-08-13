import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import Configuration as conf
import Dynamics 
import joblib
import time
from NN import model_creation 
from matplotlib import animation

# Set print options
np.set_printoptions(precision=3, linewidth=200, suppress=True)

## ==> NB before run this code set all flags in NN =0 
PLOT = 0
Animation = 0

class MpcSinglePendulum:
    def __init__(self):
        # Time 
        self.T   = conf.T                   
        self.dt  = conf.dt  
        # Weights             
        self.w_q = conf.w_q                 
        self.w_u = conf.w_u                 
        self.w_v = conf.w_v         
                
        # Creation of NN with the computed weights
        self.model   = model_creation(conf.ns)        
        self.model.load_weights("w100.weights.h5")
        self.weights = self.model.get_weights()
        
        # Scaler 
        self.scaler   = joblib.load('scaler100.pkl')
        self.sc_mean  = self.scaler.mean_
        self.sc_scale = self.scaler.scale_


    ## ==> TENSORFLOW TO CASADI
    def NN_with_sigmoid(self, params, x):
        out = cas.MX(x)  
        iteration = 0
        for param in params:
            param = cas.MX(np.array(param.tolist()))   
            if iteration % 2 == 0:
                out = param.T @ out               # Linear layer  
            else:
                out = param + out                 # Add bias
                if iteration < len(params) - 1:
                    out = cas.fmax(0., out)       # ReLU 
    
            iteration += 1
        out = 1 / (1 + cas.exp(-out))             # Sigmoid 
        return out


    def solve(self, x_init, N, X_guess = None, U_guess = None): 
        self.opti = cas.Opti()
        
        # Casadi variables
        self.q    = self.opti.variable(N)       
        self.v    = self.opti.variable(N)
        self.u    = self.opti.variable(N-1)
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
                
             
        
        ## ==> COST FUNCTION  
        self.cost = 0
        self.running_costs = [None,]*(N)      
        for i in range(N):
            self.running_costs[i] =  self.w_q * (q[i] - conf.q_target)**2  
            self.running_costs[i] += self.w_v * v[i]**2                    
            if (i<N-1):       
                self.running_costs[i] += self.w_u * u[i]**2                 
            self.cost += self.running_costs[i]
        self.opti.minimize(self.cost)     

        ## ==> CONSTRAINS
        # Terminal constrains using the NN 
        state      = [q[N-1], v[N-1]] 
        state_norm = (state - self.sc_mean) / self.sc_scale  # Manual application of scaling on the final state
        state_norm = [state_norm[0], state_norm[1]] 
        state_nn   = cas.vertcat(*state_norm)                # transormation in casADi format  
        if conf.TC_on:
            nn_output = self.NN_with_sigmoid(self.weights, state_nn)
            self.opti.subject_to(nn_output >= conf.TC_limit)
        
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
               
        
        ## ==> SOLVER
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter), 'tol' : 1e-3}
        self.opti.solver("ipopt", opts, s_opst)
        return self.opti.solve()




if __name__ == "__main__":

    # MPC problem
    mpc = MpcSinglePendulum()
    
    # Array of actual trajectory, starting from I.C
    actual_trajectory, actual_inputs = [], []
    actual_trajectory.append(conf.initial_state)
         
    start = time.time()
    # MPC iteration 
    for i in range(conf.mpc_step):       
        new_init_state = actual_trajectory[i]
        try:
            # First step, without any guess
            if (i == 0):  
                sol =  mpc.solve(new_init_state, conf.N) 
                # Define the guess array 
                new_state_guess = np.zeros((conf.N,conf.ns))
                new_input_guess = np.zeros(((conf.N)-1))
            
            # With guess
            else:
                sol = mpc.solve(new_init_state, conf.N ,new_state_guess, new_input_guess)
        
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print(f'\n########################################')
                print(mpc.opti.debug.show_infeasibilities())
                print(f'At the step {i}')
                print(mpc.opti.debug)
                break
            else:
                print(f'Runtime error: {e}')
                break
        
        # Append first step 
        actual_trajectory.append(np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])]))
        actual_inputs.append(sol.value(mpc.u[0]))

        ## ==> NEW GUESS
        # Copy in the guess array the obtained solutions, except for the first step 
        new_state_guess[:-1, 0] = sol.value(mpc.q[1:]).reshape(-1) 
        new_state_guess[:-1, 1] = sol.value(mpc.v[1:]).reshape(-1)
        new_input_guess[:-1]    = sol.value(mpc.u[1:]).reshape(-1)
        
        # The last guess's element is computed by means of the dynmaics  
        last_state = np.array([sol.value(mpc.q[-1]), sol.value(mpc.v[-1])])
        last_input = sol.value(mpc.u[-1])
        next_state = Dynamics.f_single(last_state, last_input) 
        
        new_state_guess[-1, 0] = next_state[0]
        new_state_guess[-1, 1] = next_state[1]  
        new_input_guess[-1]    = last_input     # copy last input in the last postion 
        
        print(f'Step: {i+1} /  {conf.mpc_step}')
    
    end = time.time()
    
    conf.print_time(start, end)
    # Extract positions and velocities from the actual trajectory    
    positions  = [actual_trajectory[i][0] for i in range(len(actual_trajectory))] 
    velocities = [actual_trajectory[i][1] for i in range(len(actual_trajectory))] 
        

    ## ==> PLOT MPC STEP 
    # Compute viability kernel  
    dataset = conf.grid(100,100)[0]
    Norm_dataset     = mpc.scaler.fit_transform(dataset)
    label_pred       = mpc.model.predict(Norm_dataset)
    prediction       = np.round(label_pred).flatten()
    viable_states    = dataset[prediction == 1.0]
    no_viable_states = dataset[prediction == 0.0]  
    viable_states    = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='c', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='m', label='non-viable')
        ax.legend()
    ax.scatter(positions, velocities, color='black', label='MPC')
    ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()

    
    if (PLOT == 1):
        # Torque plot
        fig = plt.figure(figsize=(12,8))
        plt.plot(actual_inputs, lw=3, c='darkgreen')
        plt.xlabel('MPC step')
        plt.ylabel('u [N/m]')
        plt.title('Torque')

        # Position plot
        fig = plt.figure(figsize=(12,8))
        plt.plot(positions, lw=3, c='darkgreen')
        plt.xlabel('MPC step')
        plt.ylabel('q [rad]')
        plt.title('Position')

 
        # Velocity plot
        fig = plt.figure(figsize=(12,8))
        plt.plot(velocities, lw=3, c='darkgreen')
        plt.xlabel('MPC step')
        plt.ylabel('v [rad/s]')
        plt.title('Velocity')


    if (Animation == 1):
        # Cartesian coordinate 
        def get_x1y1(the1, L1):
            return (L1 * np.sin(the1),
                    -L1 * np.cos(the1))
            
        q = [arr[0] for arr in actual_trajectory]
        x1, y1 = get_x1y1(q, 1)
        
        # Animation
        def animate(i):
            ln1.set_data([0, x1[i]], [0, y1[i]])
        
        fig, ax = plt.subplots(1, 1)
        ax.set_facecolor('k')
        ax.get_xaxis().set_ticks([])  
        ax.get_yaxis().set_ticks([])  
        ln1, = ax.plot([], [], 'rh-', lw=5, markersize=12)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlim(-1.5, 1.5)
        
        # Lines
        ax.plot([0,  np.sin(conf.lowerPositionLimit)], [0, -np.cos(conf.lowerPositionLimit)], 'w-', lw=2)
        ax.plot([0,  np.sin(conf.upperPositionLimit)], [0, -np.cos(conf.upperPositionLimit)], 'w-', lw=2)
        ax.plot([0,0], [0,1], 'c--', lw=1)

        # Animation
        ani = animation.FuncAnimation(fig, animate, frames=len(x1), interval=50)
        plt.show()
        #ani.save('MPC_noTC.gif', writer='pillow')
    