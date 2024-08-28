import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import Configuration_double as conf
import Dynamics_double as dyn
import joblib
import time

from NN_double import model_creation  
from matplotlib import animation

# Set print options
np.set_printoptions(precision=3, linewidth=200, suppress=True)

## ==> NB before run ensure the PLOT flag in NN is set  =0 
PLOT = 1
Animation = 1

class MpcDoublePendulum:
    def __init__(self):
        # Time 
        self.T   = conf.T                   
        self.dt  = conf.dt               
        # Weights 
        self.w_q1 = conf.w_q1               
        self.w_u1 = conf.w_u1               
        self.w_v1 = conf.w_v1
        self.w_q2 = conf.w_q2                 
        self.w_u2 = conf.w_u2                 
        self.w_v2 = conf.w_v2         
                
        # Creation of NN with the computed weights
        self.model   = model_creation(conf.ns)        
        self.model.load_weights("w54T.weights.h5")
        self.weights = self.model.get_weights()
        
        # Scaler    
        self.scaler   = joblib.load('scaler54T.pkl')
        self.sc_mean  = self.scaler.mean_
        self.sc_scale = self.scaler.scale_
    

    ## ==> TENSORFLOW TO CASADI
    def NN_with_sigmoid(self,params, x):
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
            self.running_costs[i]  =  self.w_q1 * (q1[i] - conf.q1_target)**2  # q1 
            self.running_costs[i] += self.w_v1 * v1[i]**2                      # dq1
            self.running_costs[i] +=  self.w_q2 * (q2[i] - conf.q2_target)**2  # q2 
            self.running_costs[i] += self.w_v2 * v2[i]**2                      # dq2
            if (i<N-1):       
                self.running_costs[i] += self.w_u1 * u1[i]**2                  # u1
                self.running_costs[i] += self.w_u2 * u2[i]**2                  # u2 
            self.cost += self.running_costs[i]
            
        self.opti.minimize(self.cost)     

        ## ==> CONSTRAINS
        # Terminal constrains using the NN
        state = [q1[N-1], v1[N-1],q2[N-1], v2[N-1]]
        state_norm = (state - self.sc_mean) / self.sc_scale                         # Manual application of scaling on the final state
        state_norm = [state_norm[0], state_norm[1], state_norm[2], state_norm[3]] 
        state_mx   = cas.vertcat(*state_norm)                                       # Transormation in casADi format  
        
        if conf.TC_on:
            nn_output = self.NN_with_sigmoid(self.weights, state_mx)
            self.opti.subject_to(nn_output >= conf.TC_limit)


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
        opts   = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        s_opst = {"max_iter": int(conf.iter), "tol": 1e-3}
        self.opti.solver("ipopt", opts, s_opst)
        return self.opti.solve()




if __name__ == "__main__":

    # MPC problem
    mpc = MpcDoublePendulum()
    
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
                new_input_guess = np.zeros(((conf.N)-1, conf.nu))
                
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
        actual_trajectory.append(np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]),sol.value(mpc.q2[1]), sol.value(mpc.v2[1])]))
        actual_inputs.append(np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])]))

        ## ==> NEW GUESS
        # Copy in the guess array the obtained solutions, except for the first step 
        new_state_guess[:-1, 0] = sol.value(mpc.q1[1:]).reshape(-1)
        new_state_guess[:-1, 1] = sol.value(mpc.v1[1:]).reshape(-1)
        new_state_guess[:-1, 2] = sol.value(mpc.q2[1:]).reshape(-1)
        new_state_guess[:-1, 3] = sol.value(mpc.v2[1:]).reshape(-1)
        new_input_guess[:-1, 0] = sol.value(mpc.u1[1:]).reshape(-1)
        new_input_guess[:-1, 1] = sol.value(mpc.u2[1:]).reshape(-1)
        
        # The last guess's element is computed by means of the dynmaics 
        last_state = np.array([sol.value(mpc.q1[1]), sol.value(mpc.v1[1]),sol.value(mpc.q2[1]), sol.value(mpc.v2[1])])
        last_input = np.array([sol.value(mpc.u1[0]), sol.value(mpc.u2[0])])
        next_state = dyn.f_double(last_state, last_input) 
        
        new_state_guess[-1, 0] = next_state[0]
        new_state_guess[-1, 1] = next_state[1] 
        new_state_guess[-1, 2] = next_state[2]
        new_state_guess[-1, 3] = next_state[3] 
        #  copy last inputs in the last postion 
        new_input_guess[-1, 0] = last_input[0]     
        new_input_guess[-1, 1] = last_input[1]    
        
        print(f'Step: {i+1} / {conf.mpc_step}')
    
    end = time.time()
    conf.print_time(start, end)
    
    # Extract positions and velocities from the actual trajectory 
    positions1  = [actual_trajectory[i][0] for i in range(len(actual_trajectory))] 
    velocities1 = [actual_trajectory[i][1] for i in range(len(actual_trajectory))]   
    positions2  = [actual_trajectory[i][2] for i in range(len(actual_trajectory))] 
    velocities2 = [actual_trajectory[i][3] for i in range(len(actual_trajectory))]    

    
    if (PLOT == 1):
        # Torque plot      
        first  = [arr[0] for arr in actual_inputs]
        second = [arr[1] for arr in actual_inputs]
        plt.figure(1, figsize=(12,8))
        plt.plot(first, c='red',   label='First pendulum')
        plt.plot(second, c='green', label='Second pendulum')
        plt.xlabel('mcp step')
        plt.ylabel('Torques [N/m]')
        plt.title('Torque evolutions')
        plt.legend()  
        plt.show()
        
        # First pendulum plot
        plt.figure(2)
        plt.plot(positions1, velocities1, c='red')
        plt.xlabel('q1 [rad]')
        plt.ylabel('v1 [rad/s]')
        plt.title('First pendulum')
        
        # Second pendulum plot
        plt.figure(3)
        plt.plot(positions2, velocities2, c='green')
        plt.xlabel('q2 [rad]')
        plt.ylabel('v2 [rad/s]')
        plt.title('Second pendulum')
        plt.show()
        
        
        
    if (Animation == 1):
        # Cartesian coordinate,both the1 and the2 has the same zero reference 
        def get_x1y1x2y2(the1, the2, L1, L2):
            return (L1 * np.sin(the1),
                    -L1 * np.cos(the1),
                    L1 * np.sin(the1) + L2 * np.sin(the2),
                    -L1 * np.cos(the1) - L2 * np.cos(the2))

        
        q1 = [arr[0] for arr in actual_trajectory]
        q2 = [arr[2] for arr in actual_trajectory]
        x1, y1, x2, y2 = get_x1y1x2y2(q1, q2, 1, 1)

        # Animation
        def animate(i):
            ln1.set_data([0, x1[i]], [0, y1[i]])
            ln2.set_data([x1[i], x2[i]], [y1[i], y2[i]])

            # Update the cone for the second pendulum
            cone2_1.set_data([x1[i], x1[i] + np.sin(conf.lowerPositionLimit2)], 
                             [y1[i], y1[i] - np.cos(conf.lowerPositionLimit2)])
            cone2_2.set_data([x1[i], x1[i] + np.sin(conf.upperPositionLimit2)], 
                             [y1[i], y1[i] - np.cos(conf.upperPositionLimit2)])
            cone2_3.set_data([x1[i], x1[i]], 
                             [y1[i], y1[i] +1])
            
            step_text.set_text(f'mcp_step: {i+1}')

        fig, ax = plt.subplots(1, 1)
        ax.set_facecolor('k')
        ax.get_xaxis().set_ticks([])  
        ax.get_yaxis().set_ticks([])  
        ln1, = ax.plot([], [], 'rh-', lw=5, markersize=12)
        ln2, = ax.plot([], [], 'gh-', lw=5, markersize=12)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlim(-2.5, 2.5)


        # First limit cone plot
        ax.plot([0, np.sin(conf.lowerPositionLimit1)], [0, -np.cos(conf.lowerPositionLimit1)], 'w-', lw=2)
        ax.plot([0, np.sin(conf.upperPositionLimit1)], [0, -np.cos(conf.upperPositionLimit1)], 'w-', lw=2)
        ax.plot([0, 0], [0, 1], 'c--', lw=1)

        # Second limit cone plot (empty then filled at each iter)
        cone2_1, = ax.plot([], [], 'w-', lw=2)
        cone2_2, = ax.plot([], [], 'w-', lw=2)
        cone2_3, = ax.plot([], [], 'c--', lw=1)
        
        # Write number of step 
        step_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, color='white', fontsize=15)
        
        # Animation
        ani = animation.FuncAnimation(fig, animate, frames=len(x1), interval=50)
        plt.show()
        #ani.save('Newgif.gif', writer='pillow')

