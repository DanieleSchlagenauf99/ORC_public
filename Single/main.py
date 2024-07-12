import numpy as np
import casadi as cas
import matplotlib.pyplot as plt
import Configuration as conf
import Dynamics 
import joblib
from NN import model_creation  

## ==> NB before run ensure the PLOT flag in NN is set  =0 

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
        scaler = joblib.load('scaler100.pkl')
        self.scaler_mean = scaler.mean_
        self.scaler_scale = scaler.scale_
    

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
        stateNorm = (state - self.scaler_mean) / self.scaler_scale
        aa = [stateNorm[0],stateNorm[1]]
        
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

    # Inizializzazione del controllore MPC
    mpc = MpcSinglePendulum()

    # Primo stato iniziale
    initial_state = conf.initial_state

    # Preparazione delle liste per collezionare la traiettoria e gli ingressi
    actual_trajectory = []
    actual_inputs = []

    # Preparazione delle stime iniziali per il primo passo
    new_state_guess = np.tile(initial_state, (conf.N, 1))  # Inizializza con lo stato iniziale ripetuto
    new_input_guess = np.zeros(conf.N - 1)

    # Numero di passi da eseguire (modifica secondo le tue necessità)
    num_steps = 10

    # Esecuzione dei passi successivi
    for t in range(num_steps):
        # Risoluzione del problema di ottimizzazione con le stime attuali
        init_state = actual_trajectory[t]
        sol = mpc.solve(initial_state, conf.N, X_guess=new_state_guess, U_guess=new_input_guess)

        # Aggiornamento della traiettoria e degli ingressi
        next_state = np.array([sol.value(mpc.q[1]), sol.value(mpc.v[1])])
        actual_trajectory.append(next_state)
        actual_inputs.append(sol.value(mpc.u[0]))

        # Preparazione delle nuove stime per il passo successivo
        new_state_guess[:-1, 0] = sol.value(mpc.q[1:]).reshape(-1)
        new_state_guess[:-1, 1] = sol.value(mpc.v[1:]).reshape(-1)

        # Calcolo dell'ultimo stato futuro utilizzando la dinamica del sistema
        last_state = np.array([sol.value(mpc.q[-1]), sol.value(mpc.v[-1])])
        last_input = sol.value(mpc.u[-1])
        next_state = Dynamics.f_single(last_state, last_input)

        # Imposta l'ultimo valore di new_state_guess
        new_state_guess[-1, 0] = next_state[0]
        new_state_guess[-1, 1] = next_state[1]

        # Aggiornamento delle stime degli ingressi
        new_input_guess[:-1] = sol.value(mpc.u[1:]).reshape(-1,1)
        new_input_guess[-1] = sol.value(mpc.u)[-1]  # Ultimo ingresso futuro

        
        



    # Converti le liste in array numpy per un uso successivo
    actual_trajectory = np.array(actual_trajectory)
    actual_inputs = np.array(actual_inputs)

    # ... (altre operazioni eventualmente necessarie)
