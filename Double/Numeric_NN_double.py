import numpy as np
import joblib
from NN_double import model_creation
import Configuration_double as conf

## ==> DATA
# Models and weights 
model = model_creation(4)
model.load_weights("w54T.weights.h5")
weights = model.get_weights()

# Scaler
scaler   = joblib.load('scaler54T.pkl')
sc_mean  = scaler.mean_
sc_scale = scaler.scale_

## ==> NN in numpy
def NN_with_sigmoid(params, x):
    out = np.array(x)
    iteration = 0
    for param in params:
        param = np.array(param)
        if iteration % 2 == 0:
            out = param.T @ out  
        else:
            out = param + out  
            if iteration < len(params) - 1:
                out = np.maximum(0., out)  

        iteration += 1
    out = 1 / (1 + np.exp(-out))  
    return out

## ==> STATE DEFINITION 
state = [5/4*np.pi, 0.2, 1.2*np.pi, -0.5]
state_norm = (np.array(state) - sc_mean) / sc_scale

# Numeric value
val = NN_with_sigmoid(weights, state_norm)

print(val)



