import numpy as np
import joblib
from NN import model_creation

## ==> DATA
# Model and weights
model = model_creation(2)
model.load_weights("w100.weights.h5")
weights = model.get_weights()

# Scaler
scaler   = joblib.load('scaler100.pkl')
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
state = [3/4*np.pi, 0.0]
state_norm = (np.array(state) - sc_mean) / sc_scale

# Numeric value
val = NN_with_sigmoid(weights, state_norm)

print(val)
