import numpy as np
import joblib
from NN_double import model_creation
import Configuration_double as conf

# Carica il modello e i pesi
model = model_creation(4)
model.load_weights("w32.weights.h5")
weights = model.get_weights()

# Carica lo scaler
scaler = joblib.load('scaler32.pkl')
sc_mean = scaler.mean_
sc_scale = scaler.scale_

# Definisci la funzione NN_with_sigmoid usando solo NumPy
def NN_with_sigmoid(params, x):
    out = np.array(x)
    iteration = 0
    for param in params:
        param = np.array(param)
        if iteration % 2 == 0:
            out = param.T @ out  # Strato lineare
        else:
            out = param + out  # Aggiungi il bias
            if iteration < len(params) - 1:
                out = np.maximum(0., out)  # ReLU

        iteration += 1
    out = 1 / (1 + np.exp(-out))  # Sigmoid
    return out

# Definisci lo stato e normalizzalo
#state= conf.initial_state
#state = [3/4*np.pi, 0.0, 7/8*np.pi, 0.0] 0.996
#state = [3/4*np.pi, 0.0, 5/4*np.pi, 0.0]  0.855
state = [0.78*np.pi, -1.0, 0.78*np.pi, 0.0]
state_norm = (np.array(state) - sc_mean) / sc_scale

# Calcola il valore usando la rete neurale
val = NN_with_sigmoid(weights, state_norm)

print(val)



