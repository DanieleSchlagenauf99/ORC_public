import numpy as np
import joblib
from NN import model_creation

# Carica il modello e i pesi
model = model_creation(2)
model.load_weights("w100.weights.h5")
weights = model.get_weights()

# Carica lo scaler
scaler = joblib.load('scaler100.pkl')
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
state = [3.14, 0]
#state = [3/4*np.pi, 0.0]
state_norm = (np.array(state) - sc_mean) / sc_scale

# Calcola il valore usando la rete neurale
val = NN_with_sigmoid(weights, state_norm)

print(val)
