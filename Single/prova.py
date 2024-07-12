import numpy as np 

vettore_originale = np.array([1, 2, 3, 4, 5])
print(vettore_originale)

vettore_col = vettore_originale.reshape(-1, 1)
print(vettore_col)

vettore_riga = vettore_col.reshape(1, -1)
print(vettore_riga)