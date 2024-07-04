import tensorflow as tf 
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

#import mpc_single_pendulum_conf as conf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

import seaborn as sns
import pandas as pd

# collegare URL per NN in cui appare panda 


## NN function creation
def get_critic(nx):
    inputs = layers.Input(shape=(nx,))  
    state_out1 = layers.Dense(16, activation="relu")(inputs)  
    state_out2 = layers.Dense(64, activation="relu")(state_out1)    
    state_out3 = layers.Dense(32, activation="relu")(state_out2)  
    outputs    = layers.Dense(1)(state_out3)  
    model = tf.keras.Model(inputs, outputs)  
    return model  

## Data extractiion and division
dataframe = pd.read_csv("data_single_test.csv")
labels = dataframe['viable']
dataset = dataframe.drop('viable', axis=1)
train_size = 0.8
scaler = StandardScaler()
train_data, test_data, train_label, test_label = train_test_split(dataset, labels, train_size=train_size, random_state=29)

train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


## Computation
if __name__=="__main__":
    NN = get_critic(train_data.shape[1])  #passo 2 dimensioni in input 
    NN.summary() 
    tf.keras.utils.plot_model(NN, to_file='NeuN.png', show_shapes=True)
    
    # categorical instead of binary 
    NN.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])
    
    # add batch_size -> chat not has validiation and callbacks
    history = NN.fit(train_data, train_label, epochs=300)# validation_data=(test_data, test_label))#, callbacks=[early_stopping])

    
    results = NN.evaluate(test_data, test_label)
    print("Test accuracy:", results[1])
    
    NN.save('5a.h5')