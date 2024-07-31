import numpy as np
import Configuration as conf
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Plot flag: disable to run MPC
PLOT = 0

## ==> NN model creation
def model_creation(nx):   # as input the features's dimension
    # Input layer
    inputs = layers.Input(shape=(nx,))
    # Hidden layers check
    state_out1 = layers.Dense(64, activation="relu")(inputs)
    state_out2 = layers.Dense(32, activation="relu")(state_out1)
    state_out3 = layers.Dense(16, activation="relu")(state_out2)
    # Output layer: sigmoid similar probability 
    outputs = layers.Dense(1, activation='sigmoid')(state_out3)
    #outputs = layers.Dense(1, activation='relu')(state_out3)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)  
    return model


if __name__ == "__main__":
    # Import data 
    data_path = "data100.csv"
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)  
    dataset = data[:, :-1] 
    labels  = data[:, -1]
    scaler = StandardScaler()
    
    # Train and test set creation with scaler 
    train_data, test_data, train_label, test_label = train_test_split(dataset,labels, train_size=conf.train_size, random_state=31)
    train_data = scaler.fit_transform(train_data)
    test_data  = scaler.transform(test_data)
    
    # Export scaler values
    joblib.dump(scaler, 'scaler100.pkl')
    
    
    ## ==> Train and test of NN 
    model = model_creation(nx=conf.ns)   
    model.summary()
   
    # Stop the train if not significant learn in a given # epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=conf.patience, restore_best_weights=True)

    # Train (Could be implemented by Nadam)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= conf.L_rate), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(train_data, train_label, epochs=conf.epochs, validation_data=(test_data, test_label), callbacks=[early_stopping])
    
    # Test
    loss, accuracy = model.evaluate(test_data, test_label)
    end = time.time()
    print('\n')
    conf.print_time(start, end)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    print('\n')
    
    
    ## ==> Export NN weights 
    model.save_weights("single_w.weights.h5", overwrite = True)
    


    ## ==> Check NN Viability kernel 
    # Predicted state, ensuring to be 0 or 1
    data    = np.genfromtxt("data50.csv", delimiter=",", skip_header=1)  
    dataset = data[:, :-1] 
    label   = data[:, -1]
    Norm_dataset = scaler.fit_transform(dataset)
    label_pred   = model.predict(Norm_dataset)
    prediction   = np.round(label_pred).flatten()
    
    
    viable_states    = dataset[prediction == 1.0]
    no_viable_states = dataset[prediction == 0.0]  
    viable_states    = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)
    
    
    ## ==> Plot state space     
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    if len(viable_states) != 0:
        ax.scatter(viable_states[:,0], viable_states[:,1], c='c', label='viable')
        ax.legend()
    if len(no_viable_states) != 0:
        ax.scatter(no_viable_states[:,0], no_viable_states[:,1], c='m', label='non-viable')
        ax.legend()
    ax.set_xlabel('q [rad]')
    ax.set_ylabel('dq [rad/s]')
    plt.show()
   
    
## ==> NN evaluation plot 
if(PLOT):
    # Confusion matrix    
    cm = confusion_matrix(label, prediction)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Viable", "Viable"])
    cm_display.plot(ax=ax, cmap='BuGn')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    # Loss && accuracy  
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='darkgreen')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='darkorange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='b')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='r')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    
    
    
