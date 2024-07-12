import numpy as np
import Configuration as conf
import tensorflow as tf
import matplotlib.pyplot as plt
import time 
import joblib

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay


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
    #outputs = layers.Dense(1, activation='relu')(state_out3)
    outputs = layers.Dense(1, activation='sigmoid')(state_out3)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)  
    return model


if __name__ == "__main__":
    # Import data 
    data_path = "data100.csv"
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)  
    dataset = data[:, :2] 
    labels  = data[:, 2]
    

    # Train and test set creation
    train_data, test_data, train_label, test_label = train_test_split(dataset,labels, train_size=conf.train_size, random_state=29)

    # Scale the set to ensure faster convergence 
    #scaler = StandardScaler()
    #train_data = scaler.fit_transform(train_data)
    #print(f'{scaler.mean_} and {scaler.scale_}')
    #joblib.dump(scaler, 'scalerA100.pkl')            # Export trained scaler 
    #test_data  = scaler.transform(test_data)
    #print(f'{scaler.mean_} and {scaler.scale_}')

    ## ==> Train and test of NN 
    model = model_creation(nx=train_data.shape[1])   # COULD pass the ns form conf
    model.summary()
    
    # print the img with the subbary of the network
    # tf.keras.utils.plot_model(model, to_file='model_summary.png', show_shapes=True)
   
    # Stop the train if not significant learn in a given # epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=conf.patience, restore_best_weights=True)

    # Train 
    #   Check via Nadam also 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= conf.L_rate), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(train_data, train_label, epochs=conf.epochs, validation_data=(test_data, test_label), callbacks=[early_stopping])
    
    # Test
    loss, accuracy = model.evaluate(test_data, test_label)
    end = time.time()
    conf.print_time(start, end)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    
    # Export NN weights 
    model.save_weights("single_w.weights.h5", overwrite = True)
    


    ## ==> Viability 
    # Predicted state, ensuring to be 0 or 1
    #test_data = scaler.fit_transform(test_data)
    #print(f'{scaler.mean_} and {scaler.scale_}')
    #joblib.dump(scaler, 'scaler100.pkl') 
    pred = model.predict(test_data)
    prediction = np.round(pred).flatten()

    '''
    Old version (also add .flatten() on 82)
    viable_states = []
    no_viable_states = []
    for i, label in enumerate(prediction):
        if (label):
            viable_states.append(test_data[i,:])
        else:
            no_viable_states.append(test_data[i,:])
    '''
    viable_states    = test_data[prediction == 1.0]
    no_viable_states = test_data[prediction == 0.0]      
    
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
    cm = confusion_matrix(test_label, prediction)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Viable", "Viable"])
    cm_display.plot(ax=ax, cmap='BuGn')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    
    # ROC curve
    fpr, tpr, _ = roc_curve(test_label, prediction)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='goldenrod', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='purple', lw=2, linestyle='--', label = f'random classificator')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
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

    
    
    
