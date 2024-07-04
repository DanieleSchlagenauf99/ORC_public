import numpy as np
import Configuration as conf
import time 
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Plot flag
PLOT = 1

# NN model creation, input the features's dimension 
def model_creation(nx):
    # Input layer
    inputs = layers.Input(shape=(nx,))
    
    # Hidden layers check: order neurons and last activation relu vs sigmoid
    state_out1 = layers.Dense(64, activation="relu")(inputs)
    state_out2 = layers.Dense(32, activation="relu")(state_out1)
    state_out3 = layers.Dense(16, activation="relu")(state_out2)

    # Output layer: sigmoid similar probability 
    outputs = layers.Dense(1, activation='sigmoid')(state_out3)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)  
    return model


if __name__ == "__main__":
    # Data extraction
    data_path = "data_single.csv"
    data = np.genfromtxt(data_path, delimiter=",", skip_header=1)  
    dataset = data[:, :2] 
    labels  = data[:, 2]
    
    # make sure all your features have similar scale before using them as inputs to your neural network. This ensures faster convergence.
    scaler = StandardScaler()

    # Train and test  
    train_data, test_data, train_label, test_label = train_test_split(dataset,labels, train_size=conf.train_size, random_state=29)
    train_data = scaler.fit_transform(train_data)
    test_data  = scaler.transform(test_data)


    model = model_creation(nx=train_data.shape[1])
    model.summary()
    
    # print the img with the subbary of the network
    # tf.keras.utils.plot_model(model, to_file='model_summary.png', show_shapes=True)
   
    # Stop the train if not significant increase in a given # epochs
    early_stopping = EarlyStopping(monitor='val_loss', patience=conf.patience, restore_best_weights=True)

    # Train and test of the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    history = model.fit(train_data, train_label, epochs=conf.epochs, validation_data=(test_data, test_label), callbacks=[early_stopping])
    loss, accuracy = model.evaluate(test_data, test_label)
    end = time.time()
    conf.print_time(start, end)
    print(f'Loss: {loss}, Accuracy: {accuracy}')
    
    # Save weights 
    model.save_weights("single_w.weights.h5")



    ## NN state space viability 
    pred = model.predict(test_data)
    prediction = np.round(pred)

    viable_states = []
    no_viable_states = []
    for i, label in enumerate(prediction):
        if (label):
            viable_states.append(test_data[i,:])
        else:
            no_viable_states.append(test_data[i,:])
    
    # Plot     
    viable_states    = np.array(viable_states)
    no_viable_states = np.array(no_viable_states)
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
    
    
if(PLOT):
    # Confusion matrix
    cm = confusion_matrix(test_label, prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', xticklabels=['Non-Viable', 'Viable'], yticklabels=['Non-Viable', 'Viable'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


    # ROC curve
    fpr, tpr, _ = roc_curve(test_label, prediction)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='goldenrod', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='purple', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()


    # Loss && accuracy  
    plt.figure(figsize=(12, 8))
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

    
    
