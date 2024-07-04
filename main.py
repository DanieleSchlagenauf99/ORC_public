import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Caricamento del dataset
data = pd.read_csv('data_single_test.csv')

# Separazione delle feature e delle etichette
X = data[['x0', 'v0']].values
y = data['label'].values

# Suddivisione del dataset in train e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definizione del modello
model = Sequential([
    Dense(64, input_dim=2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Addestramento del modello
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'Accuracy: {accuracy*100:.2f}%')
