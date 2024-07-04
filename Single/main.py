import tensorflow as tf
from tensorflow.keras import layers

# Path al tuo file CSV (relativo)
csv_file_path = 'data_single.csv'

# Definisci i nomi delle colonne e i tipi di dati
column_names = ['q', 'v', 'viable']

# Specifica la proporzione per il training (80%) e il testing (20%)
train_ratio = 0.8

# Carica il dataset dal CSV
batch_size = 32
full_dataset = tf.data.experimental.make_csv_dataset(
    csv_file_path,
    batch_size=batch_size,
    column_names=column_names,
    label_name='viable',
    num_epochs=1,
    shuffle=True,
)
for element in full_dataset:
  print(element)
# Conta il numero totale di campioni nel dataset
num_samples = sum(1 for _ in full_dataset)

# Calcola il numero di campioni per il training e il testing
train_size = int(train_ratio * num_samples)
test_size = num_samples - train_size

print(f'Total samples: {num_samples}, Train samples: {train_size}, Test samples: {test_size}')

# Dividi il dataset in training e test
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)

# Trasformazioni per il training dataset
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

# Trasformazioni per il test dataset
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

def map_features_labels(features, label):
    # Estrai le features 'q' e 'v' e impila in un tensore di forma (batch_size, 2)
    input_features = tf.stack([features['q'], features['v']], axis=-1)
    return input_features, label

# Applica la trasformazione del dataset
train_dataset = train_dataset.map(map_features_labels)
test_dataset = test_dataset.map(map_features_labels)

# Definizione del modello
def model_creation(nx):
    inputs = layers.Input(shape=(nx,), name='input_layer')
    state_out1 = layers.Dense(16, activation='relu')(inputs)
    state_out2 = layers.Dense(32, activation='relu')(state_out1)
    state_out3 = layers.Dense(64, activation='relu')(state_out2)
    state_out4 = layers.Dense(64, activation='relu')(state_out3)
    outputs = layers.Dense(1, activation='sigmoid')(state_out4)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Creazione del modello
model = model_creation(nx=2)
model.summary()

# Compilazione e addestramento del modello
epochs = 10
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=epochs)

# Valutazione del modello sul dataset di test
loss, accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {accuracy}')
