#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.optimizers.legacy import Adam

# Define the autoencoder model
input_dim = 10
latent_dim = 5

encoder_inputs = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(encoder_inputs)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(encoder_inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Function to load training data from a JSON file
def load_training_data(file_path):
    with open(file_path) as file:
        data = json.load(file)
    return data

# Function to preprocess the training data
def preprocess_data(train_data):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_data)
    input_sequences = tokenizer.texts_to_sequences(train_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=input_dim)
    output_data = np.roll(input_data, -1, axis=1)
    output_data[:, -1] = 0
    return tokenizer, input_data, output_data

# Load training data from a JSON file
training_file_path = 'training_data.json' # Replace with the path to your JSON file
train_data = load_training_data(training_file_path)

# Preprocess the training data
tokenizer, input_data, output_data = preprocess_data(train_data)

# Train the autoencoder
autoencoder.fit(input_data, output_data, epochs=100, batch_size=1)

# Save the trained model
model_save_path = 'trained_model.keras' # Replace with the desired save path
autoencoder.save(model_save_path)
print('Model saved successfully.')
