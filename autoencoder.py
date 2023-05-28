#!/usr/bin/env python
import tensorflow as tf
import argparse
import numpy as np
import json
from tensorflow.keras.optimizers.legacy import Adam

# Define the autoencoder model
input_dim = 10

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

#  autoencoder = tf.keras.models.load_model(model_path)
latent_dim = 5
encoder_inputs = tf.keras.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(latent_dim, activation='relu')(encoder_inputs)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.Model(encoder_inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

# Load training data from a JSON file
training_file_path = 'training_data.json' # Replace with the path to your JSON file
train_data = load_training_data(training_file_path)

# Preprocess the training data
tokenizer, input_data, output_data = preprocess_data(train_data)

# Train the autoencoder
autoencoder.fit(input_data, output_data, epochs=100, batch_size=1)
print('Model trained successfully.')


# Parse command line arguments
parser = argparse.ArgumentParser(description='Autoencoder Loading and Prediction')
parser.add_argument('--predict', nargs='+', help='Make predictions using the loaded model on the given input')
args = parser.parse_args()

if args.predict:
    # Tokenize and pad the input sequence
    input_sequence = args.predict
    input_data = tokenizer.texts_to_sequences(input_sequence)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=input_dim)

    # Make predictions using the loaded model
    output_data = autoencoder.predict(input_data)
    print('Input:', input_data)

    # Convert output probabilities to token indices
    output_indices = np.argmax(output_data, axis=-1)

    # Convert token indices to text
    #output_texts = tokenizer.sequences_to_texts(output_indices)
    ## output_texts = [tokenizer.sequences_to_texts([seq]) for seq in output_indices]
    ## output_texts = [text[0] for text in output_texts]  # Flatten the list of lists
    output_texts = [tokenizer.sequences_to_texts([[index]]) for index in output_indices.flatten()]
    output_texts = [text[0] for text in output_texts]  # Flatten the list of lists

    print('Output:', output_texts)



