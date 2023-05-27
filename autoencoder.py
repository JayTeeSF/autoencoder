#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import json
import argparse
from tensorflow.keras.optimizers import Adam

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

# Parse command line arguments
parser = argparse.ArgumentParser(description='Autoencoder Training and Prediction')
parser.add_argument('--train', action='store_true', help='Train and save the model')
parser.add_argument('--load', type=str, help='Load a saved model from the specified path')
parser.add_argument('--predict', nargs='+', help='Make predictions using the loaded model on the given input')
args = parser.parse_args()

tokenizer = None

if args.train:
    # Load training data from a JSON file
    training_file_path = 'training_data.json'  # Replace with the path to your JSON file
    train_data = load_training_data(training_file_path)

    # Preprocess the training data
    tokenizer, input_data, output_data = preprocess_data(train_data)

    # Train the autoencoder
    autoencoder.fit(input_data, output_data, epochs=100, batch_size=1)

    # Save the trained model
    model_save_path = 'trained_model.keras'  # Replace with the desired save path
    autoencoder.save(model_save_path)
    print('Model saved successfully.')

elif args.load:
    # Load a saved model
    model_path = args.load
    autoencoder = tf.keras.models.load_model(model_path)

    print('Model loaded successfully.')

    if args.predict:
        if tokenizer is None:
            print('Error: No tokenizer found. Please specify --train to train a model or --load to load a trained model.')
        else:
            # Tokenize and pad the input sequence
            input_sequence = args.predict
            input_data = tokenizer.texts_to_sequences(input_sequence)
            input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=input_dim)

            # Make predictions using the loaded model
            output_data = autoencoder.predict(input_data)

            print('Input:', input_data)
            print('Output:', output_data)

else:
    print('Please specify either --train to train a model or --load to load a trained model.')

