#!/usr/bin/env python
import tensorflow as tf
import argparse

input_dim = 10

# Parse command line arguments
parser = argparse.ArgumentParser(description='Autoencoder Loading and Prediction')
parser.add_argument('--load', type=str, help='Load a saved model from the specified path')
parser.add_argument('--predict', nargs='+', help='Make predictions using the loaded model on the given input')
args = parser.parse_args()

if args.load:
    # Load a saved model
    model_path = args.load
    print('Loading model: ', model_path,'...')
    autoencoder = tf.keras.models.load_model(model_path)
    print('Model loaded successfully.')

if args.predict:
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    # Tokenize and pad the input sequence
    input_sequence = args.predict
    input_data = tokenizer.texts_to_sequences(input_sequence)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=input_dim)

    # Make predictions using the loaded model
    output_data = autoencoder.predict(input_data)
    print('Input:', input_data)
    print('Output:', output_data)
