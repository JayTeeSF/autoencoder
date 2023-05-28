# autoencoder

rm trained_model.keras  # clean-up old model (make sure you have a backup in the unused/ directory)
./train_encoder.py # train a new model on the training_data.json
./autoencoder.py --load trained_model.keras --predict hi how # attempt to predict

