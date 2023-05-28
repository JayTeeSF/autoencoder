# autoencoder

./autoencoder.py --predict hi how # TRAINS and PREDICTS


# TODO: save-ing and load-ing the model between the following two programs fails.
rm trained_model.keras  # clean-up old model (make sure you have a backup in the unused/ directory)

./train_encoder.py # train a new model on the contents of: training_data.json

./run_encoder.py  --load trained_model.keras --predict hi how # load the trained model (trained_model.keras) and use it to predict the next token(s)


