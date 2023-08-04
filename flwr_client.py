import argparse
import os
from pathlib import Path
import pandas as pd
import tensorflow as tf

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.evaluate_metrics_aggregation_fn = lambda metrics: {
            "accuracy": metrics["accuracy"]
        }


    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}


def main() -> None:
    
    # Load and compile Keras model
    model = tf.keras.models.load_model("Best_model.hdf5")
    model.compile("adam", "mse", metrics=["accuracy"])

    df_model= pd.read_csv("lstmtrain.csv")
    df_model = df_model.sort_values("date").reset_index(drop = True)
    train= df_model.loc[(df_model["date"] <  "2017-01-01"), :]
    test=df_model.loc[(df_model["date"] >= "2017-01-01") & (df_model["date"] < "2017-04-01"), :]
    colms = [colm for colm in train.columns if colm not in ['date', 'id', "sales", "year"]]
    X_train=train[colms]
    Y_train=train['sales']
    X_val=train[colms]
    Y_val=train['sales']
    X_train_array = X_train.values
    Y_train_array = Y_train.values
    X_val_array = X_val.values
    Y_val_array = Y_val.values
    
    # Reshape X_train_array to fit the LSTM input shape (samples, time steps, features)
    time_steps = 1
    num_features = X_train_array.shape[1]
    x_train_reshaped = X_train_array.reshape((X_train_array.shape[0], time_steps, num_features))
    x_train=x_train_reshaped
    y_train=Y_train_array
    X_val_reshaped=X_val_array.reshape((X_val_array.shape[0], time_steps, num_features))
    x_val=X_val_reshaped
    y_val=Y_val_array


    

    # Start Flower client
    client = Client(model, x_train, y_train, x_val, y_val)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        
    )



if __name__ == "__main__":
    main()
