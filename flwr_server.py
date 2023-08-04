from typing import Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import flwr as fl
import tensorflow as tf


def main() -> None:
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = tf.keras.models.load_model("Best_model.hdf5")
    model.compile("adam","mse",metrics=["mae"])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""
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
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to four local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 4,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to fifteen local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 15
    return {"val_steps": val_steps}


if __name__ == "__main__":
    main()
