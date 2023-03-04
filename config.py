import json
import numpy as np
import optuna
from model.framework_model import MyModel

config_example = {

    "method": "bow",

    "learning_rate": 0.001,

    "batch_size": 16,

    "layers": 3,

    "dropout": 0.3,

}


def objective(trial):
    configs = {

        "method": trial.suggest_categorical("method", ["hierbert", "max_pooling", "bow", "model_01", "model_02"]),

        "learning_rate": 10**trial.suggest_float("lr_exp", -4, -2),

        "batch_size": 2**trial.suggest_int("batch_size_exp", 1, 7),

        "layers": trial.suggest_int("layers", 1, 8),

        "separate": trial.suggest_categorical("separate", [True, False]),

        "dropout": trial.suggest_float("dropout", 0, 0.7),

    }

    # TODO: Interface Dataset and DataLoader

    model = MyModel(method=configs["method"],
                    layers=configs["layers"],
                    separate=configs["separate"],
                    dropout=configs["dropout"])