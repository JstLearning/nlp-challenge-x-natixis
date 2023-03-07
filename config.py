import json
import numpy as np
import optuna
from model.framework_model import MyModel
from model.framework_dataset import get_data_loader

from train import train, evaluate

import torch
import torch.nn as nn

config_dummy = {

    "method": "model_01",

    "learning_rate": 0.001,

    "weight_decay": 0.,

    "batch_size": 2,

    "layers": 3,

    "dropout": 0.5,

    "separate": True,
    
    "max_corpus_len": 2

}

class Optimizer(object):
    def __init__(self, returns_train, returns_val, returns_test,
              y_train, y_val, y_test, ecb, fed, n_trials=1):
        
        self.returns_train = returns_train
        self.returns_val = returns_val
        self.returns_test = returns_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.ecb = ecb
        self.fed = fed

        self.attempts = 0
        self.n_trials = n_trials

        def objective(trial):
            configs = {

                "method": trial.suggest_categorical("method", ["model_01", "model_02", "model_03"]),

                "learning_rate": 10**trial.suggest_float("lr_exp", -4, -2),

                "weight_decay": 10**trial.suggest_float("weight_decay_exp", -5, 5),

                "batch_size": 2**trial.suggest_int("batch_size_exp", 1, 4),

                "layers": trial.suggest_int("layers", 1, 8),

                "separate": trial.suggest_categorical("separate", [True, False]),

                "dropout": trial.suggest_float("dropout", 0.2, 0.7),

            }


            train_set, train_loader, tokenizer, steps = get_data_loader(
            returns_train, ecb, fed, y_train, method=configs["method"],
            separate=configs["separate"], max_corpus_len=configs["max_corpus_len"],
            batch_size=configs["batch_size"]
            )

            val_set, val_loader, tokenizer, steps = get_data_loader(
                returns_val, ecb, fed, y_val, method=configs["method"],
                separate=configs["separate"], max_corpus_len=configs["max_corpus_len"],
                batch_size=configs["batch_size"]
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = MyModel(method=configs["method"],
                            layers=configs["layers"],
                            separate=configs["separate"],
                            dropout=configs["dropout"]).to(device)
            self.attempts += 1
            eval_losses, eval_accus, eval_f1s = \
                train(model, train_loader=train_loader, val_loader=val_loader,
                  config=config, device=device, max_epochs=25, eval_every=1,
                  name = str(self.attempts))
            
            with open(f"attempt_{self.attempts}.json", "w") as f:
                json.dump({
                    "config": config,
                    "eval_losses": eval_losses,
                    "eval_accus": eval_accus,
                    "eval_f1s": eval_f1s
                }, f)
            
            return eval_accus[-1]
        
        self.objective = objective
        self.study = optuna.create_study(direction="maximize")

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)

    def get_best_params(self):
        return self.study.best_params, self.study.best_value

            
