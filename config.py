import json
import numpy as np
import optuna
from model.framework_model import MyModel
from model.framework_dataset import get_data_loader

from train import train, evaluate

import torch
import torch.nn as nn

import datetime

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

            learning_rate_exp = trial.suggest_float("lr_exp", -5, -4)

            learning_rate_min_exp = trial.suggest_float("lr_min_exp", -7, learning_rate_exp)

            configs = {

                "method": "model_03",

                "learning_rate": 10**learning_rate_exp,

                "weight_decay": 10**trial.suggest_float("decay", -9, -5),

                "batch_size": 8,# trial.suggest_categorical("batch_size", [8, 16, 24]),

                "layers": trial.suggest_int("layers", 3, 6),

                "mlp_hidden_dim": trial.suggest_categorical("mlp_hidden_dim", [128, 256, 512]),

                "dropout": trial.suggest_float("dropout", 0., 0.4),

                "separate": False,

                "learning_rate_min": 10**learning_rate_min_exp,
                
                "max_corpus_len": 2,

                "max_epochs": 30,

                "scheduler_step": 15,

                "scheduler_ratio": 0.2,

                "scheduler_last_epoch": trial.suggest_int("last_scheduler_epoch", 5, 30),

                "early_stopping": True,

                "preload": False,
                
                "eval_every": 1

            }

            print("Begin trial with configs: \n", configs)


            train_set, train_loader, _, steps = get_data_loader(
            returns_train, ecb, fed, y_train, method=configs["method"],
            separate=configs["separate"], max_corpus_len=configs["max_corpus_len"],
            batch_size=configs["batch_size"]
            )

            val_set, val_loader, _, steps = get_data_loader(
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
            name = str(datetime.datetime.today()).replace(':', '-').replace('.', '-') + "-hyperparam-search"
            eval_losses, eval_accus, eval_f1s = \
                train(model, train_loader=train_loader, val_loader=val_loader,
                  config=configs, device=device, max_epochs=configs["max_epochs"], eval_every=configs["eval_every"],
                  name = name)
            
            with open(f"performances.json", "r") as f:
                perf_dict = json.load(f)
            
            perf_dict[self.attempts] = {
                        "name": name,
                        "config": configs,
                        "eval_losses": eval_losses,
                        "eval_accus": eval_accus,
                        "eval_f1s": eval_f1s
                        }
            
            with open(f"performances.json", "w") as f:
                json.dump(perf_dict, f, indent=6)
            
            return np.min(eval_losses)
        
        self.objective = objective
        self.study = optuna.create_study(direction="minimize")

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)

    def get_best_params(self):
        return self.study.best_params, self.study.best_value

            
