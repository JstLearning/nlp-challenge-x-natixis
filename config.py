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

            learning_rate_exp = trial.suggest_float("learning_rate_exp", -6, -1)

            learning_rate_min_exp = trial.suggest_float("learning_rate_min_exp", -10, learning_rate_exp)

            nontext_dim = 2**trial.suggest_int("output_dim_exp", 4, 10)

            separate=False

            corpus_emb_dim = 2**trial.suggest_int("out_features_ce", 5, 10)


            kwargs_nontext = {
                "input_dim": 19,
                "input_channels": 2**trial.suggest_int("input_channels_exp", 4, 8),
                "output_dim": nontext_dim,
                "layers_nontext": trial.suggest_int("layers_nontext", 2, 10),
                "dropout": trial.suggest_float("dropout_nontext", 0.2, 0.5)
            }

            kwargs_ce = {
                "out_features": corpus_emb_dim,
                "nb_layers": trial.suggest_int("nb_layers_ce", 3, 10),
                "dropout": trial.suggest_float("dropout_ce", 0., 0.5)
            }

            kwargs_classification = {
                "corpus_emb_dim": corpus_emb_dim,
                "nontext_dim": nontext_dim,
                "layers": trial.suggest_int("layers", 3, 10),
                "mlp_hidden_dim": 2**trial.suggest_int("mlp_hidden_dim_exp", 4, 10),
                "dropout": trial.suggest_float("dropout_classification", 0.1, 0.6),
                "residual": trial.suggest_categorical("residual", [True, False])
            }

            configs =  {
                  "method": "model_03",
                  "learning_rate": 10**learning_rate_exp,
                  "weight_decay": 10**trial.suggest_float("weight_decay_exp", -15, 0),
                  "batch_size": 16,
                  "separate": separate,
                  "learning_rate_min": 10**learning_rate_min_exp,
                  "max_corpus_len": 3,
                  "max_epochs": 150,
                  "scheduler_step": 15,
                  "scheduler_ratio": 0.2,
                  "scheduler_last_epoch": trial.suggest_int("scheduler_last_epoch", 5, 30),
                  "kwargs_nontext": kwargs_nontext,
                  "kwargs_ce": kwargs_ce,
                  "kwargs_classification": kwargs_classification,
                  "early_stopping": True,
                  "preload": False,
                  "eval_every": 2
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

            model = MyModel(configs["method"],
                            kwargs_nontext, kwargs_classification, kwargs_ce,
                            separate=configs["separate"]).to(device)
            # print(model)
            self.attempts += 1
            name = str(datetime.datetime.today()).replace(':', '-').replace('.', '-') + "-hyperparam-search-with-bn"
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

            
