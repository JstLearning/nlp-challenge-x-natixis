import pandas as pd
import numpy as np
import json

from model.mlp import SimpleMLP

from sklearn.model_selection import train_test_split

from model.framework_model import MyModel
from model.framework_dataset import get_data_loader
from train import train, evaluate

# from preprocessing.preprocessing import ecb_pipeline_en, fast_detect
from preprocessing.standard_scaler import get_column_transformer
from preprocessing.outlier_detection import remove_outlier

from config import Optimizer

import torch
import torch.nn as nn

import datetime

learning_rate_exp = -4

learning_rate_min_exp = -7

nontext_dim = 2**7

separate=False

corpus_emb_dim = 2**7


kwargs_nontext = {
    "input_dim": 19,
    "input_channels": 2**4,
    "output_dim": nontext_dim,
    "layers_nontext": 7,
    "dropout": 0.4
}

kwargs_ce = {
    "out_features": corpus_emb_dim,
    "nb_layers": 5,
    "dropout": 0
}

kwargs_classification = {
    "corpus_emb_dim": corpus_emb_dim,
    "nontext_dim": nontext_dim,
    "layers": 8,
    "mlp_hidden_dim": 2**8,
    "dropout": 0.2,
    "residual": False
}

config =  {
        "method": "model_03",
        "learning_rate": 10**learning_rate_exp,
        "weight_decay": 0,
        "batch_size": 16,
        "separate": separate,
        "learning_rate_min": 10**learning_rate_min_exp,
        "max_corpus_len": 2,
        "max_epochs": 200,
        "scheduler_step": 15,
        "scheduler_ratio": 0.2,
        "scheduler_last_epoch": 15,
        "kwargs_nontext": kwargs_nontext,
        "kwargs_ce": kwargs_ce,
        "kwargs_classification": kwargs_classification,
        "early_stopping": False,
        "preload": False,
        "eval_every": 1
}

def main():
    # Load data

    FILENAME = "data/train_series.csv"
    FILENAME_ECB = "data/ecb_data_preprocessed.csv"
    FILENAME_FED = "data/fed_data_preprocessed.csv"

    returns = pd.read_csv(FILENAME, index_col=0)
    ecb = pd.read_csv(FILENAME_ECB, index_col=0)
    fed = pd.read_csv(FILENAME_FED, index_col=0)

    # Preprocessing
    returns = remove_outlier(returns)

    ## One hot encoding
    returns = pd.get_dummies(returns, columns=["Index Name"])
    ## Acquire targets
    returns["Sign"] = (returns["Index + 1"] > 0).astype(int)
    y = returns["Sign"]
    returns = returns.drop(["Sign", "Index + 1"], axis=1)

    nontextual_cols = ['Index - 9',
    'Index - 8',
    'Index - 7',
    'Index - 6',
    'Index - 5',
    'Index - 4',
    'Index - 3',
    'Index - 2',
    'Index - 1',
    'Index - 0',
    'Index Name_CVIX Index',
    'Index Name_EURUSD Curncy',
    'Index Name_EURUSDV1M Curncy',
    'Index Name_MOVE Index',
    'Index Name_SPX Index',
    'Index Name_SRVIX Index',
    'Index Name_SX5E Index',
    'Index Name_V2X Index',
    'Index Name_VIX Index']
    nb_nontextfeatures = len(nontextual_cols)

    ## Train test split: 60% train, 20% val, 20% test

    small_dataset_size = len(y)//2

    returns_, returns_test, y_, y_test = train_test_split(
        returns[:small_dataset_size], y[:small_dataset_size], test_size=0.2, train_size=0.8,
        random_state=0, stratify=y[:small_dataset_size]
        )

    returns_train, returns_val, y_train, y_val = train_test_split(
        returns_, y_, test_size=0.25, train_size=0.75,
        random_state=42, stratify=y_
        )

    ## Preprocess text
    # ecb["text_"] = ecb.apply(ecb_pipeline_en, axis=1)
    # ecb["lang"] = ecb["text_"].apply(fast_detect)
    # fed["lang"] = fed["text"].apply(fast_detect)

    # Preprocess numerical data
    ct = get_column_transformer()

    returns_train = pd.DataFrame(ct.fit_transform(returns_train), columns=returns_train.columns)
    returns_val = pd.DataFrame(ct.transform(returns_val), columns=returns_train.columns)
    returns_test = pd.DataFrame(ct.transform(returns_test), columns=returns_train.columns)


    train_set, train_loader, tokenizer, steps = get_data_loader(
    returns_train, ecb, fed, y_train, method=config["method"],
    separate=config["separate"], max_corpus_len=config["max_corpus_len"],
    batch_size=config["batch_size"]
    )
    # Use returns_train and y_train for overfit tests.
    val_set, val_loader, tokenizer, steps = get_data_loader(
        returns_val, ecb, fed, y_val, method=config["method"],
        separate=config["separate"], max_corpus_len=config["max_corpus_len"],
        batch_size=config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel(config["method"],
                    kwargs_nontext, kwargs_classification, kwargs_ce,
                    separate=config["separate"]).to(device)

    # model.corpus_encoder.requires_grad_(False)
    
    name = str(datetime.datetime.today()).replace(':', '-').replace('.', '-') + "-preload-cnn"
    max_epochs = config["max_epochs"]
    eval_losses, eval_accus, eval_f1s = \
        train(model, train_loader=train_loader, val_loader=val_loader,
            config=config, device=device, max_epochs=max_epochs, eval_every=5,
            name = name)
    
    with open(f"outputs/{name}_{max_epochs}_epochs.json", "w") as f:
        json.dump({
            "config": config,
            "eval_losses": eval_losses,
            "eval_accus": eval_accus,
            "eval_f1s": eval_f1s
        }, f)
    

if __name__=="__main__":
    optimizer = main()