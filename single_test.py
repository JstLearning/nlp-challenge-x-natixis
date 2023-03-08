import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split

from model.framework_model import MyModel
from model.framework_dataset import get_data_loader
from train import train, evaluate

# from preprocessing.preprocessing import ecb_pipeline_en, fast_detect
from config import Optimizer

import torch
import torch.nn as nn

config = {

    "method": 'model_03',

    "learning_rate": 1e-3,

    "weight_decay": 0,

    "batch_size": 16,

    "layers": 3,

    "mlp_hidden_dim": 64,

    "dropout": 0.5,

    "separate": True,
    
    "max_corpus_len": 2,

    "max_epochs": 20

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

    returns_, returns_test, y_, y_test = train_test_split(
        returns, y, test_size=0.2, train_size=0.8,
        random_state=0, stratify=y
        )

    returns_train, returns_val, y_train, y_val = train_test_split(
        returns_, y_, test_size=0.25, train_size=0.75,
        random_state=42, stratify=y_
        )

    ## Preprocess text
    # ecb["text_"] = ecb.apply(ecb_pipeline_en, axis=1)
    # ecb["lang"] = ecb["text_"].apply(fast_detect)
    # fed["lang"] = fed["text"].apply(fast_detect)

    train_set, train_loader, tokenizer, steps = get_data_loader(
    returns_train, ecb, fed, y_train, method=config["method"],
    separate=config["separate"], max_corpus_len=config["max_corpus_len"],
    batch_size=config["batch_size"]
    )

    val_set, val_loader, tokenizer, steps = get_data_loader(
        returns_val, ecb, fed, y_val, method=config["method"],
        separate=config["separate"], max_corpus_len=config["max_corpus_len"],
        batch_size=config["batch_size"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MyModel(method=config["method"],
                    layers=config["layers"],
                    mlp_hidden_dim=config["mlp_hidden_dim"],
                    separate=config["separate"],
                    dropout=config["dropout"]).to(device)

    max_epochs = config["max_epochs"]
    eval_losses, eval_accus, eval_f1s = \
        train(model, train_loader=train_loader, val_loader=val_loader,
            config=config, device=device, max_epochs=max_epochs, eval_every=2,
            name = f"No_NLP")
    
    with open(f"{config['method']}_{max_epochs}_epochs.json", "w") as f:
        json.dump({
            "config": config,
            "eval_losses": eval_losses,
            "eval_accus": eval_accus,
            "eval_f1s": eval_f1s
        }, f)
    

if __name__=="__main__":
    optimizer = main()