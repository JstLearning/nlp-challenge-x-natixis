import pandas as pd
import numpy as np

from model.mlp import SimpleMLP

from sklearn.model_selection import train_test_split

from preprocessing.standard_scaler import get_column_transformer


# from preprocessing.preprocessing import ecb_pipeline_en, fast_detect
from preprocessing.outlier_detection import remove_outlier
from config import Optimizer

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

    ct = get_column_transformer()

    returns_train = pd.DataFrame(ct.fit_transform(returns_train), columns=returns_train.columns)
    returns_val = pd.DataFrame(ct.transform(returns_val), columns=returns_train.columns)
    returns_test = pd.DataFrame(ct.transform(returns_test), columns=returns_train.columns)


    optimizer = Optimizer(returns_train=returns_train,
                        returns_val=returns_val,
                        returns_test=returns_test,
                        y_train=y_train,
                        y_val=y_val,
                        y_test=y_test,
                        ecb=ecb,
                        fed=fed,
                        n_trials=10)

    optimizer.optimize()
    return optimizer

if __name__=="__main__":
    optimizer = main()