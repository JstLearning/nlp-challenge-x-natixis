import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import DistilBertTokenizer

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


class BlankReturnsDataset(Dataset):
    def __init__(self, returns):
        
        self.returns = returns
        self.y = y

    def __getitem__(self, index):
        # x_ind is of size (19)
        x_ind = torch.Tensor(self.returns.iloc[index][nontextual_cols])

        label = self.y.iloc[index]

        return x_ind, torch.Tensor(label)

    def __len__(self):
        return self.returns.shape[0]

class ReturnsDataset(Dataset):
    def __init__(self, returns, ecb, fed, y, max_corpus_len, 
                 english_only=False, separate=True, filler=""):
        
        self.returns = returns
        self.ecb = ecb
        self.fed = fed
        self.y = y
        self.english_only = english_only
        self.separate = separate
        self.max_corpus_len = max_corpus_len
        self.filler = filler

        assert max_corpus_len < 5, \
            f"The max amount of texts in a corpus is 4. Got {max_corpus_len=}"

    def __getitem__(self, index):

        index_ecb = self.returns.iloc[index]["index ecb"]
        index_ecb = [int(i) for i in index_ecb.split(",")]
        index_ecb = index_ecb[:min(self.max_corpus_len, len(index_ecb))]

        index_fed = self.returns.iloc[index]["index fed"]
        index_fed = [int(i) for i in index_fed.split(",")]
        index_fed = index_fed[:min(self.max_corpus_len, len(index_fed))]

        # For a simple model, we will only pick texts in english.
        if self.english_only:
            texts = self.ecb.loc[index_ecb]
            # ecb texts have been preprocessed.
            ecb_texts = list(texts[texts["lang"] == "en"]["text_"])
            while len(ecb_texts) < self.max_corpus_len:
                ecb_texts.append(self.filler)

            texts = self.fed.loc[index_fed]
            fed_texts = list(texts[texts["lang"] == "en"]["text"])
            while len(fed_texts) < self.max_corpus_len:
                fed_texts.append(self.filler)

        else:
            ecb_texts = list(self.ecb.loc[index_ecb]["text_"])
            while len(ecb_texts) < self.max_corpus_len:
                ecb_texts.append(self.filler)

            fed_texts = list(self.fed.loc[index_fed]["text"])
            while len(fed_texts) < self.max_corpus_len:
                fed_texts.append(self.filler)
        # Both ecb_texts and fed_texts should be of length max_corpus_len

        # x_ind is of size (19)
        x_ind = torch.Tensor(self.returns.iloc[index][nontextual_cols])

        label = self.y.iloc[index]

        return ((ecb_texts, fed_texts, x_ind), label)

    def __len__(self):
        return self.returns.shape[0]


def get_data_loader(returns, ecb, fed, y,
                    method="model_01",
                    separate=True,
                    batch_size=2,
                    max_corpus_len=2):
    """Creates a DataLoader for a certain method..

    Args:
        returns (pd.DataFrame): A DataFrame for the returns dataset.
        ecb (pd.DataFrame): A DataFrame for the ECB texts.
        fed (pd.DataFrame): A DataFrame for the FED texts.
        y (pd.DataFrame): A pd.Series for the targets.
        method (str, optional): The method intended for the DataLoader. Defaults to "model_01".
        separate (bool, optional): Whether to separate ECB texts from FED. Defaults to True.
        batch_size (int, optional): The batch size for the loader.. Defaults to 2.
        max_corpus_len (int, optional): The maximum corpus size. If separate=True, then
            the corpus for both ECB and FED is made of max_corpus_len each, leading to
            a total corpus size of 2*max_corpus_len. Defaults to 2.

    Returns:
        dataset (Dataset): A Dataset object for the specific method.
        loader (DataLoader): A DataLoader for the specific method.
        tokenizer: A tokenizer object that depends on the method.
        steps (int): The amount of time steps in the sequential data (text here).
    """
    if method in ["model_01", "model_02"]:
        return get_data_loader_distilbert(
            returns, ecb, fed, y,
            separate=separate,
            batch_size=batch_size,
            max_corpus_len=max_corpus_len,
            filler=""
        )
    elif method == "model_03" or method is None:
        return get_data_loader_distilbert(
            returns, ecb, fed, y,
            separate=separate,
            batch_size=batch_size,
            max_corpus_len=max_corpus_len,
            filler=""
        )
    elif method is None:
        return get_data_loader_blank(
            returns, ecb, fed, y,
            separate=separate,
            batch_size=batch_size,
            max_corpus_len=max_corpus_len,
            filler=""
        )


def get_data_loader_blank(returns, ecb, fed, y,
                               separate=True,
                               batch_size=2,
                               max_corpus_len=2, filler=""):
    """Creates a DataLoader with no text.

    Args:
        returns (pd.DataFrame): A DataFrame for the returns dataset.
        ecb (pd.DataFrame): A DataFrame for the ECB texts.
        fed (pd.DataFrame): A DataFrame for the FED texts.
        y (pd.DataFrame): A pd.Series for the targets.
        method (str, optional): The method intended for the DataLoader. Defaults to "model_01".
        separate (bool, optional): Whether to separate ECB texts from FED. Defaults to True.
        batch_size (int, optional): The batch size for the loader.. Defaults to 2.
        max_corpus_len (int, optional): The maximum corpus size. If separate=True, then
            the corpus for both ECB and FED is made of max_corpus_len each, leading to
            a total corpus size of 2*max_corpus_len. Defaults to 2.

    Returns:
        dataset (Dataset): A Dataset object for the specific method.
        loader (DataLoader): A DataLoader for the specific method.
        tokenizer: A tokenizer object that depends on the method.
        steps (int): The amount of time steps in the sequential data (text here).
    """
    dataset = BlankReturnsDataset(returns)
        
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6
    )
    steps = 0
    return dataset, loader, None, 0


def get_data_loader_distilbert(returns, ecb, fed, y,
                               separate=True,
                               batch_size=2,
                               max_corpus_len=2, filler=""):
    """Creates a DataLoader for the DistilBertTokenizer.

    Args:
        returns (pd.DataFrame): A DataFrame for the returns dataset.
        ecb (pd.DataFrame): A DataFrame for the ECB texts.
        fed (pd.DataFrame): A DataFrame for the FED texts.
        y (pd.DataFrame): A pd.Series for the targets.
        method (str, optional): The method intended for the DataLoader. Defaults to "model_01".
        separate (bool, optional): Whether to separate ECB texts from FED. Defaults to True.
        batch_size (int, optional): The batch size for the loader.. Defaults to 2.
        max_corpus_len (int, optional): The maximum corpus size. If separate=True, then
            the corpus for both ECB and FED is made of max_corpus_len each, leading to
            a total corpus size of 2*max_corpus_len. Defaults to 2.

    Returns:
        dataset (Dataset): A Dataset object for the specific method.
        loader (DataLoader): A DataLoader for the specific method.
        tokenizer: A tokenizer object that depends on the method.
        steps (int): The amount of time steps in the sequential data (text here).
    """
    dataset = ReturnsDataset(returns, ecb, fed, y,
                             max_corpus_len=max_corpus_len,
                             separate=separate, filler=filler)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def collate_fn(batch, separate, max_corpus_len):
        batch_size_ = len(batch)
        # print("Batch size = ", batch_size_)
        X_ind = []
        y = []

        if separate:
            X_ecb = []
            X_fed = []

            for data in batch:
                X_ecb.extend(data[0][0])
                X_fed.extend(data[0][1])
                X_ind.append(data[0][2])
                y.append(data[1])
            X_ind = torch.stack(X_ind, dim=0)
            Y = torch.Tensor(y)

            X_ecb_tokens = tokenizer(X_ecb, return_tensors="pt",
                                     truncation=True, padding='max_length', max_length=512)
            # print("size X_ecb : ", X_ecb_tokens['input_ids'].size())
            X_ecb = X_ecb_tokens['input_ids'].view(
                batch_size_, max_corpus_len, 512)
            X_ecb_att = X_ecb_tokens['attention_mask'].view(
                batch_size_, max_corpus_len, 512)

            X_fed_tokens = tokenizer(X_fed, return_tensors="pt",
                                     truncation=True, padding='max_length', max_length=512)
            X_fed = X_fed_tokens['input_ids'].view(
                batch_size_, max_corpus_len, 512)
            X_fed_att = X_fed_tokens['attention_mask'].view(
                batch_size_, max_corpus_len, 512)

            return {
                "X_ecb": X_ecb,
                "X_ecb_mask": X_ecb_att,
                "X_fed": X_fed,
                "X_fed_mask": X_fed_att,
                "X_ind": X_ind,
                "label": Y
            }
        else:
            X_text = []

            for data in batch:
                X_text.extend(data[0][0])
                X_text.extend(data[0][1])
                X_ind.append(data[0][2])
                y.append(data[1])
            
            X_ind = torch.stack(X_ind, dim=0)
            Y = torch.Tensor(y)

            X_text_tokens = tokenizer(X_text, return_tensors="pt",
                                      truncation=True, padding='max_length', max_length=512)
            # print(X_text_tokens)
            X_text = X_text_tokens['input_ids'].view(
                batch_size_, 2*max_corpus_len, 512)
            X_att = X_text_tokens['attention_mask'].view(
                batch_size_, 2*max_corpus_len, 512)

            return {
                "X_text": X_text,
                "X_mask": X_att,
                "X_ind": X_ind,
                "label": Y
            }

    loader = DataLoader(
        dataset=dataset,
        collate_fn=lambda batch : collate_fn(batch, separate, max_corpus_len),
        batch_size=batch_size,
        shuffle=True,
        num_workers=6
    )
    steps = 512
    return dataset, loader, tokenizer, steps