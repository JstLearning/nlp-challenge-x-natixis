import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

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

class ReturnsDataset(Dataset):
    def __init__(self, returns, ecb, fed, y, english_only = True):
        self.returns = returns
        self.ecb = ecb
        self.fed = fed
        self.y = y
        self.english_only = english_only

        self.max_corpus_len = 2
    
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
                ecb_texts.append("")
            
            texts = self.fed.loc[index_fed]
            fed_texts = list(texts[texts["lang"] == "en"]["text"])
            while len(fed_texts) < self.max_corpus_len:
                fed_texts.append("")
        
        else:
            ecb_texts = list(self.ecb.loc[index_ecb]["text_"])
            while len(ecb_texts) < self.max_corpus_len:
                ecb_texts.append("")

            fed_texts = list(self.fed.loc[index_fed]["text"])
            while len(fed_texts) < self.max_corpus_len:
                fed_texts.append("")

        #x_ind is of size (19)
        x_ind = torch.Tensor(self.returns.iloc[index][nontextual_cols])

        label = self.y.iloc[index]

        
        return ((ecb_texts, fed_texts, x_ind), label)

    def __len__(self):
        return self.returns.shape[0]