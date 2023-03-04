"""
A general framework for the classification task.
It consists of:
- A Corpus Encoder to create an embedding of the corpus for both ECB and FED.
- A concatenation of that embedding with nontextual data from the data points.
- A classification head, which for now is nothing but a simple MLP with an adjustable amount of layers and neurons.

x_nontext   -------------------------------- \
                                              \
                                                [Concat] ---- [MLP] ---- Sigmoid ----> output 
                                              /
(x_text, x_mask) --[ Corpus Encoder ] -------/

"""

import torch
import torch.nn as nn


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

nontext_dim = len(nontextual_cols)


class ClassificationHead(nn.module):
    """
    A classification head with an adjustable amount of corpus dimension and nontextual dimension.
    This is just a MLP.
    """
    
    def __init__(self, corpus_emb_dim, nontext_dim=nontext_dim, layers=3, dropout=0):
        super(ClassificationHead, self).__init__()
        self.layers = layers
        self.corpus_emb_dim = corpus_emb_dim
        self.nontext_dim = nontext_dim

        layers_list = []
        output_sizes = [1, 32, 128, 256, 512]
        input_size = output_sizes[1]
        for i in range(layers-1):
            # 1, 32, 128, 256, 512, 512, 512, ...
            output_sizes = output_sizes[min(i, len(output_sizes)-1)]
            input_size = output_sizes[min(i+1, len(output_sizes)-1)]
            layers_list.append(nn.Linear(input_size, output_sizes))
        output_sizes = output_sizes[min(layers-1, len(output_sizes)-1)]
        input_size = corpus_emb_dim + nontext_dim
        layers_list.append(nn.Linear(input_size, output_sizes))
        # In place reverse
        layers_list.reverse()

        self.linears = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, x_corpus, x_nontext):
        x = torch.cat([x_corpus, x_nontext])
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < self.layers-1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
        return x


class CorpusEncoder(nn.Module):
    """Generic Corpus encoder for both ECB and FED texts.
    """

    def __init__(self, method='max_pooling', separate=True, dropout=0.):
        """Initializes a Corpus Encoder with the given method.

        Args:
            method (str): {'hierbert', 'max_pooling', 'bow'} Method to use for corpus encoding. Defaults to 'max_pooling'.
            separate (bool, optional): Boolean that indicates whether to create
                    a separate encoder for ECB and for FED. Defaults to True.
            dropout (float, optional): The dropout probability. Defaults to 0.
        """
        super(CorpusEncoder, self).__init__()
        self.method = method
        self.separate=separate
        self.dropout=dropout

        if self.method=='bow':
            self.corpus_emb_dim = 1 * (1 + int(separate))
            # self.encoder = Model()
        elif self.method=='max_pooling':
            self.corpus_emb_dim = 1 * (1 + int(separate))
            # self.encoder = Model()
        elif self.method=='hierbert':
            # https://huggingface.co/kiddothe2b/hierarchical-transformer-I3-mini-1024
            self.corpus_emb_dim = 1 * (1 + int(separate))
            # self.encoder = Model()

        self.init_weights()

    def forward(self, x, x_masks):
        if self.method=='bow':
            # x = ...
            # x_masks = ...
            pass
        elif self.method=='max_pooling':
            # x = ...
            # x_masks = ...
            pass
        elif self.method=='hierbert':
            # x = ...
            # x_masks = ...
            pass
        out = self.encoder(x, x_masks)

class MyModel(nn.Module):
    """
    Custom model using the framework stated above, with one corpus encoding concatenated with the nontextual features, followed by a MLP.
    """
    def __init__(self, nontext_dim=nontext_dim, method='max_pooling', separate=True, layers=3, dropout=0.3):
        """_summary_

        Args:
            nontext_dim (int, optional): _description_. Defaults to 19.
            method (str): {'hierbert', 'max_pooling', 'bow'} Method to use for corpus encoding.
                    Defaults to 'max_pooling'.
            separate (bool, optional): Boolean that indicates whether to create
                    a separate encoder for ECB and for FED. Defaults to True.
            layers (int, optional): Number of layers to use in the classification head. Defaults to 3.
            dropout (float, optional): The dropout probability. Defaults to 0.
        """

        super(MyModel, self).__init__()
        self.method = method
        self.dropout=dropout

        self.corpus_encoder = CorpusEncoder(method=method, separate=separate, dropout=dropout)

        # TODO: Get Corpus embedding dimension
        corpus_emb_dim = self.corpus_encoder.corpus_emb_dim
        
        self.classifier = ClassificationHead(corpus_emb_dim=corpus_emb_dim, nontext_dim=nontext_dim, layers=layers, dropout=dropout)
    
    def forward(self, x_text, x_masks, x_nontext):
        """Forward method for the general framework.

        Args:
            x_text (torch.Tensor): _description_
            x_masks (torch.Tensor): _description_
            x_nontext (torch.Tensor): Tensor for the nontextual features.

        Returns:
            torch.Tensor: Probability that the sample is of a positive class.
        """

        # TODO: Text encoding
        if self.method=='bow':
            pass
        elif self.method=='max_pooling':
            pass
        elif self.method=='hierbert':
            pass
        # Temp
        x_text_ = x_text
        x_corpus = self.corpus_encoder(x_text_, x_masks)

        # Downstream classification
        out = self.classifier(x_corpus, x_nontext)
        return out