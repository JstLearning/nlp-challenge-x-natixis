"""
A general framework for the classification task.
It consists of:
- A Corpus Encoder to create an embedding of the corpus for both ECB and FED.
- A concatenation of that embedding with nontextual data from the data points.
- A classification head, which for now is nothing but a simple MLP with an adjustable amount of layers and neurons.

x_nontext   --- [Non-textual pipeline] ----- \
                                              \
                                                [Concat] ---- [MLP] ---- Sigmoid ----> output 
                                              /
(x_text, x_mask) --[ Corpus Encoder ] -------/

"""

import torch
import torch.nn as nn

from .model_01.model import CorpusEncoder as CorpusEncoder01
from .model_02.model import CorpusEncoder as CorpusEncoder02
from .model_03.model import CorpusEncoder as CorpusEncoder03


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

index_names = [
 'Index Name_CVIX Index',
 'Index Name_EURUSD Curncy',
 'Index Name_EURUSDV1M Curncy',
 'Index Name_MOVE Index',
 'Index Name_SPX Index',
 'Index Name_SRVIX Index',
 'Index Name_SX5E Index',
 'Index Name_V2X Index',
 'Index Name_VIX Index'
]

index_times = [
 'Index - 9',
 'Index - 8',
 'Index - 7',
 'Index - 6',
 'Index - 5',
 'Index - 4',
 'Index - 3',
 'Index - 2',
 'Index - 1',
 'Index - 0',
]

nontext_dim = len(nontextual_cols)

class ClassificationHead(nn.Module):
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
            output_size = output_sizes[min(i, len(output_sizes)-1)]
            input_size = output_sizes[min(i+1, len(output_sizes)-1)]
            layers_list.append(nn.Linear(input_size, output_size))
        output_sizes = output_sizes[min(layers-1, len(output_sizes)-1)]
        input_size = corpus_emb_dim + nontext_dim
        layers_list.append(nn.Linear(input_size, output_sizes))
        # In place reverse
        layers_list.reverse()

        self.linears = nn.ModuleList(layers_list)
        self.dropout = nn.Dropout(dropout)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_corpus, x_nontext):
        x = torch.cat([x_corpus, x_nontext], dim=1).float()
        for i, l in enumerate(self.linears):
            x = l(x)
            if i < self.layers-1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
        return x.view(-1)
    

class NontextualNetwork(nn.Module):
    """
    A network to process nontextual data.
    For this, we pick a CNN.
    """
    
    def __init__(self, input_dim, input_channels, output_dim=nontext_dim, layers_nontext=3, dropout=0):
        super(NontextualNetwork, self).__init__()
        self.layers_nontext = layers_nontext
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.output_dim = output_dim

        layers_list = []
        channels = [1, 4, 16, 64]

        for i in range(layers_nontext-1):
            out_channels = channels[min(i, len(channels)-1)]
            in_channels = channels[min(i+1, len(channels)-1)]
            layers_list.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=3, stride=1, padding="same"))            
        
        out_channels = channels[min(layers_nontext-1, len(channels)-1)]
        layers_list.append(nn.Conv1d(input_channels, channels, kernel_size=3, stride=1, padding="same"))
        layers_list.reverse()

        self.conv_layers = nn.ModuleList(layers_list)
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        
        # Activation functions
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward method for the NontextualNetwork.

        Args:
            x (Tensor): Tensor of size [batch_size, in_channels, nb_features].
                In our case, the number of channels corresponds to the amount of
                dummies for the one-hot encoding.

        Returns:
            Tensor: Tensor of size [batch_size, output_dim]
        """
        for i, l in enumerate(self.conv_layers):
            x = l(x)
            x = self.relu(x)
            x = self.dropout(x)
    
        batch_size_ = x.size(0)
        x = x.view(batch_size_, -1)
        x = self.linear(x)
        return x
        




class CorpusEncoder(nn.Module):
    """Generic Corpus encoder for both ECB and FED texts.
    """

    def __init__(self, method='model_01', separate=True, dropout=0.):
        """Initializes a Corpus Encoder with the given method.

        Args:
            method (str): {'hierbert', 'max_pooling', 'bow', 'model_01', 'model_02', ...}
                    Method to use for corpus encoding. Defaults to 'max_pooling'.
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
        elif self.method=='model_01':
            self.corpus_emb_dim = 32 * (1 + int(separate))
            if not separate:
                self.encoder = CorpusEncoder01(dropout=dropout)
                self.encoder_ecb = None
                self.encoder_fed = None
            else:
                self.encoder = None
                self.encoder_ecb = CorpusEncoder01(dropout=dropout)
                self.encoder_fed = CorpusEncoder01(dropout=dropout)
        elif self.method=='model_02':
            self.corpus_emb_dim = 32 * (1 + int(separate))
            if not separate:
                self.encoder = CorpusEncoder02(dropout=dropout)
                self.encoder_ecb = None
                self.encoder_fed = None
            else:
                self.encoder = None
                self.encoder_ecb = CorpusEncoder03(dropout=dropout)
                self.encoder_fed = CorpusEncoder03(dropout=dropout)
        elif self.method=='model_03':
            self.corpus_emb_dim = 32 * (1 + int(separate))
            if not separate:
                self.encoder = CorpusEncoder03(dropout=dropout)
                self.encoder_ecb = None
                self.encoder_fed = None
            else:
                self.encoder = None
                self.encoder_ecb = CorpusEncoder03(dropout=dropout)
                self.encoder_fed = CorpusEncoder03(dropout=dropout)

    def forward(self, x, x_masks):
        """_summary_

        Args:
            x (tuple(Tensor)): Tuple of length either 1 or 2 of tensors of same size.
                If separate is True, the tuple should be of length 2 and contain
                tensors for the tokens of each corpus (ECB and FED respectively).
            x_masks (tuple(Tensor)): Tuple of length either 1 or 2 of tensors of same size.
                If separate is True, the tuple should be of length 2 and contain
                tensors for the attention masks of each corpus (ECB and FED respectively).

        Returns:
            Tensor: Output Tensor of size [batch_size, self.corpus_emb_dim].
        """
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
        elif self.method=='model_01':
            pass
        if self.separate:
            x_ecb = self.encoder_ecb(x[0], x_masks[0])
            x_fed = self.encoder_fed(x[1], x_masks[1])
            out = torch.cat([x_ecb, x_fed], dim=1).float()
        else:
            out = self.encoder(x[0], x_masks[0])
        return out

class MyModel(nn.Module):
    """
    Custom model using the framework stated above, with one corpus encoding concatenated with the
    nontextual features, followed by a MLP.

    One can process the nontextual features with a pipeline, for instance a CNN.
    """
    def __init__(self, has_nontext_network=False, nontext_dim=nontext_dim, layers_nontext=3,
                 method='model_01', separate=True, layers=3, dropout=0.3):
        """_summary_

        Args:
            has_nontext_pipeline (bool, optional): Whether to apply a nontextual network on the inputs.
                Defaults to False.
            nontext_dim (int, optional): Number of non-textual features before concatenation. Defaults to 19.
                If has_nontext_pipeline is False, this will be forced to 19.
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

        if has_nontext_network:
            self.nontextual_pipeline = NontextualNetwork(
                len(index_times), len(index_names), nontext_dim, layers_nontext=layers_nontext, dropout=dropout
            )
            self.nontext_dim = nontext_dim
        else:
            self.nontextual_pipeline = None
            self.nontext_dim = len(nontextual_cols)

        self.corpus_encoder = CorpusEncoder(method=method, separate=separate, dropout=dropout)

        corpus_emb_dim = self.corpus_encoder.corpus_emb_dim
        
        self.classifier = ClassificationHead(corpus_emb_dim=corpus_emb_dim, nontext_dim=self.nontext_dim,
                                             layers=layers, dropout=dropout)
    
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