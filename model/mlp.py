import torch
import torch.nn as nn
from collections import OrderedDict

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=.4):
        super(MLPLayer, self).__init__()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout= nn.Dropout(dropout)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.relu(self.linear(self.dropout(self.bn(x))))
        return out

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=.4):
        super(ResidualBlock, self).__init__()
        self.layer1 = MLPLayer(hidden_dim, hidden_dim, dropout)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.skip = nn.Identity()

    def forward(self, x):
        z = self.bn(x)
        z = self.dropout(z)
        z = self.layer1(z)
        out = self.relu(z + self.skip(x))
        return out

class DownsamplingBlock(nn.Module):
    def __init__(self, input_dim, dropout=.4):
        super(DownsamplingBlock, self).__init__()
        assert input_dim % 2 == 0
        output_dim = input_dim//2
        self.bn = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.relu(self.linear(self.dropout(self.bn(x))))
        return out

class MLP(nn.Module):
    def __init__(self, dim_1, dim_2, nb_layers, dropout=0.):
        super(MLP, self).__init__()

        if nb_layers==1:
            self.layers = MLPLayer(input_dim=dim_1+dim_2,
                                    output_dim=1,
                                    dropout=dropout)
        elif nb_layers==2:
            self.hidden_dim = 64
            self.layers = nn.Sequential(
                MLPLayer(input_dim=dim_1+dim_2,
                            output_dim=self.hidden_dim,
                            dropout=dropout),
                MLPLayer(input_dim=self.hidden_dim,
                            output_dim=1,
                            dropout=dropout)
            )
        elif nb_layers==3:
            self.hidden_dim = 64
            self.layers = nn.Sequential(
                MLPLayer(input_dim=dim_1+dim_2,
                            output_dim=self.hidden_dim,
                            dropout=dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            hidden_dim = 2 ** (4 + nb_layers)
            layers_dict = OrderedDict()
            layers_dict['MLP_input'] = MLPLayer(input_dim=dim_1+dim_2,
                                                output_dim=hidden_dim
                                        )
            nb_blocks = (nb_layers-3)//2
            leftover = (nb_layers-3)%2
            for k in range(nb_blocks):
                layers_dict[f'step_{k+1}'] = ResidualBlock(
                    hidden_dim=hidden_dim, dropout=dropout
                )
                layers_dict[f'step_{k+2}'] = DownsamplingBlock(
                    input_dim=hidden_dim, dropout=dropout
                )
                hidden_dim = hidden_dim//2
            if leftover==1:
                layers_dict[f'step_{k+3}'] = ResidualBlock(
                    hidden_dim=hidden_dim, dropout=dropout
                )
            layers_dict['Second_last_linear'] = nn.Linear(
                hidden_dim, hidden_dim
            )
            layers_dict['Last_ReLU'] = nn.ReLU()
            layers_dict['Last_Linear'] = nn.Linear(
                hidden_dim, 1
            )

            self.layers = nn.Sequential(layers_dict)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        if x1 is None and x2 is None:
            raise ValueError("Both x1 and x2 are None.")
        elif x1 is None:
            x = x2
        elif x2 is None:
            x = x1
        else:
            x = torch.cat([x1, x2], dim=1)
        x = self.layers(x)
        x = self.sigmoid(x)
        return x