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
        # print("x in entry of MLP Layer is leaf: ", x.is_leaf, x.grad)
        z = self.bn(x)
        # print("z is leaf : ", z.is_leaf, z.grad)
        z = self.dropout(x)
        # print("After dropout, z is leaf: ", z.is_leaf)
        z = self.linear(z)
        # print("After linear, z is leaf: ", z.is_leaf)
        out = self.relu(z)
        # print("MLP Layer out is leaf: ", out.is_leaf)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=.4):
        super(ResidualBlock, self).__init__()
        self.layer1 = MLPLayer(hidden_dim, hidden_dim, dropout)
        self.bn = nn.BatchNorm1d(hidden_dim, track_running_stats=False)
        self.bn.weight.requires_grad = True
        self.bn.bias.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.skip = nn.Identity()

    def forward(self, x):
        # print("Entry of residual block is leaf: ", x.is_leaf)
        z = self.bn(x)
        x_ = self.skip(x)
        # print("After batch norm, z is leaf: ", z.is_leaf)
        z = self.dropout(z)
        # print("After dropout, z is leaf: ", z.is_leaf)
        z = self.layer1(z)
        # print("After linear layer, z is leaf: ", z.is_leaf)
        out = self.relu(z + x_)
        # print("Residual output is leaf: ", out.is_leaf)
        return out

class DownsamplingBlock(nn.Module):
    def __init__(self, input_dim, dropout=.4):
        super(DownsamplingBlock, self).__init__()
        assert input_dim % 2 == 0
        output_dim = input_dim//2
        self.bn = nn.BatchNorm1d(input_dim, track_running_stats=False)
        self.bn.weight.requires_grad = True
        self.bn.bias.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.relu(self.linear(self.dropout(self.bn(x))))
        # print("After downsampling, out is leaf: ", out.is_leaf)
        return out

class MLP(nn.Module):
    def __init__(self, input_dim, nb_layers, mlp_hidden_dim=64, dropout=0.):
        super(MLP, self).__init__()
        assert nb_layers > 0
        self.input_dim = input_dim
        self.nb_layers = nb_layers


        if nb_layers==1:
            self.layers = MLPLayer(input_dim=input_dim,
                                    output_dim=1,
                                    dropout=dropout)
        elif nb_layers==2:
            self.hidden_dim = mlp_hidden_dim
            self.layers = nn.Sequential(
                MLPLayer(input_dim=input_dim,
                            output_dim=self.hidden_dim,
                            dropout=dropout),
                MLPLayer(input_dim=self.hidden_dim,
                            output_dim=1,
                            dropout=dropout)
            )
        elif nb_layers==3:
            self.hidden_dim = mlp_hidden_dim
            self.layers = nn.Sequential(
                MLPLayer(input_dim=input_dim,
                            output_dim=self.hidden_dim,
                            dropout=dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            hidden_dim = 2 ** (4 + nb_layers)
            layers_dict = OrderedDict()
            layers_dict['MLP_input'] = MLPLayer(input_dim=input_dim,
                                                output_dim=hidden_dim
                                        )
            nb_blocks = (nb_layers-3)//2
            leftover = (nb_layers-3)%2
            k = 0
            for k in range(nb_blocks):
                layers_dict[f'step_{k+1}_residual'] = ResidualBlock(
                    hidden_dim=hidden_dim, dropout=dropout
                )
                layers_dict[f'step_{k+1}_downsampling'] = DownsamplingBlock(
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


    def forward(self, x):
        # print("Before layers, x is a leaf node: ", x.is_leaf)
        x = self.layers(x)
        # print("after layers, x is a leaf node: ", x.is_leaf)
        # print("after Sigmoid, x is a leaf node: ", x.is_leaf)
        return x



class SimpleMLP(nn.Module):
    def __init__(self, input_size, nb_layers, hidden_size, dropout=0):
        super(SimpleMLP, self).__init__()
        if nb_layers==1:
            layers =[
                nn.BatchNorm1d(input_size),
                nn.Linear(input_size, 1)
            ]
        else:
            layer_sizes = [input_size] + [hidden_size]*(nb_layers-1) + [1]
            layers = []
            for i in range(len(layer_sizes)-1):
                # layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
                if i < len(layer_sizes)-2:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class CompactMLP(nn.Module):
    def __init__(self, input_size, nb_layers, hidden_size, dropout_prob=0.0):
        super(CompactMLP, self).__init__()
        output_size = 1
        hidden_sizes = [hidden_size] * (nb_layers-1)
        layer_sizes = [input_size] + hidden_sizes
        layers = []
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            if i > 0:
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                skip_connections.append(x)
        out = skip_connections[-1]
        for i in range(len(skip_connections)-2, -1, -1):
            if skip_connections[i].shape[-1] == out.shape[-1]:
                out = out + skip_connections[i]
        return out