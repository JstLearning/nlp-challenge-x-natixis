import torch
import torch.nn as nn
import numpy as np


class VectorAttention(nn.Module):
    def __init__(self, input_size):
        super(VectorAttention, self).__init__()
        self.K = nn.Linear(input_size, input_size, bias=False)
        self.V = nn.Linear(input_size, input_size, bias=False)
        self.input_size = input_size

    def forward(self, sequence, vector):
        keys = self.K(sequence)
        # vector is the query
        values = self.V(sequence)
        attention_weights = torch.matmul(keys, vector.unsqueeze(-1)).squeeze(-1)/np.sqrt(self.input_size)
        attention_weights = torch.softmax(attention_weights, dim=-1)
        context = torch.sum(values * attention_weights.unsqueeze(-1), dim=1)
        return context


class CNN1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=.2):
        super(CNN1D, self).__init__()
        self.input_layer = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        if num_layers >= 2:
            self.hidden_layers = nn.ModuleList()
            for i in range(num_layers-1):
                self.hidden_layers.append(nn.ReLU())
                self.hidden_layers.append(nn.Dropout(dropout))
                self.hidden_layers.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.transpose(1,2)  # transpose back to (batch_size, seq_len, hidden_dim)
        return x


class ResidualCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(ResidualCNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_layer = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            self.conv_layers.append(nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1))
        
    def forward(self, x):
        out = self.input_layer(x)
        res = out
        for conv in self.conv_layers:
            out = self.relu(conv(out))
            out = self.dropout(out)
            out = conv(out)
            out = self.relu(out + res)
            out = self.dropout(out)
            res = out
        out = out.transpose(1,2)
        return out




class Residual1DCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(Residual1DCNN, self).__init__()
        kernel_size=3
        stride=1
        padding=1
        output_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout = dropout
        
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels=input_size if i==0 else hidden_size,
                                                     out_channels=hidden_size,
                                                     kernel_size=kernel_size,
                                                     padding=kernel_size//2) 
                                           for i in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(num_layers)])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        residuals = []
        out = x
        for i in range(self.num_layers):
            if i<=1:
                residuals.append(out)
                out = self.relu(self.batch_norms[i](self.conv_layers[i](out)))
            else:
                residuals.append(out)
                out = self.relu(self.batch_norms[i](self.conv_layers[i](out)) + residuals[i-1])
                
        out = out.mean(dim=-1)
        out = self.fc(out)
        return out
