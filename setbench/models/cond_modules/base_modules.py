import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers, dropout_prob, init_drop=False):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim    
        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU()] 
        layers += [nn.Dropout(dropout_prob)] if init_drop else []
        for i in range(1, len(hidden_layers)):
            layers.extend([nn.Linear(hidden_layers[i-1], hidden_layers[i]), nn.ReLU(), nn.Dropout(dropout_prob)])
        layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, with_uncertainty=False):
        return self.model(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)