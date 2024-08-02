import numpy as np
import torch
import torch.nn as nn
from setbench.models.set.deepsets_base import PermEqui1_max, PermEqui1_mean, PermEqui2_max, PermEqui2_mean

def get_deepset_layer(x_dim, d_dim, pool='max', act='elu'):
  if act == 'elu':
    activation = nn.ELU(inplace=True)
  elif act == 'tanh':
    activation = nn.Tanh()
  if pool == 'max':
      phi = nn.Sequential(
        PermEqui2_max(x_dim, d_dim),
        activation,
        PermEqui2_max(d_dim, d_dim),
        activation,
        PermEqui2_max(d_dim, d_dim),
        activation,
      )
  elif pool == 'max1':
      phi = nn.Sequential(
        PermEqui1_max(x_dim, d_dim),
        activation,
        PermEqui1_max(d_dim, d_dim),
        activation,
        PermEqui1_max(d_dim, d_dim),
        activation,
      )
  elif pool == 'mean':
      phi = nn.Sequential(
        PermEqui2_mean(x_dim, d_dim),
        activation,
        PermEqui2_mean(d_dim, d_dim),
        activation,
        PermEqui2_mean(d_dim, d_dim),
        activation,
      )
  elif pool == 'mean1':
      phi = nn.Sequential(
        PermEqui1_mean(x_dim, d_dim),
        activation,
        PermEqui1_mean(d_dim, d_dim),
        activation,
        PermEqui1_mean(d_dim, d_dim),
        activation,
      )
  return phi

class CondEmbedding(nn.Module):
  def __init__(self, cond_dim, d_dim, pool='max', act='elu', dropout_prob=0.5):
    super(CondEmbedding, self).__init__()
    if act == 'elu':
      activation = nn.ELU(inplace=True)
    elif act == 'tanh':
      activation = nn.Tanh()
    self.cond_dim = cond_dim
    self.d_dim = d_dim
    self.phi = get_deepset_layer(self.cond_dim, self.d_dim, pool, act=act)
    
    self.ro = nn.Sequential(
       nn.Dropout(p=dropout_prob),
       nn.Linear(self.d_dim, self.d_dim),
       activation,
       nn.Dropout(p=dropout_prob),
       nn.Linear(self.d_dim, self.d_dim),
    )
    print(self)

  def forward(self, cond):
    phi_output = self.phi(cond)
    sum_output = phi_output.mean(1)
    ro_output = self.ro(sum_output)
    return ro_output

class D(nn.Module):
  def __init__(self, d_dim, x_dim=3, pool = 'mean'):
    super(D, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.phi = get_deepset_layer(self.x_dim, self.d_dim, pool, act='elu')
    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.ELU(inplace=True),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.ELU(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 1),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output = phi_output.mean(1)
    ro_output = self.ro(sum_output)
    return ro_output


class DTanh(nn.Module):

  def __init__(self, d_dim, x_dim=3, pool = 'mean'):
    super(DTanh, self).__init__()
    self.d_dim = d_dim
    self.x_dim = x_dim
    self.phi = get_deepset_layer(self.x_dim, self.d_dim, pool, act='tanh')
    self.ro = nn.Sequential(
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, self.d_dim),
       nn.Tanh(),
       nn.Dropout(p=0.5),
       nn.Linear(self.d_dim, 1),
    )
    print(self)

  def forward(self, x):
    phi_output = self.phi(x)
    sum_output, _ = phi_output.max(1)
    ro_output = self.ro(sum_output)
    return ro_output


def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm


if __name__ == '__main__':
  model = CondEmbedding(3, 128, pool='max', act='elu')
  inp = torch.randn(1, 100, 3)
  out = model(inp)
  print(out.shape)