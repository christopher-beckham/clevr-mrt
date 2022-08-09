#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# Copyright 2020, Christopher Beckham
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, kaiming_uniform
from functools import partial


class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None,
               with_residual=True,
               with_batchnorm=True,
               with_film=False,
               with_coords=False,
               coord_shape=None,
               downsample=False,
               is_3d=False):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()

    if is_3d:
      conv_class = nn.Conv3d
      bn_class = nn.BatchNorm3d
    else:
      conv_class = nn.Conv2d
      bn_class = nn.BatchNorm2d

    self.with_coords = with_coords
    coord_dim = 0
    if with_coords:
      if coord_shape is None:
        raise Exception("Need to specify spatial dim of coord layers")
      else:
        if is_3d:
          self.coords = coord_map_3d(coord_shape)
          coord_dim = 3
        else:
          self.coords = coord_map(coord_shape)
          coord_dim = 2

    self.conv1 = conv_class(in_dim+coord_dim,
                            out_dim,
                            kernel_size=3,
                            padding=1)
    self.conv2 = conv_class(out_dim,
                            out_dim,
                            kernel_size=3,
                            padding=1,
                            stride=2 if downsample else 1)
    self.with_batchnorm = with_batchnorm
    self.with_film = with_film
    self.is_3d = is_3d

    if with_batchnorm:
      self.bn1 = bn_class(out_dim, affine=False)
      self.bn2 = bn_class(out_dim, affine=False)
    self.film = None
    if with_film:
      self.film = Film()
    self.with_residual = with_residual
    if not downsample and (in_dim == out_dim or not with_residual):
      self.proj = None
    else:
      if downsample:
        self.proj = nn.Sequential(
          conv_class(in_dim+coord_dim,
                     out_dim, kernel_size=1),
          nn.AvgPool3d(2) if is_3d else nn.AvgPool2d(2)
        )
      else:
        self.proj = conv_class(in_dim+coord_dim,
                               out_dim, kernel_size=1)

  def forward(self, x, embedding=None):
    orig_x = x
    if self.with_coords:
      if self.is_3d:
        coords = self.coords.repeat(x.size(0), 1, 1, 1, 1)
      else:
        coords = self.coords.repeat(x.size(0), 1, 1, 1)
      x = torch.cat((x, coords), dim=1)
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
      if self.film is not None:
        out = self.film(out, embedding)
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = orig_x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out

def coord_map(shape, start=-1, end=1):
  """
  Gives, a 2d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  """
  m, n = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
  return torch.cat([x_coords, y_coords], 0)

def coord_map_3d(shape, start=-1, end=1):
  """
  Gives, a 3d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  """
  m, n, o = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=o).type(torch.cuda.FloatTensor)
  z_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)

  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n, o))).unsqueeze(0)
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n, o))).unsqueeze(0)
  #z_coords = z_coord_row.unsqueeze(2).expand(torch.Size((m, n, o))).unsqueeze(0)
  z_coords = z_coord_row.unsqueeze(0).view(-1, m, 1, 1).repeat(1,1,n,o)

  return torch.cat([x_coords, y_coords, z_coords], 0)


class Film(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, embedding):
    gammas = embedding[:, 0:(embedding.size(1)//2)]
    betas = embedding[:, (embedding.size(1)//2)::]
    if len(x.shape) == 4:
      gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
      betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
    elif len(x.shape) == 5:
      gammas = gammas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
      betas = betas.unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
    else:
      raise Exception("")

    return (gammas * x) + betas

class ConcatBlock(nn.Module):
  def __init__(self, dim, with_residual=True, with_batchnorm=True):
    super(ConcatBlock, self).__init__()
    self.proj = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                                   with_batchnorm=with_batchnorm)

  def forward(self, x, y):
    out = torch.cat([x, y], 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.res_block(out)
    return out


class GlobalAveragePool(nn.Module):
  def forward(self, x):
    N, C = x.size(0), x.size(1)
    return x.view(N, C, -1).mean(2).squeeze(2)


class Flatten(nn.Module):
  def forward(self, x):
    return x.view(x.size(0), -1)


def build_stem(feature_dim, module_dim, num_layers=2, with_batchnorm=True,
    kernel_size=3, stride=1, padding=None):
  layers = []
  prev_dim = feature_dim
  if padding is None:  # Calculate default padding when None provided
    if kernel_size % 2 == 0:
      raise(NotImplementedError)
    padding = kernel_size // 2
  for i in range(num_layers):
    layers.append(nn.Conv2d(prev_dim, module_dim, kernel_size=kernel_size, stride=stride,
                            padding=padding))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(module_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = module_dim
  return nn.Sequential(*layers)


def build_classifier(module_C, module_H, module_W, num_answers,
                     fc_dims=[], proj_dim=None, downsample='maxpool2',
                     with_batchnorm=True, dropout=0):
  layers = []
  prev_dim = module_C * module_H * module_W
  if proj_dim is not None and proj_dim > 0:
    layers.append(nn.Conv2d(module_C, proj_dim, kernel_size=1))
    if with_batchnorm:
      layers.append(nn.BatchNorm2d(proj_dim))
    layers.append(nn.ReLU(inplace=True))
    prev_dim = proj_dim * module_H * module_W
  if 'maxpool' in downsample or 'avgpool' in downsample:
    pool = nn.MaxPool2d if 'maxpool' in downsample else nn.AvgPool2d
    if 'full' in downsample:
      if module_H != module_W:
        assert(NotImplementedError)
      pool_size = module_H
    else:
      pool_size = int(downsample[-1])
    # Note: Potentially sub-optimal padding for non-perfectly aligned pooling
    padding = 0 if ((module_H % pool_size == 0) and (module_W % pool_size == 0)) else 1
    layers.append(pool(kernel_size=pool_size, stride=pool_size, padding=padding))
    prev_dim = proj_dim * math.ceil(module_H / pool_size) * math.ceil(module_W / pool_size)
  if downsample == 'aggressive':
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    layers.append(nn.AvgPool2d(kernel_size=module_H // 2, stride=module_W // 2))
    prev_dim = proj_dim
    fc_dims = []  # No FC layers here
  layers.append(Flatten())
  for next_dim in fc_dims:
    layers.append(nn.Linear(prev_dim, next_dim))
    if with_batchnorm:
      layers.append(nn.BatchNorm1d(next_dim))
    layers.append(nn.ReLU(inplace=True))
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
    prev_dim = next_dim
  layers.append(nn.Linear(prev_dim, num_answers))
  return nn.Sequential(*layers)


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = kaiming_normal
  elif init.lower() == 'uniform':
    init_params = kaiming_uniform
  else:
    return
  for m in modules:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      init_params(m.weight)
