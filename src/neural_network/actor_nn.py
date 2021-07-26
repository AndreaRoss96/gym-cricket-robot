import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, terrain_dim, terrain_output, hidden_layers = [400,300,200], conv_layers = [800,300], kernel_sizes = [150,100,50], init_w=3e-3):
        super(Actor, self).__init__()
        
        # Action features
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.layers = []
        self.layers.append(nn.Linear(self.obs_dim, hidden_layers[0]))
        for i in range(0, len(hidden_layers)):
            if (i + 1) < len(hidden_layers):
                self.layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            else :
                self.layers.append(nn.Linear(hidden_layers[i], self.action_dim))
        self.init_weights(init_w)

        # Terrain features
        self.terrain_dim = terrain_dim
        self.terrain_output = terrain_output

        self.conv_layers = []
        self.conv_layers.append(nn.Conv2d(terrain_dim,conv_layers[0],kernel_sizes[0]))
        for i in range(0, len(conv_layers)):
            if (i+1) < len(conv_layers):
                self.conv_layers.append(nn.Conv2d(conv_layers[i], conv_layers[i+1], kernel_sizes[i+1]))
            else :
                self.conv_layers.append(nn.Conv2d(conv_layers[i], self.terrain_output, kernel_sizes[i])) 


    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fanin_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)


    def forward(self, observations):
        out = self.layers[0](observations)
        for layer in self.layers[1:]:
            print(layer)
            out = nn.ReLU()(out)
            out = layer(out)
        out = nn.Tanh()(out)

        return out 