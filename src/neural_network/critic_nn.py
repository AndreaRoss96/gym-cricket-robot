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

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, terrain_dim, terrain_output, hidden_layers = [400,300,200], conv_layers = [], kernel_sizes = [150,100,50], action_features_out = 5, output_critic = None, init_w=3e-3):
        super(Critic, self).__init__()
        if output_critic == None:
            output_critic = round(action_dim/2)
        # State features
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.obs_dim, hidden_layers[0]))
        for i in range(0, len(hidden_layers)):
            if (i + 1) < len(hidden_layers):
                self.layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            else :
                # self.layers.append(nn.Linear(hidden_layers[i] + terrain_output**2 + action_features_out, round(action_dim/2)))
                self.output_layer = nn.Linear(hidden_layers[i] + terrain_output**2 + action_features_out, output_critic)
        self.init_weights(init_w)

        # Terrain features
        self.terrain_dim = terrain_dim
        self.terrain_output = terrain_output

        self.conv_layers = nn.ModuleList()
        if len(conv_layers) == 0 :
            self.conv_layers.append(nn.Conv3d(terrain_dim,terrain_output, kernel_sizes[0]))
        else :
            self.conv_layers.append(nn.Conv3d(terrain_dim,conv_layers[0],kernel_sizes[0]))
            for i in range(0, len(conv_layers)):
                if (i+1) < len(conv_layers):
                    self.conv_layers.append(nn.Conv3d(conv_layers[i], conv_layers[i+1], kernel_sizes[i+1]))
                else :
                    self.conv_layers.append(nn.Conv3d(conv_layers[i], self.terrain_output, kernel_sizes[i]))

        # Action features
        self.action_layer = nn.Linear(action_dim, action_features_out)

    def init_weights(self, init_w):
        for layer in self.layers[:-1]:
            layer.weight.data = fanin_init(layer.weight.data.size())
        self.layers[-1].weight.data.uniform_(-init_w, init_w)


    def forward(self, observations, terrain, action):
        print('IIIIIIIIIIIIIIIII'*100)
        print(terrain)
        print(terrain.shape)
        #x, a = observations
        # print(f'x {x}')
        # print(f'a {a}')
        # observation forward
        out = self.layers[0](observations)
        for layer in self.layers[1:]:
            out = nn.ReLU()(out)
            out = layer(out)

        # terrain forward
        out_t = self.conv_layers[0](terrain)
        for layer in self.conv_layers[1:]:
            out_t = nn.ReLU()(out_t)
            out_t = layer(out_t)
        out_t = torch.flatten(out_t)

        tmp = []
        for _ in range(out.shape[0]):
            tmp.append(out_t.data.cpu().numpy())
            # out_t = torch.cat((out_t,out_t))
        if out_t.is_cuda :
            out_t = torch.FloatTensor(tmp).to(torch.device('cuda'))
        
        # action features
        out_a = self.action_layer(action)

        # add layer
        out = torch.cat((out,out_t,out_a), dim=1)

        # output layer
        # out = self.layers[-1](out)
        out = self.output_layer(out)
        out = nn.ReLU()(out)

        return out.mean()

    # def cvv_forward(self, data):
    #     out = self.conv_layers[0](data)
    #     for layer in self.conv_layers[1:]:
    #         out = nn.ReLU()(out)
    #         out = layer(out)
    #     out = nn.Tanh()(out)

    #     return out