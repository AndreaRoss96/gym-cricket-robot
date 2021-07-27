import torch
from torch._C import dtype, import_ir_module
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
from torchviz import make_dot
import numpy as np
import pywavefront as pw
from torch import autograd

from actor_nn import Actor
from critic_nn import Critic

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)
        self.cnn.fc = nn.Linear(
            self.cnn.fc.in_features, 20)
        
        self.fc1 = nn.Linear(20 + 10, 60)
        self.fc2 = nn.Linear(60, 5)
        
    def forward(self, image, data):
        x1 = self.cnn(image)
        x2 = data
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        

# model = MyModel()

# batch_size = 2
# image = torch.randn(batch_size, 3, 299, 299)
# data = torch.randn(batch_size, 10)

# output = model(image, data)
# make_dot(output, image+data)

# a = Actor(5,2, None)

# obs = torch.from_numpy(np.random.uniform(low=-3.14, high=3.14, size=(5,)))
# # obs = obs.double()
# obs = torch.randn(5)
# x = a.forward(obs)
# print(x)

scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
# print(dir(scene))
# print(scene.vertices)
# print()
terrain = np.array(scene.vertices)
terrain = np.reshape(terrain, (4,3,1,1,1))
terrain = torch.FloatTensor(terrain)

# The value of the first dimension of kernel_size is the number of image frames processed each time, followed by the size of the convolution kernel
# m = nn.Conv3d(3, 3, (3, 7, 7), stride=1, padding=0)
# input = autograd.Variable(torch.randn(1, 3, 7, 60, 40))
# print('input')
# print(input[0][0][0])
# output = m(input)
# print(output.size())

a = Critic(50,12, terrain_dim=terrain.shape[1], terrain_output=terrain.shape[0],conv_layers=[],kernel_sizes=[1])

obs = torch.from_numpy(np.random.uniform(low=-3.14, high=3.14, size=(5,)))
obs = torch.randn(50)
action = torch.rand(12)

out = a.forward(obs, terrain, action)
print(out)