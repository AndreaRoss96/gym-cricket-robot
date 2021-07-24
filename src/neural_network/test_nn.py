import torch
from torch._C import dtype, import_ir_module
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
from torchviz import make_dot
import numpy as np

from actor_nn import Actor

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

a = Actor(5,2, None)

obs = torch.from_numpy(np.random.uniform(low=-3.14, high=3.14, size=(5,)))
# obs = obs.double()
obs = torch.randn(5)
x = a.forward(obs)
print(x)