import torch
from torch._C import dtype, import_ir_module
import torch.nn as nn
import pywavefront as pw
import numpy as np

scene = pw.Wavefront('/home/andrea/Downloads/flat.obj')
bello = torch.FloatTensor(scene.vertices)
bello = np.reshape(bello, (4,3,1,1,1))

# With square kernels and equal stride
m = nn.Conv3d(3, 4, 1, stride=2)
# non-square kernels and unequal stride and with padding
# m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
# input = torch.randn(20, 16, 10, 50, 100)
input = torch.randn(3,4,1,1,1)
# print(input)
output = m(bello)
# print(output.shape)
out = torch.flatten(output)
# print(out.shape)

a = (0.0, 0.0, 0.5)
b = (0.0, -0.0, 0.0)
c = (0.0, 0.0, 0.0)
d = (0.0, 0.0, 0.0)
e = [0.097890350517154, 0.843747480271027, 1.587993373719593, 0.39666778129800706, -0.7711754893171925, 0.22739108172222178, -1.6045572789994034, 1.6941113486008996, 2.3819163639094105, 0.1335265619243593, 3.067059941017999, -3.118246636317355, -3.078253249799098, 1.8504392734572708, -0.8956540557746044, 1.931056474496562]
f = [1.8705314075676993, 2.3861119985595076, -2.059730851879511, -2.7226961905919462, 1.870208891662947, -1.824068238873925, 0.9655597606580093, -2.2671613816816816]
g = [0.0, 0.0, 0.0, 0.0]
h = np.array([0., 0., 0., 0., 0.], dtype=np.float32)

cazzo = [a,b,c,d,e,f,g,h]
tmp = []
for elem in cazzo:
    tmp += [*elem]

cazzo = [z for elem in cazzo for z in [*elem]]
print(f'cazzo {cazzo}')
i = [*a, *b, *c, *d, *e, *f, *g, *h]
i = torch.tensor(i)
print(i)