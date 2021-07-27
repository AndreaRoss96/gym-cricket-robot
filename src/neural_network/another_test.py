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
print(output.shape)
out = torch.flatten(output)
print(out.shape)





# out_t = torch.tensor([[[[[ -5.0105]]],
# [[[  4.5737]]],
# [[[  0.3838]]],
# [[[ -2.8000]]]],
# [[[[ 10.6782]]],
# [[[  9.9930]]],
# [[[  9.1778]]],
# [[[ -1.7229]]]],
# [[[[  4.4293]]],
# [[[ -5.2465]]],
# [[[ -0.1998]]],
# [[[  2.4060]]]],
# [[[[-11.2595]]],
# [[[-10.6658]]],
# [[[ -8.9938]]],
# [[[  1.3289]]]]])

# print(torch.flatten(out_t).shape)