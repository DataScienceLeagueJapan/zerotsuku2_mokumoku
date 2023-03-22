import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from numpy.random import randn

from common.layers import MatMul

c0 = np.array([1, 0, 0, 0, 0, 0, 0])
c1 = np.array([0, 0, 1, 0, 0, 0, 0])

W_in  = randn(7, 3)
W_out = randn(3, 7)

in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = (h0 + h1) / 2
s = out_layer.forward(h)
print(s)



