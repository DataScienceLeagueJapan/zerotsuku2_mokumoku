import numpy as np

a = np.random.randn(3)
print(a.dtype)

b = np.random.randn(3).astype('f')
print(b.dtype)
