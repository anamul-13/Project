import numpy as np
x = [1,3,-1,-2]
N = len(x)
n = np.arange(N)
k = n.reshape((N, 1))
e = np.exp(-2j * np.pi * k * n / N)
X = np.dot(e, x)
X