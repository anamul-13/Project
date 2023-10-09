import numpy as np
x = [3,-1,0,2]
x = np.array(x)
W = np.array(
    [
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ]
)
X = np.dot(W, x.T)
X
