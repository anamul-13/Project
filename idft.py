import numpy as np
X = np.array([4, 3+3j, 2, 3-3j])
W = np.array(
    [
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ]
)
x = (1 / X.shape[0] ) * np.dot(W, X.T)
x