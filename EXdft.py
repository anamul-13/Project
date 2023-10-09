import numpy as np
import matplotlib.pyplot as plt

Xk_str = input('Enter X(K) (space-separated values): ')
Xk = np.array([complex(x) for x in Xk_str.split()])

N = len(Xk)  # Get the number of elements in Xk

xn = np.zeros(N, dtype=complex)
k = np.arange(N)
n = range(N)

for nn in n:
    xn[nn] = np.sum(np.exp(1j * 2 * np.pi * k * nn / N) * Xk)

xn /= N

print('x(n) =')
print(xn)

plt.figure()
plt.stem(k, xn.real)
plt.xlabel('n')
plt.ylabel('Magnitude')
plt.title('IDFT OF A SEQUENCE')
plt.grid(True)
plt.show()
