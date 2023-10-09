import numpy as np
import matplotlib.pyplot as plt

# Input X(K)
Xk = np.array(input('Enter X(K) (comma-separated): ').split(','), dtype=complex)
N = len(Xk)

xn = np.zeros(N, dtype=complex)
k = np.arange(0, N)
for n in range(N):
    xn[n] = np.sum(np.exp(1j * 2 * np.pi * k * n / N) * Xk)

xn = xn / N

print('x(n):')
print(xn)

plt.figure()
plt.plot(np.real(xn))
plt.grid(True)
plt.stem(k, np.real(xn))
plt.xlabel('n')
plt.ylabel('Magnitude')
plt.title('IDFT OF A SEQUENCE')

plt.tight_layout()
plt.show()
