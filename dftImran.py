import numpy as np
import matplotlib.pyplot as plt

Fa = 10
T = 1 / Fa
t = np.arange(0, T, T/99)
y = 5 * np.sin(2 * np.pi * Fa * t) + 2 * np.sin(2 * np.pi * 2 * Fa * t) + 2 * np.sin(2 * np.pi * 3 * Fa * t)
plt.figure(1)
plt.plot(t, y)

Fs = 640
Ts = 1 / Fs
N = int(T / Ts)
n = np.arange(0, N)
yy = 5 * np.sin(2 * np.pi * Fa * n * Ts) + 2 * np.sin(2 * np.pi * 2 * Fa * n * Ts) + 2 * np.sin(2 * np.pi * 3 * Fa * n * Ts)
plt.figure(2)
plt.stem(n, yy)

h = []
b = []
for k in range(1, N+1):
    for n in range(1, N+1):
        ff = yy[n-1] * np.exp(-1j * 2 * np.pi * (k-1 - (N/2)) * (n-1 - (N/2)) / N)
        h.append(ff)
    p = sum(h)
    b.append(p)
    h = []

plt.figure(4)
f = Fs * np.arange(-N/2, N/2) / N
plt.stem(f, np.abs(b))
plt.axis([-30, 30, 0, 160])

plt.show()
