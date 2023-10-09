import numpy as np
import matplotlib.pyplot as plt

# Parameters
A = float(input("Amplitude of Transmitting signal: "))

# Transmitting Signal Generation
f = 100
T = 1 / f
t = np.arange(0, 2 * T, T / 100)
y = A * np.sin(2 * np.pi * f * t)

plt.figure(1)
plt.plot(t, y, linewidth=3)
plt.ylabel('Amplitude (volt)')
plt.xlabel('time (Sec)')
plt.title('Transmitting Signal')

# Sampling
Ts = T / 20
Fs = 1 / Ts
n = np.arange(1, int(2 * T / Ts) + 1)
y1 = A * np.sin(2 * np.pi * f * n * Ts)

plt.figure(2)
plt.stem(n, y1)
plt.ylabel('Amplitude (volt)')
plt.xlabel('discrete time')
plt.title('Discrete Time Signal After Sampling')

# Additional of DC Level
y2 = A + y1

plt.figure(3)
plt.stem(n, y2)
plt.ylabel('Amplitude (volt)')
plt.xlabel('discrete time')
plt.title('DC Level + Discrete Time Signal')

# Quantization Signal
y3 = np.round(y2)

plt.figure(4)
plt.stem(n, y3)
plt.ylabel('Amplitude (volt)')
plt.xlabel('discrete time')
plt.title('Quantized Signal')

# Binary Information Generation
y4 = np.array([bin(int(val))[2:].zfill(8) for val in y3])
Bi = y4.tolist()

plt.show()
