import numpy as np
import matplotlib.pyplot as plt

# Input the length of the sequence and the sequence itself
N = int(input('Enter the length of sequence: '))
x = np.array(input('Enter the sequence (comma-separated): ').split(','), dtype=float)

# Create arrays for n and k
n = np.arange(0, N)
k = np.arange(0, N)

# Calculate wN
wN = np.exp(-1j * 2 * np.pi / N)

# Create the nk matrix
nk = np.outer(n, k)
        
# Calculate wNnk
wNnk = wN ** nk

# Calculate Xk
Xk = np.dot(x, wNnk)

print('Xk:')
print(Xk)

# Calculate magnitude and phase
mag = np.abs(Xk)
phase = np.angle(Xk)

# Plot magnitude
plt.subplot(2, 1, 1)
plt.stem(k, mag)
plt.grid(True)
plt.xlabel('k')
plt.title('MAGNITUDE OF FOURIER TRANSFORM')
plt.ylabel('Magnitude')

# Plot phase
plt.subplot(2, 1, 2)
plt.stem(k, phase)
plt.grid(True)
plt.xlabel('k')
plt.title('PHASE OF FOURIER TRANSFORM')
plt.ylabel('Phase')

plt.tight_layout()
plt.show()
