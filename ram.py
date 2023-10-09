import numpy as np
import matplotlib.pyplot as plt
# sin signal
amplitude = 2 
frequency = 2000
signal_lenght = 2
points = np.linspace(-1, 1, num= signal_lenght * frequency)
sin_signal = amplitude * np.sin(2*np.pi*points)

# Unit
ram_signal = np.zeros(2*frequency)
ram_signal[frequency:] = 1
ram_signal[frequency:] *= points[frequency:]

# sin * unit
ram_sin_signal = sin_signal * ram_signal
plt.figure(figsize=(10,10))
plt.subplot(3,1,1).plot(points, sin_signal)
plt.title("Sin signal")
plt.subplot(3,1,2).plot(points, ram_signal)
plt.title("RAM")
plt.subplot(3,1,3).plot(points, ram_sin_signal)
plt.title("Sin signal with RAM")
plt.show()