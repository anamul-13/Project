import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
N = 2000
T = 1./200.
t = np.linspace(0, N*T, N)

# signal1
amplitude_1 = 2
frequency_1 = 1
signal_1 = amplitude_1 * np.sin(2*np.pi*frequency_1*t)

# signal2
amplitude_2 = 1
frequency_2 = 4
signal_2 = amplitude_2 * np.sin(2*np.pi*frequency_2*t)

# signal3
amplitude_3 = 4
frequency_3 = 15
signal_3 = amplitude_3 * np.sin(2*np.pi*frequency_3*t)

# signal
signal = signal_1 + signal_2 + signal_3
plt.figure(figsize=(10,10))
# signal 1
plt.subplot(4,1,1).plot(t,signal_1)
plt.title("signal_1")

# signal 2
plt.subplot(4,1,2).plot(t,signal_2)
plt.title("signal_2")

# signal 2
plt.subplot(4,1,3).plot(t,signal_3)
plt.title("signal_3")

# signal
plt.subplot(4,1,4).plot(t,signal)
plt.title("signal")
plt.show()