import numpy as np
import matplotlib.pyplot as plt
signal = np.repeat([0., 1., 0.,0.,1.,0.], 100)
window = np.hanning(50)
conv_sig = np.convolve(signal, window, mode='same')/np.sum(window)
plt.figure(figsize=(10,10))

# main signal
plt.subplot(3,1,1).plot(signal)
plt.title("Signal")

# window signal (gaussian filter)
plt.subplot(3,1,2).plot(window)
plt.title("Window")

# Filtred or windowing or convoluted signal
plt.subplot(3,1,3).plot(conv_sig)
plt.title("Convoluted signal")
plt.show()