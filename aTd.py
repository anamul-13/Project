import numpy as np
import matplotlib.pyplot as plt
# Analog signal
amplitude = 2 
frequency = 1
time_view = 2
t = np.linspace(0, time_view, 2000)
analog_signal = amplitude * np.sin(2*np.pi*frequency*t)

# sampling
smp_rate = frequency * 2
smp_period = 1 / smp_rate
smp_number = time_view / smp_period
smp_t = np.arange(0,time_view, smp_period) + smp_period/2
smp_signal = amplitude * np.sin(2*np.pi*smp_t)


#quantize
quantizing_bits = 4;
quantizing_levels = 2 ** quantizing_bits / 2;
quantizing_step = 1. / quantizing_levels;
quantizing_signal   = np.round (smp_signal / quantizing_step) * quantizing_step
plt.figure(figsize=(5,10))
# analog signal
plt.subplot(3,1,1).plot(t,analog_signal)
plt.subplot(3,1,1).axhline(y=0, color='blue')
plt.title("Analog signal")


# sampling  signal
plt.subplot(3,1,2).plot(t,analog_signal)
plt.subplot(3,1,2).axhline(y=0, color='blue')
plt.subplot(3,1,2).stem(smp_t, smp_signal, linefmt='r-', markerfmt='rs', basefmt='b-')
plt.title("Sampling signal")

# Digial signal
plt.subplot(3,1,3).stem(smp_t, quantizing_signal)
plt.subplot(3,1,3).axhline(y=0, color='blue')
plt.title("Digital signal")
plt.show()