#!/usr/bin/env python
# coding: utf-8

# # 1.Signal Generate

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


amplitude = 2 
frequency = 1
signal_lenght = 2
points = np.linspace(0, signal_lenght, num= signal_lenght * 2000)
signal = amplitude * np.sin(2*np.pi*points*frequency)


# In[6]:


plt.plot(points, signal)
plt.axhline(y=0, color='blue')
plt.title('Sine wave')
plt.xlabel('Time')
plt.ylabel('Amplitude = sin(time)')
plt.grid(True, which='both')


# # 2. Elementory Operation

# # 2.1 Exponential

# In[8]:


import numpy as np
import matplotlib.pyplot as plt


# In[11]:


frequency = 2000
signal_lenght = 2
t = np.linspace(0, signal_lenght, num= signal_lenght * frequency)

Vr_plus = 1500*np.exp(-25*t)*np.sin(t)
Vr_minus = -1500*np.exp(-25*t)*np.sin(t)


# In[13]:


plt.plot(t, Vr_plus)
plt.plot(t, Vr_minus)
plt.title('Exponential wave')
plt.xlabel('Time')
plt.grid(True, which='both')


# # 2.2 Impulse

# In[15]:


import numpy as np
import matplotlib.pyplot as plt


# In[17]:


# sin signal
amplitude = 2 
frequency = 2000
signal_lenght = 2
points = np.linspace(0, signal_lenght, num= signal_lenght * frequency)
signal = amplitude * np.sin(2*np.pi*points)

# impulse
imp = [200, 500, 700]
impulse_signal = np.zeros(2*frequency)
for x in imp:
    impulse_signal[x] = 1

# sin * impulse
impulse_sin_signla = signal * impulse_signal


# In[19]:


plt.figure(figsize=(10,10))
plt.subplot(3,1,1).plot(points, signal)
plt.title("Sin signal")
plt.subplot(3,1,2).plot(points, impulse_signal)
plt.title("Impulse")
plt.subplot(3,1,3).plot(points, impulse_sin_signla)
plt.title("Sin signal with impulse")


# # 2.3 Unit Step

# In[21]:


import numpy as np
import matplotlib.pyplot as plt


# In[23]:


# sin signal
amplitude = 2 
frequency = 2000
signal_lenght = 2
t = np.linspace(-1, 1, num= signal_lenght * frequency)
sin_signal = amplitude * np.sin(2*np.pi*points)

# Unit
unit_signal = np.zeros(2*frequency)
unit_signal[frequency:] = 1

# sin * unit
unit_sin_signal = sin_signal * unit_signal


# In[25]:


plt.figure(figsize=(10,10))
plt.subplot(3,1,1).plot(t, sin_signal)
plt.title("Sin signal")
plt.subplot(3,1,2).plot(t, unit_signal)
plt.title("Unit")
plt.subplot(3,1,3).plot(t, unit_sin_signal)
plt.title("Sin signal with Unit")


# # 2.4 RAM 

# In[27]:


import numpy as np
import matplotlib.pyplot as plt


# In[29]:


# sin signal
amplitude = 2 
frequency = 2000
signal_lenght = 2
t = np.linspace(-1, 1, num= signal_lenght * frequency)
sin_signal = amplitude * np.sin(2*np.pi*points)

# Unit
ram_signal = np.zeros(2*frequency)
ram_signal[frequency:] = 1
ram_signal[frequency:] *= t[frequency:]

# sin * unit
ram_sin_signal = sin_signal * ram_signal


# In[31]:


plt.figure(figsize=(10,10))
plt.subplot(3,1,1).plot(t, sin_signal)
plt.title("Sin signal")
plt.subplot(3,1,2).plot(t, ram_signal)
plt.title("RAM")
plt.subplot(3,1,3).plot(t, ram_sin_signal)
plt.title("Sin signal with RAM")


# # 3. Signal Operation

# # 3.1 Addition

# In[33]:


import numpy as np
import matplotlib.pyplot as plt


# In[35]:


# signal 1
t1 = [-3,-2,-1,0,1]
a1 = [1,3,0,2,-2]

# signal 2
t2 = [-1,0,1,2,3]
a2 = [-2,1,2,1,-2]

# signal additon
t_min = np.min([np.min(t1), np.min(t2)])
t_max = min = np.max([np.max(t1), np.max(t2)])

# total t length
t_length = abs(t_min)+abs(t_max) + 1

if np.min(t1) > np.min(t2):
    t1, t2 = t2, t1
    a1, a2 = a2, a1

a1_modify = np.zeros(t_length)
a1_modify[:len(a1)] = a1

a2_modify = np.zeros(t_length)
a2_modify[-len(a2):] = a2

a = a1_modify + a2_modify
t = np.arange(t_min, t_max+1)


# In[40]:


plt.figure(figsize=(5,10))
plt.subplot(3,1,1).stem(t1, a1)
plt.title("signal 1")
plt.subplot(3,1,2).stem(t2, a2)
plt.title("signal 2")
plt.subplot(3,1,3).stem(t, a)
plt.title("Addition signal")


# # 3.2 Subtraction

# In[43]:


import numpy as np
import matplotlib.pyplot as plt


# In[44]:


# signal 1
t1 = [-3,-2,-1,0,1]
a1 = [1,3,0,2,-2]

# signal 2
t2 = [-1,0,1,2,3]
a2 = [-2,1,2,1,-2]

# signal additon
t_min = np.min([np.min(t1), np.min(t2)])
t_max = min = np.max([np.max(t1), np.max(t2)])

# total t length
t_length = abs(t_min)+abs(t_max) + 1

if np.min(t1) > np.min(t2):
    t1, t2 = t2, t1
    a1, a2 = a2, a1

a1_modify = np.zeros(t_length)
a1_modify[:len(a1)] = a1

a2_modify = np.zeros(t_length)
a2_modify[-len(a2):] = a2

a = a1_modify - a2_modify
t = np.arange(t_min, t_max+1)


# In[46]:


plt.figure(figsize=(5,10))
plt.subplot(3,1,1).stem(t1, a1)
plt.title("signal 1")
plt.subplot(3,1,2).stem(t2, a2)
plt.title("signal 2")
plt.subplot(3,1,3).stem(t, a)
plt.title("Subtraction signal")


# # 3.3 Multiplication

# In[48]:


import numpy as np
import matplotlib.pyplot as plt


# In[50]:


# signal 1
t1 = [-3,-2,-1,0,1]
a1 = [1,3,0,2,-2]

# signal 2
t2 = [-1,0,1,2,3]
a2 = [-2,1,2,1,-2]

# signal additon
t_min = np.min([np.min(t1), np.min(t2)])
t_max = min = np.max([np.max(t1), np.max(t2)])

# total t length
t_length = abs(t_min)+abs(t_max) + 1

if np.min(t1) > np.min(t2):
    t1, t2 = t2, t1
    a1, a2 = a2, a1

a1_modify = np.zeros(t_length)
a1_modify[:len(a1)] = a1

a2_modify = np.zeros(t_length)
a2_modify[-len(a2):] = a2

a = a1_modify * a2_modify
t = np.arange(t_min, t_max+1)


# In[52]:


plt.figure(figsize=(5,10))
plt.subplot(3,1,1).stem(t1, a1)
plt.title("signal 1")
plt.subplot(3,1,2).stem(t2, a2)
plt.title("signal 2")
plt.subplot(3,1,3).stem(t, a)
plt.title("mul signal")


# # 3.4 Shifting

# In[54]:


import numpy as np
import matplotlib.pyplot as plt


# In[56]:


# signal 
t1 = [-3,-2,-1,0,1]
a1 = [1,3,0,2,-2]

# time shifting
st = -3
t = np.zeros(len(t1)+abs(st))
a = np.zeros(len(t1)+abs(st))
if st < 0:
    t[:len(t1)] = t1
    for x in range(len(t1), len(t), 1):
        t[x] = t[x-1] + 1
    a[-len(t1):] = a1
else:
    t[-len(t1):] = t1
    for x in range(len(t)-len(t1)-1, -1, -1):
        t[x] = t[x+1] -1
    a[:len(t1)] = a1


# In[58]:


plt.figure(figsize=(5,10))
plt.subplot(2,1,1).stem(t1, a1)
plt.title("signal")
plt.subplot(2,1,2).stem(t, a)
plt.title(f"shifting signal t={st}")


# # 3.5 Folding

# In[60]:


import numpy as np
import matplotlib.pyplot as plt


# In[62]:


# signal 
t1 = [-3,-2,-1,0,1]
a1 = [1,3,0,2,-2]

# folding
t = [-x for x in t1]
t = t[::-1]
a = a1[::-1]


# In[64]:


plt.figure(figsize=(5,10))
plt.subplot(2,1,1).stem(t1, a1)
plt.title("signal")
plt.subplot(2,1,2).stem(t, a)
plt.title(f"Folding signal")


# # 4. Analog To Digital

# In[66]:


import numpy as np
import matplotlib.pyplot as plt


# In[68]:


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


# In[70]:


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


# # 5. Fourier Transform

# In[72]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# In[74]:


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


# In[76]:


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


# In[78]:


yf = fft(signal)
xf = fftfreq(N, T)[:N//2]


# In[80]:


plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))


# # 6. DFT & IDFT

# # 6.1 DFT

# In[91]:


import numpy as np


# # Matix Form

# In[92]:


x = [3,-1,0,2]
x = np.array(x)
W = np.array(
    [
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ]
)


# In[93]:


X = np.dot(W, x.T)
X


# # Equation

# In[94]:


x = [1,3,-1,-2]
N = len(x)
n = np.arange(N)
k = n.reshape((N, 1))
e = np.exp(-2j * np.pi * k * n / N)


# In[96]:


X = np.dot(e, x)
X


# # 6.2 IDFT

# In[98]:


import numpy as np


# In[100]:


X = np.array([4, 3+3j, 2, 3-3j])
W = np.array(
    [
        [1, 1, 1, 1],
        [1, -1j, -1, 1j],
        [1, -1, 1, -1],
        [1, 1j, -1, -1j]
    ]
)


# In[101]:


x = (1 / X.shape[0] ) * np.dot(W, X.T)
x


# # 7. Convolution

# In[103]:


import numpy as np
import matplotlib.pyplot as plt


# In[104]:


signal = np.repeat([0., 1., 0.,0.,1.,0.], 100)
window = np.hanning(50)
conv_sig = np.convolve(signal, window, mode='same')/np.sum(window)


# In[105]:


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

