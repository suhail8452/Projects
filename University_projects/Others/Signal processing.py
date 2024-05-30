# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:10:30 2022

@author: suhail
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
#from numpy.fft import fft, ifft
import scipy.fftpack
from scipy import signal
import matplotlib.widgets as widgets
from matplotlib.widgets import RadioButtons
from scipy.fft import fft, fftfreq
import tkinter as tk


#The parametrized function to be plotted
def f(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

def square(t,amplitude,frequency):
    return amplitude*signal.square(2 * np.pi * frequency * t)
    
def sawtooth(t,amplitude,frequency):
    return amplitude*signal.sawtooth(2 * np.pi * frequency * t)

def time(N,T):
    return np.linspace(0,T,N,endpoint=False)

def space(T,N):
    return T/N




# Define initial parameters
init_amplitude = 5
init_frequency = 10
N=100 #number of sample point


#sampling spacing
T=1
dt=space(T,N)



t = np.linspace(0,T,N,endpoint=False)
y=f(t, init_amplitude, init_frequency)

y2=square(t, init_amplitude, init_frequency)

y3=sawtooth(t, init_amplitude, init_frequency)


def wavefunct(label):
    hzdict = {'sine': y, 'square': y2, 'sawtooth': y3}
    ydata = hzdict[label]
    line.set_ydata(ydata)
    yf = scipy.fftpack.fft(ydata)
    plus.set_ydata(2.0/N * np.abs(yf[:N//2]))
    plt.draw()


    
    



# Create the figure and the line that we will manipulate
fig, ax = plt.subplots(1,2,figsize=(10,6))
line, = ax[0].plot(t, y, lw=2)
ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Amplitude [A]')
ax[0].set_title('Signal')
ax[0].set_ylim(-10,10)
ax[0].set_xlim(0,1)


#create fft 

yf = scipy.fftpack.fft(y)
xf = fftfreq(N, dt)[:N//2]



#power spectrum
plus, =ax[1].plot(xf, 2.0/N * np.abs(yf[:N//2]))
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel('Amplitude [A]')
ax[1].set_title('Power spectrum')
ax[1].set_ylim(0,10)
ax[1].set_xlim(0,100)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.1, bottom=0.4)

# Make a horizontal slider to control the frequency.
axfreq = fig.add_axes([0.13, 0.25, 0.3, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label='Frequency [Hz]',
    valmin=0.0,
    valmax=50,
    valinit=init_frequency,
    valstep=0.1,
    orientation="horizontal"
)



# Make a horizontal oriented slider to control the amplitude
axamp = fig.add_axes([0.13, 0.20, 0.3, 0.03])
amp_slider = Slider(
    ax=axamp,
    label="Amplitude [A]",
    valmin=0,
    valmax=10,
    valinit=init_amplitude,
    valstep=0.1,
    orientation="horizontal"
)



# Make a horizontal oriented slider to control the No points
axpoints = fig.add_axes([0.13, 0.15, 0.3, 0.03])
point_slider = Slider(
    ax=axpoints,
    label="No points",
    valmin=0,
    valmax=500,
    valinit=N,
    valstep=1,
    orientation="horizontal"
)

# Make a horizontal oriented slider to control the time
axdur = fig.add_axes([0.13, 0.10, 0.3, 0.03])
duration_slider = Slider(
    ax=axdur,
    label="Duration [s]",
    valmin=0,
    valmax=1,
    valinit=T,
    valstep=0.01,
    orientation="horizontal"
)

rax = fig.add_axes([0.60, 0.1, 0.2, 0.2])
radio = widgets.RadioButtons(rax,('sine','square','sawtooth'))
radio.on_clicked(wavefunct)


# The function to be called anytime a slider's value changes
def update(val):
    t = time(point_slider.val,duration_slider.val)  #duration_slider
    if radio.value_selected=='sine':
        y=f(t, amp_slider.val, freq_slider.val)
    elif radio.value_selected=='square':
        y=square(t, amp_slider.val, freq_slider.val)
    else:
        y=sawtooth(t, amp_slider.val, freq_slider.val)
    line.set_ydata(y)
    line.set_xdata(t)
    fig.canvas.draw_idle()
    
def update1(val):
    dt=space(duration_slider.val,point_slider.val)
    t = time(point_slider.val,duration_slider.val)  
    xf = fftfreq(point_slider.val, dt)[:(point_slider.val)//2]
    if radio.value_selected=='sine':
        y=f(t, amp_slider.val, freq_slider.val)
    elif radio.value_selected=='square':
        y=square(t, amp_slider.val, freq_slider.val)
    else:
        y=sawtooth(t, amp_slider.val, freq_slider.val)
    yf=scipy.fftpack.fft(y)
    plus.set_ydata(2.0/point_slider.val * np.abs(yf[:point_slider.val//2]))
    plus.set_xdata(xf)




# register the update function with each slider
freq_slider.on_changed(update)
amp_slider.on_changed(update)
point_slider.on_changed(update)
duration_slider.on_changed(update)
radio.on_clicked(update)

freq_slider.on_changed(update1)
amp_slider.on_changed(update1)
point_slider.on_changed(update1)
duration_slider.on_changed(update1)
radio.on_clicked(update1)


# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.6, 0.025, 0.08, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

#radiobutton






def reset(event):
    freq_slider.reset()
    amp_slider.reset()
    point_slider.reset()
    duration_slider.reset()
button.on_clicked(reset)

plt.show()


  
# Function for closing window

def closeCallback(event):
     plt.close('all') 

    
# Create a `matplotlib.widgets.Button` to close figure
closetax = fig.add_axes([0.72, 0.025, 0.08, 0.04])
button1 = Button(closetax, 'Close', hovercolor='0.975')
button1.on_clicked(closeCallback)

  
plt.show()
  
