# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:53:07 2024

@author: suhai
"""

#Power law scaling for EBL
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# find the values of the parameters
def PowerLaw(r,nu,v,gamma,sigma,beta):
    coefficent = 1/(np.pi*(1+ nu))
    first_term = (v)**(-2) / (1 + (r/gamma)*sigma)
    second_term =  nu/(beta**2)  *np.exp(-(r/beta)**2)
    
    wave = coefficent*(first_term + second_term)
    
    return wave

pix_size =int(1000)

#First image: shots
shots = np.zeros((pix_size,pix_size))

dose = 75

shots[0::3, 0::3] = dose

#second image: gaussian

exp = np.zeros((pix_size,pix_size))
sim = np.zeros((pix_size,pix_size))

cx = int(pix_size/2)
cy = int(pix_size/2)


#pixel distance from each other is 1nm
###parameter used for the experiment
#gamma short range
gamma = 8.01

#gaussian weight by sigma and nu
sigma,nu = 2.6,0.75

#long range effect
beta = 28500

#chosen to normalise PSF
v = 18.3

###parameter used for the simulation
#gamma short range
gamma1 = 3.07

#gaussian weight by sigma and nu
sigma1,nu1 = 4.45,0.22

#long range effect
beta1 = 28500

#chosen to normalise PSF
v1 = 3.48

for x in range(pix_size):
    for y in range(pix_size):
        r = (x-cx)**2 + (y-cx)**2
        exp[x,y]= PowerLaw(r,nu,v,gamma,sigma,beta)
        sim[x,y]= PowerLaw(r,nu1,v1,gamma1,sigma1,beta1)

#convolves
EBLexp = signal.convolve(shots,exp)
EBLsim = signal.convolve(shots,sim)

#figures
plt.figure()
plt.title('experiment')
plt.imshow(EBLexp)
plt.colorbar()

plt.figure()
plt.title('Simulation')
plt.imshow(EBLsim)
plt.colorbar()

#cut out the areas that are more than one
strate = np.where(EBLsim>1, 1, 0)
plt.figure()
plt.imshow(strate)
plt.colorbar()
plt.title('Cut-out')

#%%

#recreate the figures in the paper
#number of pixel of the image
#only consider line of sight of distribution since symetrical
pix_size =int(1e6)


cx = int(pix_size/2)
cy = int(pix_size/2)


#optimize code from above

x = np.linspace(0,pix_size,pix_size + 1)
y = np.linspace(0,pix_size,pix_size + 1)

r = np.sqrt((x[:]-cx)**2 + (y[:]-cx)**2)

r1 = r[cx:pix_size+1] 


expLOS = PowerLaw(r1[:],nu,v,gamma,sigma,beta)
simLOS = PowerLaw(r1[:],nu1,v1,gamma1,sigma1,beta1)


#plot a graph of eqn 5 by the distance from the radius
#radially symmetric, take line of sight of both images
 
r2 = r[cx:pix_size+1] * 1e-3   #radius in micrometers

#plot logarithmic graphs
plt.figure()
plt.semilogx(r2,expLOS,'r')
plt.title('Experiment')
plt.xlabel('radius (micrometers)')
plt.ylabel('intensity')
plt.figure()
plt.semilogx(r2,simLOS,'g')
plt.title('Simulation')
plt.xlabel('radius (micrometers)')
plt.ylabel('intensity')


