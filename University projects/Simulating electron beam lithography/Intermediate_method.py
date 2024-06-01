# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:53:07 2024

@author: suhai
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Powerlaw for the scattering effect of the electrons
def PowerLaw(r,nu,v,gamma,sigma,beta):
    coefficent = 1/(np.pi*(1+ nu))
    first_term = (v)**(-2) / (1 + (r/gamma)*sigma)
    second_term =  nu/(beta**2)  *np.exp(-(r/beta)**2)
    
    wave = coefficent*(first_term + second_term)
    
    return wave


# 10 nm for each pixels shown therfore,
# the grid size is 1.01 micrometers by 1.01 micrometers
pix = 10 #nm
pix_size =int(101)

#First image: shots
shots = np.zeros((pix_size,pix_size))

# Doseage and shots where the dosage is applied
dose = 1
shots[30:60,49:52] =  dose


# Second image: gaussian using experimental and simulation values
exp, sim = np.zeros((pix_size,pix_size)), np.zeros((pix_size,pix_size))
# centre of the image
cx,cy = int(pix_size/2), int(pix_size/2)


# Pixel distance from each other is 10nm
# Parameter used for the experiment

# Gamma short range
gamma = 8.01
# Gaussian weight by sigma and eta
sigma,nu = 2.6,0.75
# Long range effect
beta = 28500 
# Chosen to normalise PSF
v = 18.3 

# Parameter used for the simulation
# Gamma short range
gamma1 = 3.07
# Gaussian weight by sigma and nu
sigma1,nu1 = 4.45,0.22
# Long range effect
beta1 = 28500
# Chosen to normalise PSF
v1 = 3.48

# Go through each pixel and apply the effect of the scattering of an electron
# at the centre. This is the point square function (PSF)
for x in range(pix_size):
    for y in range(pix_size):
        r = np.sqrt((x-cx)**2 + (y-cx)**2)*10 #increase distance since each pixels is 10nm
        exp[x,y]= PowerLaw(r,nu,v,gamma,sigma,beta)
        sim[x,y]= PowerLaw(r,nu1,v1,gamma1,sigma1,beta1)


# Normalised PSF 
exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
sim = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))

# Convolution of the PSF and the shots
EBLexp = signal.convolve(shots,exp, mode='same')
EBLsim = signal.convolve(shots,sim, mode='same')

# Subplot of the experiment parameters and simulation, along with the cut-out
# for both.

fig = plt.figure()

plt.subplot(2,2,1)
plt.title('Experiment', size=10)
plt.imshow(EBLexp)
plt.colorbar()

plt.subplot(2,2,2)
plt.title('Simulation', size=10)
plt.imshow(EBLsim)
plt.colorbar()

#cut out the areas that are more than one
strate = np.where(EBLexp>1, 1, 0)
plt.subplot(2,2,3)
plt.title('cut-out experiment', size=10)
plt.imshow(strate)
plt.colorbar()

strate = np.where(EBLsim>1, 1, 0)
plt.subplot(2,2,4)
plt.title('cut-out simulation', size=10)
plt.imshow(strate)
plt.colorbar()

plt.tight_layout()

#plt.savefig('simulation_vs_experiment.png')





#%%

# Shows the point spread function as well as the intensity distribution
# from the line of sight from the centre pixel.
fig = plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(exp)
plt.title('Point spread function',size =10)
plt.colorbar(fraction = 0.046)

plt.subplot(1,2,2)
x = np.linspace(0,pix_size-1, pix_size)
y = exp[:,int((pix_size)/2)]
plt.plot(x,y)
plt.title('Intensity distribution' ,size =10)

plt.tight_layout()

#plt.savefig('PSF_Gaussian')

# Use the simulation parameter since they give a better result from the code
# above.

#%%

# Test pattern
# 100nm wide by 10 micro meter space by 400nm, ten rectangles, 
# mentioned in project brief.

# Size of grid
pix_x = 2001
pix_y = 1000
shots = np.zeros((pix_y,pix_x))

#dosage of each of the rectangles
dose = [0.03, 0.02, 0.017, 0.0135, 0.019, 0.014, 0.013, 0.02, 0.02, 0.03]

#centre pixels
cx, cy = int(pix_x/2), int(pix_y/2)

#second image: gauss
exp = np.zeros((pix_y,pix_x))
sim = np.zeros((pix_y,pix_x))

for i in range(10):
    shots[40*i+350:40*i+360, 500:1500:2] = dose[i]
    if i >0 and i<9:
        shots[40*i+350:40*i+360, 750:1250:2] = dose[i]/1.2
        shots[40*i+350:40*i+360, 850:1150:2] = dose[i]/1.4
        shots[40*i+350:40*i+360, 950:1050:2] = dose[i]/1.5


# point spread function
for x in range(pix_x):
    for y in range(pix_y):
        r = np.sqrt((x-cx)**2 + (y-cy)**2)*10  #each pixel is 10nm
        exp[y,x]= PowerLaw(r,nu,v,gamma,sigma,beta)

# Normalize PSF
exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))


# Convolves PSF with shots
EBLexp = signal.convolve(shots,exp, mode='same')

# Subplots for the mask, image, postive and negative resist

fig = plt.figure()
plt.subplot(2,2,1)
plt.imshow(shots)
plt.title('Mask',size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,2)
plt.imshow(EBLexp)
plt.title('EBL experiment' ,size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,3)
cut1 = np.where(EBLexp>1, 1, 0)
plt.imshow(cut1)
plt.title('Negative resist' ,size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,4)
cut2 = np.where(EBLexp<1, 1, 0)
plt.imshow(cut2)
plt.title('Positive resist' ,size =10)
plt.colorbar(fraction=0.023)

plt.tight_layout()

#plt.savefig('Positive_and_negative_resist')

# Positive and negative resist are the different type of resist used 
# for development of substrates in lithography.

#%%

#test pattern and dose correction
# cross
pix_x = 200
pix_y = 200
shots = np.zeros((pix_y,pix_x))
dose = 0.05

#cross in the center
shots[95:106, 60:140] = dose
shots[60:140, 95:106] = dose

cx = int(pix_x/2)
cy = int(pix_y/2)

#second image: gaussian
exp = np.zeros((pix_y,pix_x))

for x in range(pix_x):
    for y in range(pix_y):
        r = np.sqrt((x-cx)**2 + (y-cy)**2)*10  #each pixel is 10nm
        exp[y,x]= PowerLaw(r,nu,v,gamma,sigma,beta)
        
exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
#convolves
EBLexp = signal.convolve(shots,exp, mode='same')
cut1 = np.where(EBLexp>1, 1, 0)



#dose correction
#change the dose of the center pixels
dose = 0.01
#correct horizontal bars and center
shots[95:106, 90:111] = dose
shots[90:111, 95:106] = dose

shots[95:106, 61:65]  = 0.2
shots[95:106, 136:140] = 0.2

#correct vertical bars
shots[61:65, 95:106]  = 0.2
shots[136:140, 95:106]  = 0.2

# PSF
for x in range(pix_x):
    for y in range(pix_y):
        r = np.sqrt((x-cx)**2 + (y-cy)**2)*10  #each pixel is 10nm
        exp[y,x]= PowerLaw(r,nu,v,gamma,sigma,beta)
        
exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
EBLexp1 = signal.convolve(shots,exp, mode='same')

# Subplots
fig = plt.figure()
plt.subplot(2,2,1)
plt.imshow(EBLexp)
plt.title('EBL',size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,3)
cut1 = np.where(EBLexp>1, 1, 0)
plt.imshow(cut1)
plt.title('EBL cut-out' ,size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,2)
plt.imshow(EBLexp1)
plt.title('EBL correct' ,size =10)
plt.colorbar(fraction=0.023)


plt.subplot(2,2,4)
cut2 = np.where(EBLexp1>1, 1, 0)
plt.imshow(cut2)
plt.title('EBl corrected cut-out' ,size =10)
plt.colorbar(fraction=0.023)

plt.tight_layout()

#plt.savefig('Does_correction.png')

# Change the dosage to correct the cut-out shape

#%%

# Create a graph of the intensity against radius for the simulation and
# experiment parameters.

#increae pixel size
pix_size =int(1001)
#First image: shots
shots = np.zeros((pix_size,pix_size))

#dosage
dose = 1
shots[300:600,450:560]=  dose

#second image: PSF

exp = np.zeros((pix_size,pix_size))
sim = np.zeros((pix_size,pix_size))

cx = int(pix_size/2)
cy = int(pix_size/2)


for x in range(pix_size):
    for y in range(pix_size):
        r = np.sqrt((x-cx)**2 + (y-cx)**2)*10 #increase distance since each pixels is 10nm
        exp[x,y]= PowerLaw(r,nu,v,gamma,sigma,beta)
        sim[x,y]= PowerLaw(r,nu1,v1,gamma1,sigma1,beta1)


#normalised PSF
exp = (exp - np.min(exp)) / (np.max(exp) - np.min(exp))
sim = (sim - np.min(sim)) / (np.max(sim) - np.min(sim))

#convolves
EBLexp = signal.convolve(shots,exp, mode='same')
EBLsim = signal.convolve(shots,sim, mode='same')


x = np.linspace(0,pix_size,pix_size + 1)
r = np.sqrt((x[:]-cx)**2) *10 
r1 = r[cx:pix_size+1] 


LOSexp = EBLexp[cx ,cy:pix_size]
LOSsim = EBLsim[cx ,cy:pix_size]
r2 = r[cx:pix_size] * 1e-3 *10  #radius in micrometers

#plot logarithmic graphs
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.loglog(r2,LOSexp,'r')
plt.title('Experiment')
plt.xlabel('Radius from center' + r' $\mu m$')
plt.ylabel('Intensity')

plt.subplot(2,1,2)
plt.loglog(r2,LOSsim,'g')
plt.title('Simulation')
plt.xlabel('Radius from center' + r' $\mu m$')
plt.ylabel('Intensity')

plt.tight_layout()

#plt.savefig('intensity_over_distance')

#%%

### Here is an optimize version of the code above to find the values of
### symmetrical shape used for EBL. This is suppose to find the values from the
### centre of the image to the edge.

# Code runs faster so can use larger grid size
pix_size =int(1e6)
cx = int(pix_size/2)
cy = int(pix_size/2)


#optimize code from above
x = np.linspace(0,pix_size,pix_size + 1)
#y = np.linspace(0,pix_size,pix_size + 1)

r = np.sqrt((x[:]-cx)**2) *10 #+ (y[:]-cx)**2)

r1 = r[cx:pix_size+1] 


expLOS = PowerLaw(r1[:],nu,v,gamma,sigma,beta) 
expLOS = (expLOS - np.min(expLOS)) / (np.max(expLOS) - np.min(expLOS))

simLOS = PowerLaw(r1[:],nu1,v1,gamma1,sigma1,beta1) 
simLOS = (simLOS - np.min(simLOS)) / (np.max(simLOS) - np.min(simLOS))

#plot a graph of eqn 5 by the distance from the radius
#radially symmetric, take line of sight of both images


r2 = r[cx:pix_size+1] * 1e-3 *10  #radius in micrometers

#plot logarithmic graphs
plt.figure(figsize=(6,8))
plt.subplot(2,1,1)
plt.loglog(r2,expLOS,'r')
plt.title('Experiment')
plt.xlabel('Radius from center' + r' $\mu m$')
plt.ylabel('Intensity')

plt.subplot(2,1,2)
plt.loglog(r2,simLOS,'g')
plt.title('Simulation')
plt.xlabel('Radius from center' + r' $\mu m$')
plt.ylabel('Intensity')

plt.tight_layout()

#plt.savefig('intensity_over_distance')
