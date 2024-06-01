# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 14:58:52 2024

@author: suhai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
 
def level(n,n_g):
    lev = ((n)**2 -2*n*n_g + (n_g)**2)
    return lev

def e(E_j,E_c):
    return (-E_j/(2*E_c))*np.ones(1000)

 ### part a
 
h = 6.63e-34
 
E_c = (500e6)*h
E_j=  [125e6*h, 125e7*h, 125e8*h]
n_g = np.linspace(-0.5,0.5,1000)
zero = np.zeros(1000)
ejvals = ['125Mhz','1.25Ghz','12.5Ghz']
#plot into 3 by 1 subplot
fig, axes = plt.subplots(1,3, figsize=(10,5))
for p in range(3):
    
    ej = e(E_j[p],E_c)
    a = np.array([[level(-1,n_g), ej, zero],
               [ej, level(0,n_g), ej],
               [zero, ej, level(1,n_g)]]).T
    eigenvalues, eigenvectors = LA.eig(a)
    
    #to prevent the graph from jumping
    eigenvalues = np.sort(eigenvalues)
    
    #plot results
    for i in range(3):
        E = eigenvalues[:,i]
        axes[p].plot(n_g, E , label = 'E'+str(i))
        axes[p].set_xlabel('$n_g$')
        axes[p].set_ylabel('$E [E_c]$')
        axes[p].title.set_text(r'$\frac{E_J}{h} =$'+str(ejvals[p]))
        axes[p].legend()
        

  
fig.suptitle('Q1 part a')
fig.tight_layout()
#plt.savefig('Q1_part.jpg')
 #%%
###part b
ratio = [0.25, 2.5, 25] #E_j/E_c
n_g = 0.5
zero = 0
n = [-1, 0 ,1]
#plot into 3 by 1 subplot
fig, axes = plt.subplots(1,3, figsize=(10,5))
for p in range(3):
    
    ej = -0.5*(ratio[p])
    a = np.array([[level(-1, n_g), ej, zero],
               [ej, level(0, n_g), ej],
               [zero, ej, level(1, n_g)]]).T
    eigenvalues, eigenvectors = LA.eig(a)
    
    #probabilities
    c_1  = eigenvectors[0,:]**2
    c_2  = eigenvectors[1,:]**2
    c_3  = eigenvectors[2,:]**2
    
    axes[p].plot(n, c_1 ,label = r'$E_0$')
    axes[p].plot(n, c_2 ,label = r'$E_1$')
    axes[p].plot(n, c_3 ,label = r'$E_2$')
    axes[p].set_xlabel('$n$')
    axes[p].set_ylabel('$probability$')
    axes[p].legend()
    axes[p].title.set_text(str(ejvals[p]))
fig.suptitle('Q1 part b')
fig.tight_layout()
plt.savefig('Q1_partb.jpg')
    
#%%
 ###part c
empty =[]
for i in range(3):
    ej = -0.5*(ratio[i])
    
    ham = np.array([[level(-1, n_g), ej, zero],
               [ej, level(0, n_g), ej],  
               [zero, ej, level(1, n_g)]]).T
    eigenvalues, eigenvectors = LA.eig(ham)
    
    a = np.abs(eigenvectors[0,:])
    c = np.abs(eigenvectors[2,:])
    #dervive from calculation
    inside = a**2 + c**2 -(-(a**2) + c**2)**2
    
    uncertainty = np.sqrt(inside)
    
    empty.append(uncertainty)
y= np.concatenate(empty).reshape(3,3).T
x = np.repeat(E_j,3).reshape(3,3).T
x = x/E_c
plt.figure()
for i in range(3):
    plt.plot(x[i,:], y[i,:],label = r'E'+str(i))
    plt.xlabel('$E_j[E_c]$')
    plt.ylabel('uncertainty in number state $\Delta n$')
    plt.title('Uncertainty against $E_j$')
    plt.legend()
plt.show()
#plt.savefig('Q1_partc.jpg')
 #%%
 ### Used in part d to compare to analytical solutions.
ej = -0.5*(ratio[0])
ham = np.array([[level(-1, n_g), ej, zero],
               [ej, level(0, n_g), ej],
               [zero, ej, level(1, n_g)]]).T

eigenvalues, eigenvectors = LA.eig(ham)
 #compare values from analytical and numerical
for i in range(3):
    print('E_'+str(i)+' numerical='+str(eigenvalues[i]))