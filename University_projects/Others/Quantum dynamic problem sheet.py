# -*- coding: utf-8 -*-
"""
Spyder Editor

"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA

def level(n,n_g):
    lev = ((n)**2 - 2*n*n_g + (n_g)**2)
    return lev

def e(E_j,E_c):
    return (-E_j/(2*E_c))*np.ones(100)

#part a
h = 6.63e-34
E_c = (500e6)*h

E_j=  [125e6*h, 125e7*h, 125e8*h]


n_g = np.linspace(-0.5,0.5,100)

zero = np.zeros(100)

ejvals = ['125Mhz','1.25Ghz','12.5Ghz']

#plot into 3 by 1 subplot
fig, axes = plt.subplots(1,3, figsize=(10,5))
for p in range(3):
    
    ej = e(E_j[0],E_c)

    a = np.array([[level(-1, n_g), ej, zero],
               [ej, level(0, n_g), ej],
               [zero, ej, level(1, n_g)]]).T

    eigenvalues, eigenvectors = LA.eig(a)

    eigenvalues = np.sort(eigenvalues)
    
    #eigenvectors need to match eigenvalues

    #plot results
    for i in range(3):
        E = eigenvalues[:,i]
        axes[p].plot(n_g, E , label = 'E'+str(i))
        axes[p].set_xlabel(f'$n_g$')
        axes[p].set_ylabel(f'$E [E_c]$')
        axes[p].title.set_text(r'$\frac{E_J}{h} =$'+str(ejvals[p]))
        axes[p].legend()
        
fig.tight_layout()

#%%

#part b

e = [f'$|<n|E_0>|^2$',f'$|<n|E_1>|^2$',f'$|<n|E_2>|^2$']
ejvals = [r'$\frac{E_J}{h} =125Mhz$',r'$\frac{E_J}{h} =1.25Ghz$',r'$\frac{E_J}{h} =12.5Ghz$']

# plot into three by three subplots
fig, axes = plt.subplots(3,3, figsize=(10,10))
for q in range(3):
    for p in range(3):
        ej = E_j[q]*np.ones(100)

        a = np.array([[level(0, E_c, n_g), ej, zero],
               [ej, level(1, E_c, n_g), ej],
               [zero, ej, level(2, E_c, n_g)]]).T

        eigenvalues, eigenvectors = LA.eig(a)

        for i in range(3):
            prob = eigenvectors[:,i,p]**2
            axes[q,p].plot(n_g, prob , label = r'$<$'+str(i)+'|')
            axes[q,p].set_xlabel(f'$n_g$')
            axes[q,p].set_ylabel(f'$probability$')
            axes[q,p].title.set_text(str(e[p])+','+str(ejvals[q]))
            axes[q,p].legend()

fig.tight_layout()

#%%

#part c

#lets consider for 125Mhz and ng =0.5

n_g = 0.5

E_j =  [125e6*h, 125e7*h, 125e8*h]

E_c = (500e6)*h

zero = 0 #np.zeros(3)


empty =[]

for i in range(3):
    ej = E_j[i] 
    
    a = np.array([[level(0, E_c, n_g), ej, zero],
               [ej, level(1, E_c, n_g), ej],
               [zero, ej, level(2, E_c, n_g)]]).T


    eigenvalues, eigenvectors = LA.eig(a)
    
    b = np.abs(eigenvectors[1,:])
    c = np.abs(eigenvectors[2,:])


    inside = -(b)**4 -4*(b*c)**2 -4*(c)**4 +(b)**2 +(c)**2  
    
    uncertainty = np.sqrt(np.abs(inside))
    
    empty.append(uncertainty)



y= np.concatenate(empty).reshape(3,3).T

x = np.repeat(E_j,3).reshape(3,3).T

x = x/E_c
plt.figure()
for i in range(3):
    plt.plot(x[i,:], y[i,:],label = 'E_'+str(i))
    plt.xlabel(f'$E_j[E_c]$')
    plt.ylabel(f'uncertainty in number state $\Delta n$')
    plt.title(f'Uncertainty against $E_j$')
    plt.legend()


#%%

#eigenvalues at n_g = 0 for part a

def level(n, E_c , n_g):
    lev = E_c*((n)**2 - 2*n*n_g + (n_g)**2)
    return lev

#part a
h = 6.63e-34
E_c = (500e6)*h

E_j=  125e8*h


z = -E_j/E_c -0.5*np.sqrt(2*E_j/E_c) -3/48

x = -E_j/E_c -1.5*np.sqrt(2*E_j/E_c) -15/48

c = -E_j/E_c -2.5*np.sqrt(2*E_j/E_c) -39/48

n_g = 0

zero = 0

ej = E_j

a = np.array([[level(0, E_c, n_g), ej, zero],
               [ej, level(1, E_c, n_g), ej],
               [zero, ej, level(2, E_c, n_g)]]).T /E_c

eigenvalues, eigenvectors = LA.eig(a)




b = np.array([[z, 0, -0.1767],
               [0, x, 0],
               [-0.1767, 0, c]]).T 

eigenvaluesb, eigenvectorsb = LA.eig(b)

#compare values from analytical and numerical
for i in range(3):
    print('E_'+str(i)+' numerical='+str(eigenvalues[i]))
    print('E_'+str(i)+' analytical='+str(eigenvaluesb[i]))
    print('')




