# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:27:45 2022

@author: suhail
"""


import numpy as np 
import matplotlib.pyplot as plt


def funct(phi,E,a):
    if x<a:
        V_0=10
    else:
        V_0=0
    return ((-E+(-V_0/(a**2))))*phi


#graph
plt.figure(1,figsize=(6,4))
plt.xlim(0,11)
plt.ylim(-2,2)
plt.xlabel("x")
plt.ylabel("Phi")

#intial conditions
phi=1.000000
dx=0.020000
E=-8.59256728774783

x=0.000000000
v=0.000000000
a=1.000000000


#used to find energy value by narrowing in the value of E
    
while E<-8.592567287747766:
    while x<(a+10):
        
        phi_h=phi +v*(dx/2)
        v_h= v + funct(phi_h, E, a) *(dx/2)

        
        phi= phi+ v_h*(dx)
        v=v+ funct(phi,E,a)*dx
  
        x=x+dx
 
    if np.abs(phi)<0.01:
        x=0.0000000
        phi=1.0000000
        v=0.0000000
        print("Energy value which approach zero= ",E)
        
    x=0.0000000
    phi=1.0000000
    v=0.0000000
    E=E+0.000000000000001
  
    

#graph
plt.figure(1,figsize=(6,4))
plt.xlim(0,3)
plt.ylim(-1,1.5)
plt.xlabel("x")
plt.ylabel("Phi")

#Picked the middle energy values

E=-8.592567287747801

x=0.000000000
v=0.000000000
a=1.000000000
phi=1.22897
dx=0.020000

area=[]
integral=0

while x<(a+2.01):
    phi_h=phi +v*(dx/2)
    v_h= v + funct(phi_h,E,a) *(dx/2)

        
    phi= phi+ v_h*(dx)
    v=v+ funct(phi,E,a)*dx
    
    integral+=(phi**2)*dx
    area.append(integral)
    x=x+dx
    plt.plot(x,phi,"r.")
    plt.pause(0.1)
    

print("The area under the graph is= ",area[-1])
    