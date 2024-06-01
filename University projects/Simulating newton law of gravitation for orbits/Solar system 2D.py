# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 18:38:43 2022

@author: suhail
"""

# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
start_time = time.time()


#defintions

def year(n):
    return n*365*24*60*60

def acc(M,x1,x2,distance):
    G = 6.67e-11
    acc= G*M*(x2-x1)/np.abs((distance)**3)
    return acc




#function for odeint    
def pend(intial,t):


    
#empty array for accelerations
    accelerations_x=[]
    accelerations_y=[]

    


    #co-ordinates for x,y,z in 3 rows *(number of masses) columns
    coords= np.array([intial[0:int(N/5)],intial[int(N/5):int((2*N)/5)]])
                       

   
    
    for a in range(go):
        #finding the distance between each mass code
        distant=np.sqrt((coords[0,a]-coords[0,:])**2 + 
                        (coords[1,a]-coords[1,:])**2 )
        
        
   
   
        #acceleration in each plane for each mass

        acc_x= G*M_T[:]*(coords[0,:]-coords[0,a])/np.abs((distant)**3+1e-30)
        acc_y= G*M_T[:]*(coords[1,:]-coords[1,a])/np.abs((distant)**3+1e-30)



        accelerations_x.append(acc_x)
        accelerations_y.append(acc_y)

    
    #sum of acceleration for each body in each plane
    accelerations_x=np.sum(accelerations_x,axis=1)
    accelerations_y=np.sum(accelerations_y,axis=1)

    
   

    #velocity for return function
    vx=intial[int((2*N)/5) : int((3*N)/5)]
    vy=intial[int((3*N)/5) : int((4*N)/5)]


    
    #return list of values
    listp=[vx,vy,accelerations_x,accelerations_y,oppo]
    listp=np.array(np.concatenate( listp , axis=None ))

    
    return listp


#Astronomical unit
Au= 1.496e11
#parsec
pc=3.09e16
#Mass of the sun
M_s=1.99e30
#mass
M=1e24
L=1e9

###These values are the intial conditions for the system and number of bodies 
###in the system

#masses
M_i=[M_s,0.330*M,4.87*M,5.97*M,0.642*M]

#x-position
x=[0,57.9*L,108.2*L,149.6*L,228*L]

#y-position
y=[0,0,0,0,0]


#x-velocity
vx=[0,0,0,0,0]

#y-velocity
vy=[0,47.4e3,35e3,29.8e3,24.1e3]



intial=[x,y,vx,vy,M_i]
intial=np.array(np.concatenate( intial , axis=None ))



#time
t=np.linspace(0,year(2),1000)




intial=np.array(intial)
N=np.size(intial)
#mass values
M_T=intial[int(4*N/5):int(N)]
M_T=np.array(M_T)
go=int(N/5)
G = 6.67e-11

#mass dosen't change, return zero from function
oppo=np.size(M_T)
oppo=np.zeros(oppo)


#solution 
sol = odeint(pend, intial, t,atol=1e-10 ,rtol=1e-10)

#Astronomical unit
Au= 1.496e11
#parsec
pc=3.09e16
#Mass of the sun
M_s=1.99e30
#mass
M=1e24
L=1e9

###These values are the intial conditions for the system and number of bodies 
###in the system

#masses
M_i=[M_s,1898*M,568*M,86.8*M,102*M]

#x-position
x=[0,778.5*L,1432*L,2867*L,4514*L]

#y-position
y=[0,0,0,0,0]


#x-velocity
vx=[0,0,0,0,0]

#y-velocity
vy=[0,13.1e3,9.7e3,6.8e3,5.4e3]



intial=[x,y,vx,vy,M_i]
intial=np.array(np.concatenate( intial , axis=None ))



#time
t1=np.linspace(0,year(165),10000)


intial=np.array(intial)
N=np.size(intial)
#mass values
M_T=intial[int(4*N/5):int(N)]
M_T=np.array(M_T)
go=int(N/5)
G = 6.67e-11

#mass dosen't change, return zero from function
oppo=np.size(M_T)
oppo=np.zeros(oppo)

sol1 = odeint(pend, intial, t1,atol=1e-10 ,rtol=1e-10)



print("--- %s seconds ---" % (time.time() - start_time))

#%%

#plot subplot 

fig, ax=plt.subplots(2,2,figsize=(8,6))

#since N is the same for inner and outer planets code can be simplified


#Top figure, y against x for the two bodies


solar_system1=['Sun','Mercury',"Venus", 'Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
solar_system2=['Sun','Jupiter','Saturn','Uranus','Neptune']
colour=['']

for m in range(int((N)/5)):
    ax[0,0].plot(sol[:, (m)]/Au, sol[:, (m+(int((N)/5)))]/Au ,'C'+str(m),label=solar_system1[m])
        
    ax[1,0].plot(t/year(1),sol[:, (m)]/Au ,'C'+str(m))


    ax[0,1].plot(sol1[:, (m)]/Au, sol1[:, (m+(int((N)/5)))]/Au ,'C'+str(m+5),label=solar_system2[m])
    
    ax[1,1].plot(t1/year(1),sol1[:, (m)]/Au,'C'+str(m+5))



ax[0,0].set_xlabel('x [Au]',fontsize=14)
ax[0,0].set_ylabel('y [Au]',fontsize=14)
ax[0,0].legend(loc='best')

ax[0,1].set_xlabel('x [Au]',fontsize=14)
ax[0,1].set_ylabel('y [Au]',fontsize=14)
ax[0,1].legend(loc='best')

ax[1,0].set_xlabel('t [yrs]',fontsize=14)
ax[1,0].set_ylabel('x [Au]',fontsize=14)


ax[1,1].set_xlabel('t [yrs]',fontsize=14)
ax[1,1].set_ylabel('x [Au]',fontsize=14)

plt.tight_layout()

