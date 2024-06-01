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



#function for a year
def year(n):
    return n*365*24*60*60



#function for odeint    
def pend(intial,t):


    #empty array for accelerations
    accelerations=[]
    

    #co-ordinates for x,y,z in 3 rows *(number of masses) columns
    coords=  np.array([intial[0:int(3*N/7)]]) 
    coords= coords.reshape(3,go)


   
    
    for a in range(go):
        #finding the distance between each mass code
        distant=np.sqrt((coords[0,a]-coords[0,:])**2 + 
                        (coords[1,a]-coords[1,:])**2 +
                        (coords[2,a]-coords[2,:])**2) 
        
 
        #acceleration in each plane for each mass
        
        acc= G*M_T[:]*(coords[0,:]-coords[0,a],coords[1,:]-coords[1,a], 
                             coords[2,:]-coords[2,a])/np.abs((distant)**3 +1e-9)
        
        
        accelerations.append(acc)
    
    #sum of acceleration for each body in each plane
    accelerations =np.concatenate( accelerations , axis=0 )

    c=(np.sum( accelerations, axis=1)).reshape((go,3)).T

    accelerations =np.concatenate( c , axis=0 )
 


    #velocity for return function
    velocity= intial[3*go:6*go]

    
    #return list of values
    listp=[velocity,accelerations,oppo]
    listp=np.array(np.concatenate( listp , axis=None ))

    return listp



#Astronomical unit
Au= 1.496e11
#parsec
pc=3.09e16
#Mass of the sun
M_s=1.99e30
#big G
G = 6.67e-11
#useful
M=1e24
L=1e9


###These values are the intial conditions for the system and number of bodies 
###in the system. 


#masses
M_i=[M_s,1898*M,568*M,86.8*M,102*M]


 
#x-position
x=[-1.4e9,7.3e11,1.2e12,2.0e12,4.5e12]

#y-position
y=[ 4.0e7,1.3e11,-8.4e11,2.2e12,-4.5e11]

#z-position
z=[3.1e7,-1.7e10,-3.4e10,-1.8e10,-9.3e10]

#x-velocity
vx=[1.2,-2.5e3,5.0e3,-5.0e3,5.1e2]

#y-velocity
vy=[-1.6e1,1.4e4,7.9e3,4.3e3,5.4e3]

#z-velocity
vz=[1.0e-1,6.0e-1,-3.4e2,8.1e1,-1.2e2]


#put all values into a list and send to ode
intial=[x,y,z,vx,vy,vz,M_i]
intial=np.array(np.concatenate( intial , axis=None ))


#time
t=np.linspace(0,year(170),17000)

#values used in the loop
intial=np.array(intial)
N=np.size(intial)
#spacing between each co-ordinate
go=int(N/7)
#mass values
M_T=intial[int(6*N/7):int(N)]
M_T=np.array(M_T)
    

#mass dosen't change, return zero from function since mass dosen't need to be solved
oppo=np.size(M_T)
oppo=np.zeros(oppo)



#solution to the ode
#atol and rtol can be change to get lower error in energy
sol = odeint(pend, intial, t) #atol=1e-6 ,rtol=1e-6)



solar_system=['Sun','Jupiter','Saturn','Uranus','Neptune']



ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')

    
#3D plot, using a loop for each plot
for m in range(int((N)/7)):
    ax.plot(sol[:, (m)]/Au, sol[:, (m+(int((N)/7)))]/Au , 
            sol[:, (m+(int(2*N/7)))]/Au ,
            label=solar_system[m])


ax.set_xlabel('x [Au]')
ax.set_ylabel('y [Au]')
ax.set_zlabel('z [Au]')
ax.set_zlim(-30,30)
ax.legend(loc='best')
plt.tight_layout()

print("--- %s seconds ---" % (time.time() - start_time))


#%%

#energy values, sum of KE + GPE= constant


N=np.size(intial)

#get each velocity value from solution
velocity_x=sol[: , int((3*N)/7):int((4*N)/7)]
velocity_y=sol[: , int((4*N)/7):int((5*N)/7)]
velocity_z=sol[: , int((5*N)/7):int((6*N)/7)]

#number of rows from solution of odes
numb=np.size(t)    
go=int(N/7)




veles=[]

#find the magnitude of velocity for each mass
for a in range(numb):
    #all velocity value in each coordinate
    v= (velocity_x[a,:])**2 + (velocity_y[a,:])**2 + (velocity_z[a,:])**2
    veles.append(v)


veles=np.array(veles)



#find KE for each mass at each time interval
ke_array=[]
for q in range(int(go)):
    ke_1= 0.5*M_i[q]*veles[:,q]
    ke_array.append(ke_1)
    
 
ke_array=np.array(ke_array)    
 

ke_T= np.sum(ke_array,axis=0)


M_i=np.array(M_i)



   

x=sol[:,0:int((N)/7)]
y=sol[:,int((N)/7):int((2*N)/7)]
z=sol[:,int((2*N)/7):int((3*N)/7)]


numb=np.size(t)    

#empty array for positions
x1=[]
y1=[]
z1=[]

#finding the distance between every mass
for a in range(numb):
    op=[]
    io=[]
    zo=[]
    for k in range(go):
        m=[]
        fl=[]
        oup=[]
        for i in range(go): 
            xl=(x[a,k]-x[a,i])
            yl=(y[a,k]-y[a,i])
            zl=(z[a,k]-z[a,i])
            m.append(xl)
            fl.append(yl)
            oup.append(zl)
        m=np.array(np.concatenate( m , axis=None ))
        fl=np.array(np.concatenate( fl , axis=None ))
        oup=np.array(np.concatenate( oup , axis=None ))
        op.append(m)
        io.append(fl)
        zo.append(oup)
    op=np.array(np.concatenate( op , axis=None ))
    op=np.reshape(op,(go,go)) 
    io=np.array(np.concatenate( io , axis=None ))
    io=np.reshape(io,(go,go))
    zo=np.array(np.concatenate( zo , axis=None ))
    zo=np.reshape(zo,(go,go)) 
    x1.append(op)
    y1.append(io)
    z1.append(zo)
 
#distance on each axes into an array   
x1=np.array(x1)
y1=np.array(y1)
z1=np.array(z1)
#distance between each mass
r=np.sqrt(x1**2 +y1**2+z1**2)


gpe_a=[]
for op in range(numb):
    qwem=r[op]
    gpe=[]
    for q in range(go):
        for f in range(go):
            G = 6.67e-11
            k=qwem[q,f]
            if k!=0.00:
                gpe_1= -(G*M_i[q]*M_i[f])/(k)
                gpe.append(gpe_1)
    gpe=np.sum(gpe)
    gpe_a.append(gpe)

#since the GPE found in the loop has double the value 
#gpe needs to be halfed
gpe_a=np.array(gpe_a)*(0.5)



Energy = gpe_a +ke_T

#find de and plot de against against time
N=np.size(Energy)

E_0 = Energy[0] * np.ones((1, N))


change_E= (Energy- E_0)/(E_0)

#Transpose array
change_E=change_E.T


plt.figure(figsize=(6,6))
plt.plot(t, change_E, 'g')
#plt.title('Relative change in energy against time')
plt.xlabel('time [s]')
plt.ylabel('Relative $\Delta$ E ')
plt.tight_layout()


