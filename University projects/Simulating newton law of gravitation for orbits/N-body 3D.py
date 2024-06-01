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

    intial=np.array(intial)
    N=np.size(intial)
    #mass values
    M_T=intial[int(6*N/7):int(N)]
    M_T=np.array(M_T)
    
    #empty array for accelerations
    accelerations_x=[]
    accelerations_y=[]
    accelerations_z=[]
    
    go=int(N/7)
    G = 6.67e-11
    
    #mass dosen't change, return zero from function
    oppo=np.size(M_T)
    oppo=np.zeros(oppo)
    

    #co-ordinates for x,y,z in 3 rows *(number of masses) columns
    coords=  np.array([intial[0:int(N/7)], 
                       intial[int(N/7):int((2*N)/7)], 
                       intial[int(2*N/7):int(3*N/7)]])
    
   
    
    for a in range(go):
        #finding the distance between each mass code
        distant=np.sqrt((coords[0,a]-coords[0,:])**2 + 
                        (coords[1,a]-coords[1,:])**2 +
                        (coords[2,a]-coords[2,:])**2)
        
        
   
   
        #acceleration in each plane for each mass

        acc_x= G*M_T[:]*(coords[0,:]-coords[0,a])/np.abs((distant)**3+1e-30)
        acc_y= G*M_T[:]*(coords[1,:]-coords[1,a])/np.abs((distant)**3+1e-30)
        acc_z= G*M_T[:]*(coords[2,:]-coords[2,a])/np.abs((distant)**3+1e-30)


        accelerations_x.append(acc_x)
        accelerations_y.append(acc_y)
        accelerations_z.append(acc_z)
    
    #sum of acceleration for each body in each plane
    accelerations_x=np.sum(accelerations_x,axis=1)
    accelerations_y=np.sum(accelerations_y,axis=1)
    accelerations_z=np.sum(accelerations_z,axis=1)
    
   

    #velocity for return function
    vx=intial[int((3*N)/7) : int((4*N)/7)]
    vy=intial[int((4*N)/7) : int((5*N)/7)]
    vz=intial[int((5*N)/7) : int((6*N)/7)]

    
    #return list of values
    listp=[vx,vy,vz,accelerations_x,accelerations_y,accelerations_z,oppo]
    listp=np.array(np.concatenate( listp , axis=None ))

    return listp



#Astronomical unit
Au= 1.496e11
#parsec
pc=3.09e16
#Mass of the sun
M=1.99e30

###These values are the intial conditions for the system and number of bodies 
###in the system
###these values have to be changed

#masses
M_i=[M,M/1000,M/1000,M/1000]

#x-position
x=[0,3*Au,-Au,0]

#y-position
y=[0,0,0,Au]

#z-position
z=[0,0,0,0]

#x-velocity
vx=[10,0,0,0]

#y-velocity
vy=[0,0,30e3,0]

#z-velocity
vz=[0,17e3,0,30e3]


intial=[x,y,z,vx,vy,vz,M_i]
intial=np.array(np.concatenate( intial , axis=None ))



#time
t=np.linspace(0,year(10),1000)






#solution to the ode
#change odeint to decrease uncertainty
sol = odeint(pend, intial, t, atol=1e-10 ,rtol=1e-10)

N=np.size(intial)



#plot subplot 

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1, 1, 1, projection='3d')

#Top figure, 3D plot
for m in range(int((N)/7)):
    ax.plot(sol[:, (m)]/Au, sol[:, (m+(int((N)/7)))]/Au , 
            sol[:, (m+(int(2*N/7)))]/Au ,
            label='Body'+str(m+1))


#labels for top figure
ax.set_xlabel('x [Au]')
ax.set_ylabel('y [Au]')
ax.set_zlabel('z [Au]')
ax.legend(loc='best')


plt.tight_layout()






#energy values, sum of KE + GPE= constant


N=np.size(intial)

#get each velocity value from solution
velocity_x=sol[: , int((3*N)/7):int((4*N)/7)]
velocity_y=sol[: , int((4*N)/7):int((5*N)/7)]
velocity_z=sol[: , int((5*N)/7):int((6*N)/7)]


numb=np.size(t)    
go=int(N/7)




veles=[]
cols = len(velocity_x[0])
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
 
    
x1=np.array(x1)
y1=np.array(y1)
z1=np.array(z1)
r=np.sqrt(x1**2 +y1**2+z1**2)


#distance between each mass


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

       
gpe_a=np.array(gpe_a)*(0.5)


Energy = gpe_a +ke_T


#find de and plot de against against time

N=np.size(Energy)

E_0 = Energy[0] * np.ones((1, N))


change_E= (Energy- E_0)/(E_0)

#Transpose array
change_E=change_E.T
#Convert to microJoules
change_Emu= change_E

plt.figure(figsize=(6,6))
plt.plot(t/year(1), change_Emu, 'g')
#plt.title('Relative change in energy against time')
plt.xlabel('time [yrs]')
plt.ylabel('Relative $\Delta$ E ')


print("--- %s seconds ---" % (time.time() - start_time))

