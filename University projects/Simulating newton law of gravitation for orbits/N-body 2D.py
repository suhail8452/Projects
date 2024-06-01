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
    accelerations=[]


    


    #co-ordinates for x,y,z in 3 rows *(number of masses) columns
    coords=  np.array([intial[0:int(2*N/5)]]) 
    coords= coords.reshape(2,go)                   

   
    
    for a in range(go):
        #finding the distance between each mass code
        distant=np.sqrt((coords[0,a]-coords[0,:])**2 + 
                        (coords[1,a]-coords[1,:])**2 )
        
        
   
   
        #acceleration in each plane for each mass

        
        acc= G*M_T[:]*(coords[0,:]-coords[0,a],coords[1,:]-coords[1,a])/np.abs((distant)**3 +1e-9)
        
        accelerations.append(acc)
        

    
    #sum of acceleration for each body in each plane
    accelerations =np.concatenate( accelerations , axis=0 )

    c=(np.sum( accelerations, axis=1)).reshape((go,2)).T

    accelerations =np.concatenate( c , axis=0 )
 


    #velocity for return function
    velocity= intial[2*go:4*go]

    
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

M=1e24
L=1e9

###These values are the intial conditions for the system and number of bodies 
###in the system that can be changed

#masses
M_i=[M_s,0.330*M,4.87*M,5.97*M]

#x-position
x=[0,57.9*L,108.2*L,149.6*L]

#y-position
y=[0,0,0,0]

#x-velocity
vx=[0,0,0,0]

#y-velocity
vy=[0,47.4e3,35e3,29.8e3]


#place all values in an array
intial=[x,y,vx,vy,M_i]
intial=np.array(np.concatenate( intial , axis=None ))



#time
t=np.linspace(0,year(2),100)



##These values are used in the function pend()
intial=np.array(intial)
N=np.size(intial)
#mass values
M_T=intial[int(4*N/5):int(N)]
M_T=np.array(M_T)
#number of columns for each co-ordinate/mass
go=int(N/5)
G = 6.67e-11

#mass dosen't change, return zero from function as mass dosen't need to be solved
oppo=np.size(M_T)
oppo=np.zeros(oppo)




#solution 
#tolerance changed to get more precise result
sol = odeint(pend, intial, t,atol=1e-12 ,rtol=1e-12)



#plot subplot 

fig, ax=plt.subplots(2,1,figsize=(10,8),gridspec_kw={'height_ratios': [7, 3]})



for m in range(go):
    ax[0].plot(sol[:, (m)]/Au, sol[:, (m+(int((N)/5)))]/Au ,label='body'+str(m+1))



ax[0].set_xlabel('x [Au]',fontsize=14)
ax[0].set_ylabel('y [Au]',fontsize=14)
#ax[0].axis('equal')
ax[0].legend(loc='best')

#bottom figure, x-axis against time
for m in range(go):
    ax[1].plot(t/year(1),sol[:, m]/Au)

ax[1].set_xlabel('Time [yr]',fontsize=14)
ax[1].set_ylabel('x [Au]',fontsize=14)

plt.tight_layout()



#energy values, sum of KE + GPE= constant


velocity_x=sol[:,int((2*N)/5):int((3*N)/5)]
velocity_y=sol[:,int((3*N)/5):int((4*N)/5)]

numb=np.size(t)    

veles=[]

#velocity squared
for a in range(numb):
    v=(velocity_x[a,:])**2 + (velocity_y[a,:])**2
    veles.append(v)
    
veles=np.array( veles)




#kinetic energy of each mass
ke_array=[]
for q in range(go):
    ke_1= 0.5*M_i[q]*veles[:,q]
    ke_array.append(ke_1)
    
ke_array=np.array(ke_array )

#KE of all mass at each time t
ke_T= np.sum(ke_array,axis=0)


M_i=np.array(M_i)


#x and y coordinates
x=sol[:,0:int((N)/5)]
y=sol[:,int((N)/5):int((2*N)/5)]
#number of rows for x and y
numb=np.size(t)    

x1=[]
y1=[]

#finding the distance between every mass
for a in range(numb):
    op=[]
    io=[]
    for k in range(go):
        m=[]
        fl=[]
        for i in range(go): 
            #difference between all the x values
            xl=(x[a,k]-x[a,i])
            #difference between all the y values
            yl=(y[a,k]-y[a,i])
            m.append(xl)
            fl.append(yl)
        m=np.array(np.concatenate( m , axis=None ))
        fl=np.array(np.concatenate( fl , axis=None ))
        op.append(m)
        io.append(fl)
    op=np.array(np.concatenate( op , axis=None ))
    op=np.reshape(op,(go,go)) 
    io=np.array(np.concatenate( io , axis=None ))
    io=np.reshape(io,(go,go)) 
    x1.append(op)
    y1.append(io)
 



#distance between each mass in each co-ordinate
x1=np.array(x1)
y1=np.array(y1)

#distance between each mass
r=np.sqrt(x1**2 +y1**2)




#gpe for each mass for each combination for distance
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

#gpe is doubled so needs to be halved
gpe_a=np.array(gpe_a)*(0.5)



Energy = gpe_a +ke_T

#find de and plot de against against time

N=np.size(Energy)

E_0 = Energy[0] * np.ones((1, N))

change_E= (Energy- E_0)/(E_0)

#Transpose array
change_E=change_E.T

change_Emu= change_E

#plot figure
plt.figure(figsize=(6,6))
plt.plot(t/year(1), change_Emu, 'g')
#plt.title('Relative change in energy against time')
plt.xlabel('time [yr]')
plt.ylabel('Relative $\Delta$ E')

plt.tight_layout()

print("--- %s seconds ---" % (time.time() - start_time))