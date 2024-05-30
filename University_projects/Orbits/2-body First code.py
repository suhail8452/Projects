# -*- coding: utf-8 -*-


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#defintions

def year(n):
    return n*365*24*60*60


#function for odeint    
def pend(intial,t,M1,M2):
    
    #intial values
    x1=intial[0]
    x2=intial[1]
    y1=intial[2]
    y2=intial[3]
    vx1=intial[4]
    vx2=intial[5]
    vy1=intial[6]
    vy2=intial[7]
    
    G = 6.67e-11
    
    #distance btween two centers of mass
    distance=np.sqrt((x1-x2)**2 + (y2-y1)**2)
    
    #velocity
    vel_x1=vx1
    vel_x2=vx2
    vel_y1=vy1
    vel_y2=vy2
    
    #acceleration using Newton LAw of gravitation
    acc_x1= G*M2*(x2-x1)/np.abs((distance)**3)
    acc_x2= G*M1*(x1-x2)/np.abs((distance)**3)
    acc_y1= G*M2*(y2-y1)/np.abs((distance)**3)
    acc_y2= G*M1*(y1-y2)/np.abs((distance)**3)
    

    new=[vel_x1,vel_x2,vel_y1,vel_y2,acc_x1,acc_x2,acc_y1,acc_y2]
    
    return new


#Astronomical unit
Au= 1.496e11
#Mass of the sun
M=1.99e30
#mass of two bodies
M1=1.99e30
M2=1.99e30 

#intial values of position and velocity
x1=-0.5*Au
x2=0.5*Au
y1=0
y2=0
vx1=0
vx2=0
vy1=-15e3
vy2=15e3


intial=[x1,x2,y1,y2,vx1,vx2,vy1,vy2]

#time
t=np.linspace(0,year(5),1000)

#solution to the eight coupled odes
sol = odeint(pend, intial, t, args=(M1,M2))




#plot subplot 

fig, ax=plt.subplots(2,1,figsize=(10,8),gridspec_kw={'height_ratios': [7, 3]})

#fig.suptitle('Two body System',fontsize=18)

#Top figure, y against x for the two bodies
ax[0].plot(sol[:, 0]/Au, sol[:, 2]/Au, 'r',label='Body 1')

ax[0].plot(sol[:, 1]/Au, sol[:, 3]/Au, 'b', label='Body 2')
ax[0].set_xlabel('x [Au]',fontsize=14)
ax[0].set_ylabel('y [Au]',fontsize=14)
ax[0].axis('equal')
ax[0].legend(loc='best')

#bottom figure, x-axis against time
ax[1].plot(t/year(1),sol[:, 0]/Au, 'r')
ax[1].plot(t/year(1),sol[:, 1]/Au, 'b')
ax[1].set_xlabel('Time [yr]',fontsize=14)
ax[1].set_ylabel('x [Au]',fontsize=14)


#energy values, sum of KE + GPE= constant


def GPE(M,m,r):
    G = 6.67e-11
    gpe= -(G*M*m)/(r)
    return gpe

x1=sol[:,0]
x2=sol[:,1]
y1=sol[:,2]
y2=sol[:,3]
vx1=sol[:,4]
vx2=sol[:,5]
vy1=sol[:,6]
vy2=sol[:,7]


velocity_1_2= vx1**2 + vy1**2

velocity_2_2= vx2**2 + vy2**2

ke_1= 0.5*M1*velocity_1_2 
ke_2= 0.5*M2*velocity_2_2

ke=ke_1+ke_2

r=np.sqrt((x1-x2)**2 + (y1-y2)**2)

y1

gpe= GPE(M1,M2,r)

Energy = gpe +ke

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
plt.xlabel('time [yr]')
plt.ylabel('Relative $\Delta$ E ')


    
        


