# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 12:23:47 2022

@author: suhail
"""

# -*- coding: utf-8 -*-

# set up random 3-d positions
#
import numpy as np
import time
import errno
N = 20000
seed = 1234
np.random.seed(seed)
pos = np.random.random((3,N))
start_time = time.time()
# deliberately slow code to find nearest neighbours within periodic unit cube
#                                                                            
#  You may only change the code between here and the line "end_time=time.time()")   
#                         



matchedIndices = np.zeros(N)
allvalues=pos[:, :]

for a in range(N):    
    matchedIndices[a] = 0 
    L=[]
    take=pos[:,a]
    take=take.reshape((3,1))

    arr=np.abs(allvalues-take)
    fx=(1-arr)
    gm=np.minimum(arr,fx)
    gx=np.sum(gm**2, axis=0)
    gx=gx.reshape((1,N))


    m=np.amin(np.array(gx)[gx != np.amin(gx)])
    q=np.where(gx == m)
    index=q[1]
    
    matchedIndices[a] = index
    


end_time = time.time()
print('Elapsed time = ', repr(end_time - start_time))

# generate filename from N and seed
filename = 'pyneigh' + str(N) + '_' + str(seed)
# if a file with this name already exists read in the nearest neighbour
# list found before and compare this to the current list of neighbours,
# else save the neighbour list to disk for future reference
try:
    fid = open(filename,'rb')
    matchedIndicesOld = np.loadtxt(fid)
    fid.close()
    if (matchedIndicesOld == matchedIndices).all(): 
        print('Checked match')
    else:
        print('Failed match')
except OSError as e:
    if e.errno == errno.ENOENT:
        print('Saving neighbour list to disk')
        fid = open(filename,'wb')
        np.savetxt(fid, matchedIndices, fmt="%8i")
        fid.close()
    else:
        raise
#        
