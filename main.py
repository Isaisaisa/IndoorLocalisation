import Reader as reader
import matplotlib.pyplot as plt
import numpy as np



#data = reader.readCSVData('2019-05-2711.00.19.csv')
#print(data)

#np.save('dataLoaded.npy', data)
data = np.load('dataLoaded.npy')



# time,gFx.gFy,gFz.wx,wy.wz,
fig, axs = plt.subplots(3, 2)
axs[0,0].plot(data[:,0], data[:,1])
axs[0,0].set_xlabel('time')
axs[0,0].set_ylabel('gFx')
axs[0,0].grid(True)
axs[1,0].plot(data[:,0], data[:,2])
axs[1,0].set_xlabel('time')
axs[1,0].set_ylabel('gFy')
axs[1,0].grid(True)
axs[2,0].plot(data[:,0], data[:,3])
axs[2,0].set_xlabel('time')
axs[2,0].set_ylabel('gFz')
axs[2,0].grid(True)

axs[0,1].plot(data[:,0], data[:,4])
axs[0,1].set_xlabel('time')
axs[0,1].set_ylabel('wx')
axs[0,1].grid(True)
axs[1,1].plot(data[:,0], data[:,5])
axs[1,1].set_xlabel('time')
axs[1,1].set_ylabel('wy')
axs[1,1].grid(True)
axs[2,1].plot(data[:,0], data[:,6])
axs[2,1].set_xlabel('time')
axs[2,1].set_ylabel('wz')
axs[2,1].grid(True)

fig.tight_layout()
plt.show()


import numpy as np
import peakutils

cb = np.array([-0.010223, ... ])
indexes = peakutils.indexes(data[:,1], min_dist=100)
print('d')