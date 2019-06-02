import Reader as reader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import config_dev as cfg
import math
import peakutils

DATAFILE = '2019-05-2711.00.19.csv'

# ----- Load the data to a numpy array and save the array
#data = reader.readCSVData(DATAFILE)
#np.save(cfg.SAVEPATH + 'dataLoaded.npy', data)
data = np.load(cfg.SAVEPATH + 'dataLoaded.npy')

# ----- Visualize the data
# time,gFx.gFy,gFz.wx,wy.wz,
fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(data[:, 0], data[:, 1])
axs[0, 0].set_xlabel('time')
axs[0, 0].set_ylabel('gFx')
axs[0, 0].grid(True)
axs[1, 0].plot(data[:, 0], data[:, 2])
axs[1, 0].set_xlabel('time')
axs[1, 0].set_ylabel('gFy')
axs[1, 0].grid(True)
axs[2, 0].plot(data[:, 0], data[:, 3])
axs[2, 0].set_xlabel('time')
axs[2, 0].set_ylabel('gFz')
axs[2, 0].grid(True)

axs[0, 1].plot(data[:, 0], data[:, 4])
axs[0, 1].set_xlabel('time')
axs[0, 1].set_ylabel('wx')
axs[0, 1].grid(True)
axs[1, 1].plot(data[:, 0], data[:, 5])
axs[1, 1].set_xlabel('time')
axs[1, 1].set_ylabel('wy')
axs[1, 1].grid(True)
axs[2, 1].plot(data[:, 0], data[:, 6])
axs[2, 1].set_xlabel('time')
axs[2, 1].set_ylabel('wz')
axs[2, 1].grid(True)

# fig.tight_layout()
#plt.show()

# ----- Calculate the norm of the acceleration
dataLength = data.shape
dataAppend = np.zeros(shape = (dataLength[0],1))
for x in range(0, dataLength[0]):
    norm = math.sqrt(data[x, 1]**2 + data[x, 2]**2 + data[x, 3]**2)
    dataAppend[x] = norm

data = np.append(data, dataAppend, axis = 1)
# Save new Data
np.save(cfg.SAVEPATH + 'dataWithAccNorm.npy', data)
data = np.load(cfg.SAVEPATH + 'dataWithAccNorm.npy')

# Plot new Data

plt.figure()
plt.plot(data[:, 0], data[:, 7])
plt.xlabel('time')
plt.ylabel('Acc. Norm')
plt.grid(True)
##plt.show()

# ----- First try to get the peaks of the Acc. norm


indexes = peakutils.indexes(data[:, 7], thres=0.4 * max(data[:, 7]), min_dist=50)
print('Anzahl peaks:', indexes.shape[0])
plt.plot(data[indexes,0], data[indexes, 7], 'ro')
plt.show()