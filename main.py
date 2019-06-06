import os

import math
import matplotlib.pyplot as plt
import numpy as np
import peakutils
from scipy import signal

import Reader as reader
import config_dev as cfg

DATAFILE = 'record1.csv'
LOWPASSFILTER = True
filename = os.path.splitext(os.path.basename(DATAFILE))[0]

# ----- Load the data to a numpy array and save the array
filepath = cfg.SAVEPATH + filename + '_dataLoaded.npy'
if not os.path.exists(filepath):
    if not os.path.exists(cfg.SAVEPATH):
        os.makedirs(cfg.SAVEPATH, exist_ok=True)
    print('numpy array does not exist yet')
    data = reader.readCSVData(DATAFILE)
    np.save(filepath, data)

# Load data from array
data = np.load(filepath)

# ----- Visualize the data
# time,gFx.gFy,gFz.wx,wy.wz,
fig, axs = plt.subplots(3, 2)
# G-force data
# Plot data from first and second column
axs[0, 0].plot(data[:, 0], data[:, 1])
axs[0, 0].set_xlabel('time')
axs[0, 0].set_ylabel('gFx')
axs[0, 0].grid(True)
# Plot data from first and third column
axs[1, 0].plot(data[:, 0], data[:, 2])
axs[1, 0].set_xlabel('time')
axs[1, 0].set_ylabel('gFy')
axs[1, 0].grid(True)
# Plot data from first and fours column
axs[2, 0].plot(data[:, 0], data[:, 3])
axs[2, 0].set_xlabel('time')
axs[2, 0].set_ylabel('gFz')
axs[2, 0].grid(True)

# Gyroscope data
# Plot data from first and fifth column
axs[0, 1].plot(data[:, 0], data[:, 4])
axs[0, 1].set_xlabel('time')
axs[0, 1].set_ylabel('wx')
axs[0, 1].grid(True)
# Plot data from first and sixth column
axs[1, 1].plot(data[:, 0], data[:, 5])
axs[1, 1].set_xlabel('time')
axs[1, 1].set_ylabel('wy')
axs[1, 1].grid(True)
# Plot data from first and seventh column
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
np.save(cfg.SAVEPATH + filename + '_dataWithAccNorm.npy', data)
data = np.load(cfg.SAVEPATH + filename + '_dataWithAccNorm.npy')

# Plot new Data
plt.figure()
plt.plot(data[:, 0], data[:, 7])
plt.xlabel('time')
plt.ylabel('Acc. Norm')
plt.grid(True)
##plt.show()


# ----- First try to get the peaks of the Acc. norm and plot it
# akueller Threshold gibt nut bei '2019-05-2711.00.19.csv' gute Ergebnisse !!
indexes = peakutils.indexes(data[:, 7], thres=0.35 * max(data[:, 7]), min_dist=50)
print('Anzahl peaks:', indexes.shape[0])
plt.plot(data[indexes,0], data[indexes, 7], 'ro')
plt.show()


if LOWPASSFILTER:
    # ---- First try of a low pass filter
    fc = 3  # Cut-off frequency of the filter
    fs = 400  # Sampling frequency
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, data[:,7])
    plt.figure()
    plt.plot(data[:,0], output, label='filtered')
    plt.legend()
    plt.grid(True)

    # ---- visualize the results after filtering the signal
    indexes2 = peakutils.indexes(output, thres=0.4 * max(output), min_dist=50)
    print('Anzahl peaks (nach Filter):', indexes2.shape[0])
    plt.plot(data[indexes2,0], output[indexes2], 'ro')
    #plt.show()

    # ---- extract peaks by using scipy.signal
    avg = np.mean(output)
    print('Average height: ', avg)
    print('Min height of peak: ', 1.1 * avg)
    [idx, prop] = signal.find_peaks(output, height=1.1*avg)
    print('Anzahl Peaks mit scipy + Filter: ', len(idx))
    plt.figure()
    plt.xlabel('time')
    plt.ylabel('Acc.')
    plt.plot(data[:, 0], output, label='filtered')
    plt.plot(data[idx, 0], output[idx], 'ro')
    plt.show()