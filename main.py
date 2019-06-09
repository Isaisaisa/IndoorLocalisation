import os

import math
import matplotlib.pyplot as plt
import numpy as np
import peakutils
from scipy import signal

import Reader as reader
import config_dev as cfg
import AngularIntegration as AI
import Plotter

DATAFILE = 'record2.csv'
VISUALIZE_RECORDED_DATA = True
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
if VISUALIZE_RECORDED_DATA:
    Plotter.plotRecordedData(data)

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
Plotter.plotOneDataSet(dataX=data[:,0], dataY=data[:,7], xLabel='time',yLabel='Acc. Norm', title='calculated acceleration norm')


# ----- First try to get the peaks of the Acc. norm and plot it
# akueller Threshold gibt nut bei '2019-05-2711.00.19.csv' gute Ergebnisse !!
indexes = peakutils.indexes(data[:, 7], thres=0.35 * max(data[:, 7]), min_dist=50)
print('Anzahl peaks:', indexes.shape[0])
# visualize the peaks
Plotter.plotOneDataSetWithPeaks(dataX=data[:,0], dataY=data[:,7], peakIndexes=indexes, xLabel='time', yLabel='Acc. Norm', title='Peaks with a simple threshold')



if LOWPASSFILTER:
    # ---- First try of a low pass filter
    fc = 3  # Cut-off frequency of the filter
    fs = 400  # Sampling frequency
    w = fc / (fs / 2) # Normalize the frequency
    b, a = signal.butter(5, w, 'low')
    output = signal.filtfilt(b, a, data[:,7])

    # ---- visualize the results after filtering the signal
    indexes2 = peakutils.indexes(output, thres=0.4 * max(output), min_dist=50)
    print('Anzahl peaks (nach Filter):', indexes2.shape[0])
    Plotter.plotOneDataSetWithPeaks(dataX=data[:,0], dataY=output, peakIndexes=indexes2, xLabel='time', yLabel='Acc. Norm (filtered)', title='signal and peaks (threshold) after low pass filter')

    # ---- extract peaks by using scipy.signal
    avg = np.mean(output)
    print('Average height: ', avg)
    print('Min height of peak: ', 1.1 * avg)
    [idx, prop] = signal.find_peaks(output, height=1.1*avg)
    print('Anzahl Peaks mit scipy + Filter: ', len(idx))
    Plotter.plotOneDataSetWithPeaks(dataX=data[:,0], dataY=output, peakIndexes=idx, xLabel='time', yLabel='Acc. Norm', title='signal and peaks (scipy) after low pass filter')


# ----- Calculate Turning Angle
turning_Angle = AI.calculateTurningAngle(dataWithAccNorm=data, mu = 0.5)
# ---- Plot estimated track with different step estimation results
estimatedTrack = AI.estimateTrack(turning_Angle, indexes ,stepLength=0.7)
Plotter.plotEstimatedTrack(dataX= estimatedTrack[:, 0], dataY= estimatedTrack[:,1], title='Estimated track, peaks with threshold')
estimatedTrack2 = AI.estimateTrack(turning_Angle, indexes2, stepLength=0.7)
Plotter.plotEstimatedTrack(dataX= estimatedTrack2[:, 0], dataY= estimatedTrack2[:,1], title='Estimated track, peaks with threshold after low pass filter')
estimatedTrack3 = AI.estimateTrack(turning_Angle, idx, stepLength=0.7)
Plotter.plotEstimatedTrack(dataX= estimatedTrack3[:, 0], dataY= estimatedTrack3[:,1], title='Estimated track, peaks with scipy after low pass filter')

plt.show()