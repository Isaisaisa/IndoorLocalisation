import matplotlib.pyplot as plt
# This file provides often used functions to plot data

# ---- This function plots the initial recorded data
# arguments:
#   data:    data to be plotted, format: time,gFx.gFy,gFz.wx,wy.wz,
def plotRecordedData(data):
    # time,gFx.gFy,gFz.wx,wy.wz,
    fig, axs = plt.subplots(3, 2)
    fig.suptitle('recorded data')
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

# ---- This function plots one dataset
# arguments:
#   dataX and dataY: data to be plotted, x (time) and y (values) axis
#   x/yLabel:        labels for the x and y axis
#   title:           title for the figure
#   draw:            whether the plot is to be displayed
def plotOneDataSet(dataX, dataY, xLabel='', yLabel='', title='', draw = True):
    plt.figure()
    plt.plot(dataX, dataY)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(title)
    plt.grid(True)
    if draw:
        plt.draw()

# ---- This function plots one dataset with peaks
# arguments:
#   dataX and dataY: data to be plotted, x (time) and y (values) axis
#   peakIndexes:     indexes of the peaks
#   x/yLabel:        labels for the x and y axis
#   title:           title for the figure
def plotOneDataSetWithPeaks(dataX, dataY, peakIndexes, xLabel, yLabel, title):
        plotOneDataSet(dataX = dataX, dataY = dataY, xLabel = xLabel, yLabel = yLabel, title = title, draw = False)
        plt.plot(dataX[peakIndexes], dataY[peakIndexes], 'ro')

# ---- This function plots the estimated track
# arguments:
#   dataX and dataY: estimated track with x and y space dimensions
#   title:           title for the figure
def plotEstimatedTrack(dataX, dataY, title):
    plt.figure()
    plt.plot(dataX, dataY, '-o')
    plt.title(title)
    plt.draw()