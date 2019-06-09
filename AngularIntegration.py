import numpy as np
from scipy import signal
import math
import matplotlib.pyplot as plt

def calculateTurningAngle(dataWithAccNorm):

    data = dataWithAccNorm

    # ----- estimate g_B' at t = 0 with an approximation (from the first 4 seconds of the recorded data)
    # --> page 16
    g_B_apostrophe_start = np.zeros(shape=(3, 1))
    approximationTime = 4 # 4 seconds
    frequency = 400
    length = approximationTime * frequency


    g_B_apostrophe_start[0] = np.mean(data[length, 1])
    g_B_apostrophe_start[1] = np.mean(data[length, 2])
    g_B_apostrophe_start[2] = np.mean(data[length, 3])
    print('g_B_apostrophe at t = 0 \n', g_B_apostrophe_start)

    # ----- Calculate g_B' for the whole time sequence
    # --> page 17
    mu = 0.9
    g_B_apostrophe = np.zeros(shape=(3, data[:,0].shape[0]))
    g_B_apostrophe[:,0] = g_B_apostrophe_start[:,0]

    # g_B_apostrophe --> (3x72677)
    for x in range(1,data[:,0].shape[0]):
        g_B_apostrophe[:,x] = (mu * g_B_apostrophe[:,x-1]) + ((1-mu) * data[x,1:4])


    # ----- calculate matrix R and the inverse of R for the time sequence
    # --> page 14
    arr = np.array([0, 1, 0])
    matrixR = np.zeros(shape=(data[:,0].shape[0], 3, 3))
    matrixR_Inverse = np.zeros(shape=(data[:,0].shape[0], 3, 3))
    for x in range(0, data[:, 0].shape[0]):


        u_z = g_B_apostrophe[:, x]
        u_x = np.cross(arr, u_z)
        u_y = np.cross(u_z, u_x)

        u_z = u_z / np.linalg.norm(u_z)
        u_x = u_x / np.linalg.norm(u_x)
        u_y = u_y / np.linalg.norm(u_y)

        # transpose the vectors from (3,1) to (1,3) and store it as matrix
        matrix = np.matrix(np.array([u_x.reshape((-1,1)), u_y.reshape((-1,1)), u_z.reshape((-1,1))]))
        matrixR[x, :, :] = matrix
        #matrixR_Inverse[x, :, :] = np.linalg.inv(matrix)
        matrixR_Inverse[x, :, :] = np.transpose(matrix)

    # ----- Calculate projected turn rate
    # --> page 18
    turn_Rate = np.zeros(shape = (data[:,0].shape[0], 3))
    for x in range(0,data[:,0].shape[0]):
        turn_Rate[x,:] = np.dot(matrixR_Inverse[x,:,:] ,data[x,4:7].reshape((-1,1)))[:,0]


    # ---- Calculate estimated turning angle at time t
    # --> page 18
    turning_Angle = np.zeros(shape = (data[:,0].shape[0], 3))
    turning_Angle[0, :] = turn_Rate[0,:] * data[0,0]
    for x in range(1,data[:,0].shape[0]):
        delta_t = (data[x,0] - data[x-1,0])
        if delta_t == 0:
            delta_t = 0.0001
        turning_Angle[x,:] = turning_Angle[x-1,:] + turn_Rate[x,:] * delta_t

    return turning_Angle

def plotTrack(turning_Angle, steps):
    # ----- Plotting the estimated tracks
    ##x_y_t = np.zeros(shape= (turning_Angle.shape[0], 2))
    x_y_t = np.zeros(shape=(steps.shape[0], 2))
    x_y_t[0,0] = 0
    x_y_t[0,1] = 0
    for i in range(1, steps.shape[0]):
        x_y_t[i, 0] = x_y_t[i - 1, 0] + 0.8 * math.cos(turning_Angle[steps[i],2])
        x_y_t[i, 1] = x_y_t[i - 1, 1] + 0.8 * math.sin(turning_Angle[steps[i], 2])
    plt.figure()
    plt.plot(x_y_t[:,0],x_y_t[:,1], '-o')
    plt.show()