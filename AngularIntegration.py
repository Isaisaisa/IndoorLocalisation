import numpy as np
import math

# ---- This function calculates the turning angle from the recorded data (page 13 f. of the exercise slides)
# arguments:
#   dataWithAccNorm: recorded data (format: time,gFx.gFy,gFz.wx,wy.wz) with an appended column for the acceleration norm
#   mu:              mu for the calculation of g_B_apostroph (page 17 of the exercise)
def calculateTurningAngle(dataWithAccNorm, mu = 0.5):

    data = dataWithAccNorm

    # ----- estimate g_B' at t = 0 with an approximation (from the first 4 seconds of the recorded data)
    # --> page 16
    g_B_apostrophe_start = np.zeros(shape=(3, 1))
    approximationTime = 4 # 4 seconds
    frequency = 400       # sampling frequency
    length = approximationTime * frequency

    g_B_apostrophe_start[0] = np.mean(data[length, 1])
    g_B_apostrophe_start[1] = np.mean(data[length, 2])
    g_B_apostrophe_start[2] = np.mean(data[length, 3])

    # print results
    print('g_B_apostrophe at t = 0 \n', g_B_apostrophe_start)

    # ----- Calculate g_B' for the whole time sequence
    # --> page 17
    g_B_apostrophe = np.zeros(shape=(3, data[:, 0].shape[0]))
    g_B_apostrophe[:, 0] = g_B_apostrophe_start[:, 0]

    # g_B_apostrophe --> (3x72677)
    for x in range(1, data[:, 0].shape[0]):
        g_B_apostrophe[:, x] = (mu * g_B_apostrophe[:, x-1]) + ((1-mu) * data[x, 1:4])


    # ----- calculate matrix R and the inverse of R for the time sequence
    # --> page 14
    arr = np.array([0, 1, 0])
    matrixR = np.zeros(shape=(data[:, 0].shape[0], 3, 3))
    matrixR_Inverse = np.zeros(shape=(data[:, 0].shape[0], 3, 3))
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
    turn_Rate = np.zeros(shape = (data[:, 0].shape[0], 3))
    for x in range(0, data[:, 0].shape[0]):
        turn_Rate[x, :] = np.dot(matrixR_Inverse[x, :, :] ,data[x, 4:7].reshape((-1, 1)))[:, 0]

    # ---- Calculate estimated turning angle at time t
    # --> page 18
    turning_Angle = np.zeros(shape = (data[:, 0].shape[0], 3))
    turning_Angle[0, :] = turn_Rate[0, :] * data[0, 0]
    for x in range(1,data[:,0].shape[0]):
        delta_t = (data[x,0] - data[x-1,0])
        if delta_t == 0:
            delta_t = 0.0001
        turning_Angle[x,:] = turning_Angle[x-1,:] + turn_Rate[x,:] * delta_t

    return turning_Angle

# This function estimates the track from the turning angle and the estimated steps
# arguments:
#   turning_Angle: calculated turning angle
#   steps:         resut of the step estimator as index of the peaks/steps
#   stepLength:    constant step length
# output:
#   x_y_t:         estimated track in x and y space dimension
def estimateTrack(turning_Angle, steps, stepLength = 0.7):
    x_y_t = np.zeros(shape=(steps.shape[0], 2))
    x_y_t[0,0] = 0
    x_y_t[0,1] = 0
    for i in range(1, steps.shape[0]):
        x_y_t[i, 0] = x_y_t[i - 1, 0] + stepLength * math.cos(turning_Angle[steps[i],2])
        x_y_t[i, 1] = x_y_t[i - 1, 1] + stepLength * math.sin(turning_Angle[steps[i], 2])
    return x_y_t