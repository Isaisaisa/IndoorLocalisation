import numpy as np
import config_dev as cfg

# laod the required data
data = np.load(cfg.SAVEPATH + 'dataWithAccNorm.npy')

# ----- estimate g_B' at t = 0 with an approximation (from the first 4 seconds of the recorded data)
# --> page 16
g_B_apostrophe_start = np.zeros(shape=(3, 1))
approximationTime = 4 # 4 seconds
frequency = 400
length = approximationTime * frequency

g_B_apostrophe_start[0] = np.mean(data[length, 1])
g_B_apostrophe_start[1] = np.mean(data[length, 2])
g_B_apostrophe_start[2] = np.mean(data[length, 3])
print('g_B_g_B_apostrophe at t = 0 \n', g_B_apostrophe_start)

# ----- Calculate g_B' for the whole time sequence
# --> page 17
mu = 0.9
g_B_apostrophe = np.zeros(shape=(3, data[:,0].shape[0]))
g_B_apostrophe[:,0] = g_B_apostrophe_start[:,0]

# g_B_apostrophe --> (3x72677)
for x in range(1,data[:,0].shape[0]):
    g_B_apostrophe[:,x] = (mu * g_B_apostrophe[:,x-1]) + ((1-mu) + data[x,1:4])


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
    matrixR_Inverse[x, :, :] = np.linalg.inv(matrix)

# ----- Calculate projected turn rate
# --> page 18
turn_Rate = np.zeros(shape = (data[:,0].shape[0], 3))
#for x in range(0,data[:,0].shape[0]):
#    turn_Rate[x,:] =
print('a')