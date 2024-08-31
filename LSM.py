
# LSM函数是为了使用最小二乘法来计算两组点之间的仿射变换参数。


import numpy as np
import math


def lsm(match1, match2, change_form):
    A = np.zeros([2 * len(match1), 4])
    for i in range(len(match1)):
        A[2 * i:2 * i + 2] = np.tile(match1[i], (2, 2))
    B = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    B = np.tile(B, (len(match1), 1))
    A = A * B
    B = np.array([[1, 0], [0, 1]])
    B = np.tile(B, (len(match1), 1))
    A = np.hstack((A, B))
    b = match2.reshape(1, int(len(match2) * len(match2[0]))).T

    if change_form == "affine":
        Q, R = np.linalg.qr(A)
        parameters = np.zeros([8, 1])
        parameters[:6] = np.linalg.solve(R, np.dot(Q.T, b))
        N = len(match1)
        M = np.array([[parameters[0][0], parameters[1][0]], [parameters[2][0], parameters[3][0]]])
        match1_test_trans = M.dot(match1.T) + np.tile([parameters[4], parameters[5]], (1, N))
        match1_test_trans = match1_test_trans.T
        test = match1_test_trans - match2
        rmse = math.sqrt(sum(sum(np.power(test, 2))) / N)
    return np.squeeze(parameters), rmse
