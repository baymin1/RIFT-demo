"""
LSM函数是为了使用最小二乘法来计算两组点之间的仿射变换参数。
match1和match2：两组对应的点的坐标，每组点是一个Nx2的矩阵。
change_form：指定变换的类型，这里是（'affine'）'仿射'变换。

构建系数矩阵A:
使用match1的坐标构建一个系数矩阵A，用于后续的线性方程求解。
构建观测矩阵b:
使用match2的坐标构建观测矩阵b。

求解线性方程:
使用QR分解来求解线性方程Ax = b，其中x是所求的变换参数。
如果change_form为'仿射'（'affine'），则这个方程将为6个未知数（仿射变换的参数）的方程。

计算RMSE:
使用计算出的仿射变换参数变换match1的坐标，并与match2的坐标进行比较，从而计算均方根误差（RMSE）。
输出:
parameters：计算出的仿射变换参数。
rmse：均方根误差，表示变换精度。
"""

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
