




"""
这个函数实现的是RANSAC（Random Sample Consensus）方法的一个变体，用于估计两组点之间的变换（可能是相似性、仿射性或透视变换）。
RANSAC是一种鲁棒方法，用于从包含大量离群值的数据中估计模型参数。

输入参数:
    cor1和cor2：两组对应的点的坐标。
    change_form：定义需要计算的变换类型（'affine'）
    error_t：RANSAC的阈值，用于判断一个点是否为内点。
初始化:
    根据所选的变换类型决定需要的最小点数（n）以及最大迭代次数（max_iteration）。
RANSAC主循环:
    在每次迭代中，随机选择n个点，并使用LSM（未在代码中定义，但我假设它是用于最小二乘法估计变换参数的函数）计算这些点的变换参数。
    使用计算出的变换参数将所有的cor1点变换到cor2的空间，然后计算每个点的误差。
    使用error_t阈值决定哪些点是内点。
    如果当前的内点数大于之前的最大内点数，则更新最大内点数和对应的内点。
后处理:
    删除重复的内点。
    使用所有内点再次计算变换参数，这次得到的参数应该更准确。
输出:
    返回计算出的变换矩阵。
    这个函数的主要目的是在存在大量离群值的情况下，鲁棒地估计两组点之间的变换。。
"""

import numpy as np
from LSM import LSM


def FSC(cor1, cor2, change_form, error_t):
    (M, N) = np.shape(cor1)
    if (change_form == 'similarity'):
        n = 2
        max_iteration = M * (M - 1) / 2
    elif (change_form == 'affine'):
        n = 3
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)
    elif (change_form == 'perspective'):
        n = 4
        max_iteration = M * (M - 1) * (M - 2) / (2 * 3)

    if (max_iteration > 10000):


        iterations = 10000
    else:
        iterations = max_iteration

    most_consensus_number = 0
    cor1_new = np.zeros([M, N])
    cor2_new = np.zeros([M, N])

    for i in range(iterations):
        while (True):
            a = np.floor(1 + (M - 1) * np.random.rand(1, n)).astype(np.int_)[0]
            cor11 = cor1[a]
            cor22 = cor2[a]
            if n == 2 and (a[0] != a[1]) and sum(cor11[0] != cor11[1]) and sum(cor22[0] != cor22[1]):
                break
            if n == 3 and (a[0] != a[1] and a[0] != a[2] and a[1] != a[2]) and sum(cor11[0] != cor11[1]) and sum(
                    cor11[0] != cor11[2]) and sum(cor11[1] != cor11[2]) and sum(cor22[0] != cor22[1]) and sum(
                cor22[0] != cor22[2]) and sum(cor22[1] != cor22[2]):
                break
            if n == 4 and (
                    a[0] != a[1] and a[0] != a[2] and a[0] != a[3] and a[1] != a[2] and a[1] != a[3] and a[2] != a[
                3]) and sum(cor11[0] != cor11[1]) and sum(cor11[0] != cor11[2]) and sum(cor11[0] != cor11[3]) and sum(
                cor11[1] != cor11[2]) and sum(cor11[1] != cor11[3]) and sum(cor11[2] != cor11[3]) and sum(
                cor22[0] != cor11[1]) and sum(cor22[0] != cor22[2]) and sum(cor22[0] != cor22[3]) and sum(
                cor22[1] != cor22[2]) and sum(cor22[1] != cor22[3]) and sum(cor22[2] != cor22[3]):
                break
        parameters, __ = LSM(cor11, cor22, change_form)
        solution = np.array([[parameters[0], parameters[1], parameters[4]],
                             [parameters[2], parameters[3], parameters[5]],
                             [parameters[6], parameters[7], 1]])
        match1_xy = np.ones([3, len(cor1)])
        match1_xy[:2] = cor1.T

        if change_form == 'affine':
            t_match1_xy = solution.dot(match1_xy)
            match2_xy = np.ones([3, len(cor1)])
            match2_xy[:2] = cor2.T
            diff_match2_xy = t_match1_xy - match2_xy
            diff_match2_xy = np.sqrt(sum(np.power(diff_match2_xy, 2)))
            index_in = np.argwhere(diff_match2_xy < error_t)
            consensus_num = len(index_in)
            index_in = np.squeeze(index_in)

        if consensus_num > most_consensus_number:
            most_consensus_number = consensus_num
            cor1_new = cor1[index_in]
            cor2_new = cor2[index_in]
    unil = cor1_new
    __, IA = np.unique(unil, return_index=True, axis=0)
    IA_new = np.sort(IA)
    cor1_new = cor1_new[IA_new]
    cor2_new = cor2_new[IA_new]
    unil = cor2_new
    __, IA = np.unique(unil, return_index=True, axis=0)
    IA_new = np.sort(IA)
    cor1_new = cor1_new[IA_new]
    cor2_new = cor2_new[IA_new]

    parameters, rmse = LSM(cor1_new, cor2_new, change_form)
    solution = np.array([[parameters[0], parameters[1], parameters[4]],
                         [parameters[2], parameters[3], parameters[5]],
                         [parameters[6], parameters[7], 1]])
    return solution
