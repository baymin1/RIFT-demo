# 这个函数实现的是RANSAC方法的一个变体

import numpy as np
from LSM import lsm


def fsc(cor1, cor2, change_form, error_t):
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
        iterations = int(max_iteration)

    most_consensus_number = 0
    cor1_new = np.zeros([M, N])
    cor2_new = np.zeros([M, N])

    for i in range(iterations):
        while (True):
            a = np.floor(1 + (M - 1) * np.random.rand(1, n)).astype(np.int_)[0]
            cor11 = cor1[a]
            cor22 = cor2[a]
            if n == 2 and (a[0] != a[1]) and (cor11[0] != cor11[1]) and (cor22[0] != cor22[1]):
                break
            if n == 3 and (a[0] != a[1] and a[0] != a[2] and a[1] != a[2]) and (cor11[0] != cor11[1]) and (
                    cor11[0] != cor11[2]) and (cor11[1] != cor11[2]) and (cor22[0] != cor22[1]) and (
                    cor22[0] != cor22[2]) and (cor22[1] != cor22[2]):
                break
            if n == 4 and (
                    a[0] != a[1] and a[0] != a[2] and a[0] != a[3] and a[1] != a[2] and a[1] != a[3] and a[2] != a[
                3]) and (cor11[0] != cor11[1]) and (cor11[0] != cor11[2]) and (cor11[0] != cor11[3]) and (
                    cor11[1] != cor11[2]) and (cor11[1] != cor11[3]) and (cor11[2] != cor11[3]) and (
                    cor22[0] != cor22[1]) and (cor22[0] != cor22[2]) and (cor22[0] != cor22[3]) and (
                    cor22[1] != cor22[2]) and (cor22[1] != cor22[3]) and (cor22[2] != cor22[3]):
                break

        parameters, __ = lsm(cor11, cor22, change_form)
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

    parameters, rmse = lsm(cor1_new, cor2_new, change_form)
    solution = np.array([[parameters[0], parameters[1], parameters[4]],
                         [parameters[2], parameters[3], parameters[5]],
                         [parameters[6], parameters[7], 1]])
    return solution
