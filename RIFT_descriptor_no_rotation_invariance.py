import numpy as np
import math
from einops import repeat

"""
img: 输入图像。
kps: 关键点坐标。
eo: 一个由多个尺度和方向组成的滤波器响应列表。
patch_size: 用于计算描述符的图像块的大小。
s: 尺度数量。
o: 方向数量。

构造一个方向显著性图(CS): 这里，对于每个方向和每个尺度，计算图像的方向显著性。
确定最显著的方向: 使用np.argmax确定每个像素点最显著的方向。
初始化描述符矩阵: 根据关键点数量和方向数量初始化一个描述符矩阵。
计算RIFT描述符:
对于每个关键点，提取一个大小为patch_size的图像块。将该块划分为ns x ns的子块。
对于每个子块，使用直方图统计每个方向的出现次数，从而获得RIFT描述符。标准化描述符: 对于每个RIFT描述符，进行L2范数标准化。

输出：
KPS_out: 过滤后的关键点坐标（可能去除了一些无效的关键点）。
des_out: 对应的RIFT描述符。
"""


def RIFT_descriptor_no_rotation_invariance(img, kps, eo, patch_size, s, o):
    KPS = kps.T
    (yim, xim, _) = np.shape(img)
    CS = np.zeros([yim, xim, o], np.float64)
    for j in range(o):
        for i in range(s):
            # 将各个scale的变换结果的幅度相加
            CS[..., j] = CS[..., j] + np.abs(np.array(eo[j][i]))
    mim = np.argmax(CS, axis=2)
    des = np.zeros([36 * o, np.size(KPS, 1)])
    kps_to_ignore = np.ones([1, np.size(KPS, 1)], bool)
    for k in range(np.size(KPS, 1)):
        x = round(KPS[0][k])
        y = round(KPS[1][k])
        x1 = max(0, x - math.floor(patch_size / 2))
        y1 = max(0, y - math.floor(patch_size / 2))
        x2 = min(x + math.floor(patch_size / 2), np.size(img, 1))
        y2 = min(y + math.floor(patch_size / 2), np.size(img, 0))

        # if y2 - y1 != patch_size or x2 - x1 != patch_size:
        #     kps_to_ignore[0][i] = 0
        #     continue

        patch = mim[y1:y2, x1:x2]
        ys, xs = np.size(patch, 0), np.size(patch, 1)
        ns = 6;
        RIFT_des = np.zeros([ns, ns, o])
        for j in range(ns):
            for i in range(ns):
                clip = patch[round((j) * ys / ns):round((j + 1) * ys / ns),
                       round((i) * xs / ns): round((i + 1) * xs / ns)]
                x, __ = np.histogram(clip.T.flatten(), bins=6, range=(0, o), density=False)
                te = RIFT_des[j][i]
                RIFT_des[j][i] = x.reshape(1, 1, len(x))
        RIFT_des = RIFT_des.T.flatten()

        df = np.linalg.norm(RIFT_des)
        if df != 0:
            RIFT_des = RIFT_des / df
        des[:, [k]] = np.expand_dims(RIFT_des, axis=1)
    m = repeat(kps_to_ignore, '1 n -> c n', c=2)
    v = KPS[m]
    KPS_out = v.reshape(2, int(len(v) / 2)).T
    w = repeat(kps_to_ignore, '1 n -> c n', c=len(des))
    z = des[w]
    des_out = z.reshape(len(des), int(len(z) / len(des))).T
    des_out = np.float32(des_out) * 100
    return KPS_out, des_out