import numpy as np
import math
from einops import repeat

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

        if y2 - y1 != patch_size or x2 - x1 != patch_size:
            kps_to_ignore[0][i] = 0
            continue

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