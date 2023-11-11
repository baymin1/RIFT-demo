import cv2
import numpy as np
from FSC import FSC
from phasepack import phasecong
from RIFT_descriptor_no_rotation_invariance import RIFT_descriptor_no_rotation_invariance


def RIFT(img1, img2):
    # 计算图像相位一致性：m1：相位一致协方差的最大矩，eo1：一个包含复值卷积结果的列表
    m1, __, __, __, __, eo1, __ = phasecong(img=img1, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                            sigmaOnf=0.75, g=3, k=1)
    m2, __, __, __, __, eo2, __ = phasecong(img=img2, nscale=4, norient=6, minWaveLength=3, mult=1.6,
                                            sigmaOnf=0.75, g=3, k=1)

    # 根据最大最小值做归一化
    m1, m2 = map(lambda img: (img.astype(np.float64) - img.min()) / (img.max() - img.min()), (m1, m2))
    cm1 = m1 * 255
    cm2 = m2 * 255
    fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
    kp1 = fast.detect(np.uint8(cm1), None)  # keypoint  参数1应该为一个单通道图像，故转为unit8格式
    kp2 = fast.detect(np.uint8(cm2), None)

    # 对关键点根据其响应值进行排序
    kp1 = sorted(kp1, key=lambda kp: kp.response, reverse=True)
    kp2 = sorted(kp2, key=lambda kp: kp.response, reverse=True)

    # 选择响应值最大的关键点
    top_kp1 = kp1[:min(5000, len(kp1))]
    top_kp2 = kp2[:min(5000, len(kp2))]

    # # 绘制关键点
    # img1_keypoints = cv2.drawKeypoints(img1, top_kp1, None, color=(0, 255, 0))
    # img2_keypoints = cv2.drawKeypoints(img2, top_kp2, None, color=(0, 255, 0))
    # cv2.imshow('Keypoints', img1_keypoints)
    # cv2.imshow('Keypoints2', img2_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 得到关键点的坐标矩阵
    m1_point = np.array([kp.pt for kp in top_kp1])
    m2_point = np.array([kp.pt for kp in top_kp2])

    # RIFT描述子对特征点进行描述
    kps1, des1 = RIFT_descriptor_no_rotation_invariance(img1, m1_point, eo1, 96, 4, 6)
    kps2, des2 = RIFT_descriptor_no_rotation_invariance(img2, m2_point, eo2, 96, 4, 6)

    # 生成匹配，获取匹配点坐标序列
    bf = cv2.FlannBasedMatcher()
    # matches里面有两幅图的特征点索引和相似度
    matches = bf.match(des1, des2)
    match_point1 = np.zeros([len(matches), 2], int)
    match_point2 = np.zeros([len(matches), 2], int)

    # 根据关键点的索引找到其在kps1中的坐标
    for m in range(len(matches)):
        match_point1[m] = kps1[matches[m].queryIdx]
        match_point2[m] = kps2[matches[m].trainIdx]

    # 去重，得到去重后的坐标数组，IA是在matches中点对的索引。
    match_point2, IA = np.unique(match_point2, return_index=True, axis=0)
    match_point1 = match_point1[IA]

    # 从 match_point1 中提取描述符索引
    des_indices_1 = [np.where((kps1 == point).all(axis=1))[0][0] for point in match_point1]
    descriptor1 = des1[des_indices_1]

    des_indices_2 = [np.where((kps2 == point).all(axis=1))[0][0] for point in match_point2]
    descriptor2 = des2[des_indices_2]
    # 得到两幅图匹配点的描述符并计算均方差误差
    best_costs = np.mean((descriptor1 - descriptor2) ** 2)

    # 计算初始值作为最优
    best_match_point1 = match_point1
    best_match_point2 = match_point2
    best_transform = FSC(match_point1, match_point2, 'affine', 2)

    return best_costs, best_match_point1, best_match_point2, best_transform
