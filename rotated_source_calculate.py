import sys
import os
from phasepack import phasecong
from FSC import FSC
from RIFT_descriptor_no_rotation_invariance import RIFT_descriptor_no_rotation_invariance
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

step = 10
device = "cuda:0"

def rotated_source_calculate(img1, img2, best_cost, best_transform, best_img, best_match_point1,best_match_point2):

    # 计算图像中心点，每次旋转angle度
    for angle in range(-180, 180, step):
        print("=================================")
        print(f"旋转角度为{angle}")
        height, width = img1.shape[:2]
        center = (width // 2, height // 2)

        # 旋转源图像
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))

        # 对旋转源图像和参考图像进行描述，并求变换矩阵
        m1, __, __, __, __, eo1, __ = phasecong(img=img1, nscale=4, norient=6, minWaveLength=3, mult=1.6, sigmaOnf=0.75,
                                                g=3, k=1)
        m2, __, __, __, __, eo2, __ = phasecong(img=rotated_img2, nscale=4, norient=6, minWaveLength=3, mult=1.6, sigmaOnf=0.75,
                                                g=3, k=1)
        # 根据最大最小值做归一化
        m1, m2 = map(lambda img: (img.astype(np.float64) - img.min()) / (img.max() - img.min()), (m1, m2))
        cm1 = m1 * 255
        cm2 = m2 * 255
        fast = cv2.FastFeatureDetector_create(nonmaxSuppression=True, type=cv2.FAST_FEATURE_DETECTOR_TYPE_7_12)
        kp1 = fast.detect(np.uint8(cm1), None)
        kp2 = fast.detect(np.uint8(cm2), None)
        # 对关键点根据其响应值进行排序
        kp1 = sorted(kp1, key=lambda kp: kp.response, reverse=True)
        kp2 = sorted(kp2, key=lambda kp: kp.response, reverse=True)
        # 选择响应值最大的关键点
        top_kp1 = kp1[:min(5000, len(kp1))]
        top_kp2 = kp2[:min(5000, len(kp2))]
        m1_point = np.array([kp.pt for kp in top_kp1])
        m2_point = np.array([kp.pt for kp in top_kp2])
        # RIFT描述子对特征点进行描述
        kps1, des1 = RIFT_descriptor_no_rotation_invariance(img1, m1_point, eo1, 96, 4, 6)
        kps2, des2 = RIFT_descriptor_no_rotation_invariance(rotated_img2, m2_point, eo2, 96, 4, 6)
        # 生成匹配，获取匹配点坐标序列
        bf = cv2.FlannBasedMatcher()
        matches = bf.match(des1, des2)
        match_point1 = np.zeros([len(matches), 2], int)
        match_point2 = np.zeros([len(matches), 2], int)
        # 获取匹配点对索引
        for m in range(len(matches)):
            match_point1[m] = kps1[matches[m].queryIdx]
            match_point2[m] = kps2[matches[m].trainIdx]
        # 去重
        match_point2, IA = np.unique(match_point2, return_index=True, axis=0)
        match_point1 = match_point1[IA]

        # 记录当前变换
        current_transform = FSC(match_point1, match_point2, 'affine', 2)

        # 计算当前损失
        desc_indices_1 = [np.where((kps1 == point).all(axis=1))[0][0] for point in match_point1]
        descriptors_1 = des1[desc_indices_1]

        desc_indices_2 = [np.where((kps2 == point).all(axis=1))[0][0] for point in match_point2]
        descriptors_2 = des2[desc_indices_2]

        current_cost = np.mean((descriptors_1 - descriptors_2) ** 2)
        print(f"当前损失为{current_cost}")
        
        # 判断
        if current_cost < best_cost:
            best_cost = current_cost
            best_transform = current_transform
            best_match_point1 = match_point1
            best_match_point2 = match_point2
            best_img = rotated_img2

    return best_match_point1, best_match_point2, best_transform, best_img
