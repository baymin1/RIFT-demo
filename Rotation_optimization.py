import sys
import os
from phasepack import phasecong
from FSC import fsc
from RIFT import rift
from RIFT_Descriptor_No_Rotation_Invariance import RIFT_descriptor_no_rotation_invariance
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

step = 10
device = "cuda:0"


def rotated_source_calculate(img1, img2, best_mean_error, best_inliersIndex, best_match_point1, best_match_point2):
    best_img = img2
    # 计算图像中心点，每次旋转angle度
    for angle in range(step, 360, step):
        print("=================================")
        print(f"旋转角度为{angle}")
        height, width = img1.shape[:2]
        center = (width // 2, height // 2)

        # 旋转源图像
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))

        # RIFT
        current_mean_error, inliersIndex, match_point1, match_point2 = rift(img1, rotated_img2)
        print(f"current_cost为{current_mean_error}")

        # 判断
        if current_mean_error < best_mean_error:
            best_mean_error = current_mean_error
            best_inliersIndex = inliersIndex
            best_match_point1 = match_point1
            best_match_point2 = match_point2
            best_img = rotated_img2

    return best_inliersIndex,best_match_point1, best_match_point2, best_img
