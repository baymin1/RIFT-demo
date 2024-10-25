import os
import cv2
import numpy as np
import openslide
from RIFT import rift
from FSC import fsc
from Rotation_optimization import rotation_optimization

s_img1 = cv2.imread("image/HE.png")
s_img2 = cv2.imread("image/IHC_CD3.png")
target_width = 1000
target_height = 1000

# 下采样
img1 = cv2.resize(s_img1, (target_width, target_height))
img2 = cv2.resize(s_img2, (target_width, target_height))

# 初始配准
costs, inliersIndex, match_point1, match_point2 = rift(img1, img2)

# 旋转优化
inliersIndex, match_point1, match_point2, img2 = rotation_optimization(img1, img2, costs,
                                                                       inliersIndex, match_point1,
                                                                       match_point2)

# 选出误差足够小的点对
clean_point1 = match_point1[inliersIndex]
clean_point2 = match_point2[inliersIndex]

kp1 = [cv2.KeyPoint(float(clean_point1[i][0]), float(clean_point1[i][1]), 1) for i in range(len(clean_point1))]
kp2 = [cv2.KeyPoint(float(clean_point2[i][0]), float(clean_point2[i][1]), 1) for i in range(len(clean_point2))]
matches = [cv2.DMatch(i, i, 1) for i in range(len(clean_point1))]

# 匹配
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

# 可视化
cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


