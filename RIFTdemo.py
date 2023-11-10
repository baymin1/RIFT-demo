import os
import cv2
import numpy as np
import openslide
from FSC import FSC
from phasepack import phasecong
from RIFT_descriptor_no_rotation_invariance import RIFT_descriptor_no_rotation_invariance
from rotated_source_calculate import rotated_source_calculate


# # 遍历文件夹进行配准
# HE_images_directory = r'D:\datasets\HE'
# IHC_images_directory = r'D:\datasets\IHC'
#
# HE_svs_files = [f for f in os.listdir(HE_images_directory) if f.endswith('.svs')]
# IHC_svs_files = [f for f in os.listdir(IHC_images_directory) if f.endswith('.svs')]
#
# for i in range(0, len(HE_svs_files)):
#     image1_path = os.path.join(HE_images_directory, HE_svs_files[i])
#     image2_path = os.path.join(IHC_images_directory, IHC_svs_files[i])
#
#     slide1 = openslide.OpenSlide(image1_path)
#     slide2 = openslide.OpenSlide(image2_path)
#
#     level = 2
#     img_array = np.array(slide1.read_region((0, 0), level, slide1.level_dimensions[level]))
#     img_array2 = np.array(slide2.read_region((0, 0), level, slide2.level_dimensions[level]))
#
#     # Convert the ARGB image to BGR   BGR为opencv中要求的图片格式
#     bgr_image1 = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
#     bgr_image2 = cv2.cvtColor(img_array2, cv2.COLOR_RGBA2BGR)
#     target_width = 1000
#     target_height = 1000
#     img1 = cv2.resize(bgr_image1, (target_width, target_height))
#     img2 = cv2.resize(bgr_image2, (target_width, target_height))

"testw"
# # 单独两张图片测试
s_img1 = cv2.imread(r"D:\samples\HE.png")
s_img2 = cv2.imread(r"D:\samples\IHC_PD-L1.png")
target_width = 1000
target_height = 1000
img1 = cv2.resize(s_img1, (target_width, target_height))
img2 = cv2.resize(s_img2, (target_width, target_height))

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

m1_point = np.array([kp.pt for kp in top_kp1])
m2_point = np.array([kp.pt for kp in top_kp2])

# RIFT描述子对特征点进行描述
kps1, des1 = RIFT_descriptor_no_rotation_invariance(img1, m1_point, eo1, 96, 4, 6)
kps2, des2 = RIFT_descriptor_no_rotation_invariance(img2, m2_point, eo2, 96, 4, 6)

# 生成匹配，获取匹配点坐标序列，python中没有matlab的matchFeatures函数的对应函数，因此用了opencv
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

# 计算初始值作为最优
costs = np.mean((des2 - des1) ** 2, axis=1)
lowest_costs = np.sort(costs)[0:500]  # 挑500个最小的cost出来，做平均

best_costs = np.mean(lowest_costs)
best_transform = FSC(match_point1, match_point2, 'affine', 2)
best_img = img2
best_match_point1 = match_point1
best_match_point2 = match_point2

# 对变换矩阵进行优化
match_point1, match_point2, H, rotated_img2 = rotated_source_calculate(img1, img2, best_costs, best_transform,best_img, best_match_point1, best_match_point2)


# 计算误差
Y_ = np.ones([3, len(match_point1)])
Y_[:2] = match_point1.T
Y_ = H.dot(Y_)  # 得到图1内点进行变换后的坐标矩阵

Y_[0] = Y_[0] / Y_[2]
Y_[1] = Y_[1] / Y_[2]

E = np.sqrt(sum(np.power((Y_[0:2] - match_point2.T), 2)))
inliersIndex = np.squeeze(np.argwhere(E < 3))  # 3

# 误差足够小的点对连线可视化
clean_point1 = match_point1[inliersIndex]
clean_point2 = match_point2[inliersIndex]

kp1 = [cv2.KeyPoint(float(clean_point1[i][0]), float(clean_point1[i][1]), 1) for i in range(len(clean_point1))]
kp2 = [cv2.KeyPoint(float(clean_point2[i][0]), float(clean_point2[i][1]), 1) for i in range(len(clean_point2))]

matches = [cv2.DMatch(i, i, 1) for i in range(len(clean_point1))]
img3 = cv2.drawMatches(img1, kp1, rotated_img2, kp2, matches, None, flags=2)
cv2.imshow('img3', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 可视化图像进行存储
# store_path = f"D:\\img{i}.jpg"
# cv2.imwrite(store_path, img3)


