import os
import cv2
import numpy as np
# import openslide
from RIFT import rift
from Rotated_Source_Calculate import rotated_source_calculate

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


# # 单独两张图片测试
s_img1 = cv2.imread("/CSTemp/hjh/myRIFT-master/image/HE.png")
s_img2 = cv2.imread("/CSTemp/hjh/myRIFT-master/image/IHC_CD3.png")
target_width = 1000
target_height = 1000
img1 = cv2.resize(s_img1, (target_width, target_height))
img2 = cv2.resize(s_img2, (target_width, target_height))

# 图像标准化
"""
这是通过首先将RGB图像转换为极坐标CAM16-UCS颜色空间，设置C=0.2和H=0（也可以使用其他值），
然后再转换回RGB（使用Python的color-science包进行RGB到CAM16-UCS转换）
然后，将转换后的RGB图像转换为灰度并进行反转，使背景变暗，而组织变亮。
在处理所有图像（IHC和/或IF）之后，它们被归一化以具有相似的像素强度值分布。
归一化方法:首先确定所有像素值的第5百分位数、平均值和第95百分位数。
然后，这些目标值被用作三次插值中的节点，然后将每个图像的像素值适应到目标值。
最后，使用总变差（TV）进行最终去噪步骤，轻微地平滑图像同时保留边缘，从而减少可能混淆配准的噪声。
"""

# 掩膜，使配准集中在组织上，避免去对齐背景噪声
"""
其基本思想是通过计算每个像素的颜色与背景颜色有多不相似来将背景（玻片）与前景（组织）分离。
第一步将图像转换为CAM16-UCS颜色空间，得到L（亮度）、A和B通道。
对于亮场图像，假定背景将是明亮的，因此背景颜色是具有亮度大于所有像素的99%的像素的平均LAB值。
然后计算每个LAB颜色与背景LAB之间的欧几里得距离，得到一个新的图像D，其中较大的值表示每个像素与背景的颜色有多不同。
然后对D应用Otsu阈值处理，大于该阈值的像素被视为前景，生成一个二进制掩模。
最终的掩模是通过使用OpenCV查找并填充所有轮廓来创建的，从而得到一个覆盖组织区域的掩模。
然后可以在特征检测和非刚性配准过程中应用该掩模，以便将配准焦点集中在组织上。
"""



# RIFT首次配准，得到初始值作为best，再进行优化
costs, inliersIndex, match_point1, match_point2 = rift(img1, img2)

# 旋转优化
inliersIndex, match_point1, match_point2, rotated_img2 = rotated_source_calculate(img1, img2, costs,
                                                                                  inliersIndex, match_point1,
                                                                                  match_point2)

# 选出误差足够小的点对
clean_point1 = match_point1[inliersIndex]
clean_point2 = match_point2[inliersIndex]

kp1 = [cv2.KeyPoint(float(clean_point1[i][0]), float(clean_point1[i][1]), 1) for i in range(len(clean_point1))]
kp2 = [cv2.KeyPoint(float(clean_point2[i][0]), float(clean_point2[i][1]), 1) for i in range(len(clean_point2))]
matches = [cv2.DMatch(i, i, 1) for i in range(len(clean_point1))]

# 可视化
img3 = cv2.drawMatches(img1, kp1, rotated_img2, kp2, matches, None, flags=2)

## 服务器上没法显示图片
# cv2.imshow('img3', img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 可视化图像进行存储
store_path = "/CSTemp/hjh/myRIFT-master/image/registration.png"
cv2.imwrite(store_path, img3)
