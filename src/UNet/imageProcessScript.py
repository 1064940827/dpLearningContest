"""
用于高亮标签图像的脚本
"""

import os
import numpy as np
import cv2

img_dir = "D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-gray\\annotations\\test"
strong_img_dir = "D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-enhance\\annotations\\test"


def enhance_img(img_path):
    img = cv2.imread(img_path)[:, :, 0]
    color_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    color_img[img == 0] = [255, 255, 255]  # 白色
    color_img[img == 1] = [0, 0, 255]  # 红色
    color_img[img == 2] = [0, 255, 0]  # 绿色
    color_img[img == 3] = [255, 0, 0]  # 蓝色

    image_save_dir = os.path.join(strong_img_dir, os.path.basename(img_path))
    cv2.imwrite(image_save_dir, color_img)


if not os.path.isdir(strong_img_dir):
    os.makedirs(strong_img_dir)

for filename in os.listdir(img_dir):
    enhance_img(os.path.join(img_dir, filename))
    # img = cv2.imread(os.path.join(img_dir, filename))
    #
    # exit(1)
