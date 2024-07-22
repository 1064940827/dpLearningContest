import os

import cv2

img_dir = "D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\annotations\\training"
gray_img_dir = "D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-gray\\annotations\\training"

def is_gray(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    return len(img.shape) == 2

if not os.path.isdir(gray_img_dir):
    os.makedirs(gray_img_dir)

for filename in os.listdir(img_dir):
    img=cv2.imread(os.path.join(img_dir,filename))
    img_gray=img[:,:,0]
    cv2.imwrite(os.path.join(gray_img_dir,filename),img_gray)
