import os
from PIL import Image
from tqdm import tqdm
import time

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode = 0  # 设置为0以计算mIoU并生成预测结果，或设置为1和2以进行单独的步骤
    num_classes = 4
    name_classes = ["_background_", "Inclusions", "Patches", "Scratches"]
    VOCdevkit_path = 'VOCdevkit'
    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/testanno/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
        
        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result and calculate FPS.")
        start_time = time.time()  # 开始计时
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/testimage/" + image_id + ".jpg")
            image = Image.open(image_path)
            # 假设deeplab.get_miou_png是进行预测并返回PIL图像的方法
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))

            # 计算FPS，可以在一定迭代后计算平均值，或者在循环结束后计算总的平均FPS
            # 这里我们选择在处理完所有图像后计算总的平均FPS
        end_time = time.time()  # 结束计时
        total_time = end_time - start_time
        total_frames = len(image_ids)
        fps = total_frames / total_time if total_time > 0 else 0
        print(f"Prediction and FPS calculation done. Average FPS: {fps:.2f}")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)