import numpy as np
import torch
import torch.nn.functional as F
import argparse
import os
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from DC_UNet import DC_Unet
from dataloader import TestDataset
from train import confused_matrix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--weight_path", type=str,
                        default="", help="网络权重文件路径")

    parser.add_argument("-rp", "--result_save_path", type=str,
                        default="./result/", help="结果保存路径")

    parser.add_argument("-td", "--test_dir_path", type=str,
                        default="/root/autodl-tmp/data/", help="测试数据根路径")
    opt = parser.parse_args()



    image_root = '{}/images/test/'.format(opt.test_dir_path) # 测试图像的所在目录
    gt_root = '{}/annotations/test/'.format(opt.test_dir_path) # 测试图像对应标签的所在目录
    result_path = '{}/{}/'.format(opt.result_save_path, datetime.now().strftime("%Y-%m-%d %H:%M"))
    result_log='{}/log.txt'.format(result_path)

    os.makedirs(result_path, exist_ok=True)
    logFile=open(result_log,'w')

    test_dataset = TestDataset(image_root, gt_root, 256)
    test_dataset_timeCalculate = TestDataset(image_root, gt_root, 256)

    torch.cuda.set_device(0)
    model = DC_Unet()

    if opt.weight_path == "":
        print("No weight path input!")
    else:
        model.load_state_dict(torch.load(opt.weight_path))
    model = model.cuda().eval()

    confused_matrix_sum = np.zeros((4, 4), dtype=np.int64)

    for i in tqdm(range(test_dataset.size),desc='IoU calculate progress',unit="image",ncols=120):
        image, target, name = test_dataset.load_data()

        image = image.cuda()
        # 期望输入image.size=[1,c,h,w]
        result = model(image)
        # 获取图像的预测结果
        result = result.argmax(dim=1).data.cpu()
        result = F.interpolate(result, size=(200, 200), mode='nearest').numpy()[0,0,:]

        # 结果可视化
        enhanced_result = np.zeros((result.shape[0], result.shape[1],3), dtype=np.int64)
        enhanced_result[result==0] = [255,255,255]
        enhanced_result[result==1] = [0,0,255]
        enhanced_result[result==2] = [0,255,0]
        enhanced_result[result==3] = [255,0,0]

        img = Image.fromarray(enhanced_result)
        image.save(result_path+'{}-result.png'.format(name))

        target = target.data.cpu()
        confused_matrix_temp = confused_matrix(result.flatten(), target.flatten(), 4)
        confused_matrix_sum += confused_matrix_temp
        logFile.write(name)
        logFile.write(confused_matrix_temp)

    IoUs = (np.diag(confused_matrix_sum) /
            np.maximum(
                (confused_matrix_sum.sum(axis=1) + confused_matrix_sum.sum(axis=0) - np.diag(confused_matrix_sum)),
                torch.ones(4, dtype=torch.int64)))
    mIoU = IoUs.nanmean()
    logFile.write("总混淆矩阵为:")
    logFile.write(IoUs)
    logFile.write("mIoUs为:")
    logFile.write(mIoU)
    time1 = datetime.now()
    for i in range(test_dataset_timeCalculate.size):
        image, target, name = test_dataset_timeCalculate.load_data()
        image = image.cuda()
        result = model(image)
    time2 = datetime.now()
    fps=test_dataset.size/((time2-time1).total_seconds())
    print("图像处理完毕,处理结果汇总:")
    print("总混淆矩阵:")
    print(confused_matrix_sum)
    print("IoUs:{},mIoU:{},fps:{}".format(IoUs,mIoU,fps))



