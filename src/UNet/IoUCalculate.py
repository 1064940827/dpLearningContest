import os
import random as rd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader
from UNet import UNet
from printAndSavaImage import print_tensor


def culIOU(net, batchsize=16):
    image_rootpath = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\images\\test'  # 用于训练的图像路径
    mask_rootpath = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\annotations\\test'  # 用于训练的标签路径

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.eval()  # 将模型转化为评测模式

    _images_info = []
    images_path = [os.path.join(image_rootpath, image) for image in os.listdir(image_rootpath)]
    _masks_info = []
    masks_path = [os.path.join(mask_rootpath, mask) for mask in os.listdir(mask_rootpath)]
    for image_path, mask_path in zip(images_path, masks_path):
        image = Image.open(image_path).resize((192, 192), Image.Resampling.BICUBIC)
        mask = Image.open(mask_path).resize((192, 192), Image.Resampling.NEAREST)
        image_info = np.array(image)
        mask_info = np.array(mask)
        _images_info.append(image_info)
        _masks_info.append(mask_info)
    images_info = np.array(_images_info)[:, :, :, 0]
    masks_info = np.array(_masks_info)

    # print(images_info.shape)   (840, 192, 192)
    # print(masks_info.shape)   (840, 192, 192)

    images_info_tensor = torch.tensor(images_info.astype('float32')[np.newaxis, :, :, :]).permute(1, 0, 2, 3)
    masks_info_tensor = torch.tensor(masks_info.astype('long')[np.newaxis, :, :, :]).permute(1, 0, 2, 3)
    # print(images_info_tensor.shape)  # [840,1,192,192]

    dataset = TensorDataset(images_info_tensor, masks_info_tensor)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    class_num = 4
    total_batch = 0

    count_1 = 0
    count_2 = 0
    count_3 = 0
    IoUs = np.zeros(class_num)

    with torch.no_grad():
        for images_batch, masks_batch in dataloader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            outputs = net(images_batch)
            preds = torch.argmax(outputs, dim=1)

            # 打印测试集中的随机图像及其对应的预测结果
            # if total_batch % rd.randint(18, 50) == 1:
            #     print_tensor(masks_batch)
            #     print_tensor(preds.unsqueeze(1))

            # 统计含有缺陷1,2,3的图片数量,该过程可能过于耗时,可在单次运行求值后直接赋值
            for i in range(masks_batch.shape[0]):
                preds_single = preds[i].cpu()
                masks_single = masks_batch[i].cpu()
                unique_eles = np.unique(masks_single)
                if 1 in unique_eles:
                    count_1 += 1
                if 2 in unique_eles:
                    count_2 += 1
                if 3 in unique_eles:
                    count_3 += 1
                # print(masks_single.shape) # torch.Size([1, 192, 192])
                for cls in range(class_num):
                    intersection = ((preds_single == cls) & (masks_single.squeeze(0) == cls)).sum().item()
                    union = ((preds_single == cls) | (masks_single.squeeze(0) == cls)).sum().item()

                    if union != 0:
                        IoUs[cls] += intersection / union
            total_batch += 1
    IoUs[0] = IoUs[0] / images_info_tensor.shape[0]
    IoUs[1] = IoUs[1] / count_1
    IoUs[2] = IoUs[2] / count_2
    IoUs[3] = IoUs[3] / count_3
    mIoUs = np.mean(IoUs[1:])
    # print(count_1, count_2, count_3)
    net.train()  # 将模型转化为训练模式

    return IoUs, mIoUs


if __name__ == '__main__':
    x = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                      [[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]])
    y = torch.tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                      [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
    print(x.shape)
    print(y.shape)
    z1 = (x == 0)
    print(z1)
    z2 = (y == 0)
    print(z2)
    z3 = z1 & z2
    print(z3)
    z4 = torch.logical_and(z1, z2)
    print(z4)
    print(z3.sum().item())
