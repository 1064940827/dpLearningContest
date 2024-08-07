import os
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
    print(images_info_tensor.shape)  # [1, 840, 192, 192]

    dataset = TensorDataset(images_info_tensor, masks_info_tensor)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)

    class_num = 4
    total_batch = 0
    IoUs = np.zeros(class_num)

    with torch.no_grad():
        for images_batch, masks_batch in dataloader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)

            outputs = net(images_batch)
            preds = torch.argmax(outputs, dim=1)
            print_tensor(masks_batch)
            print_tensor(preds.unsqueeze(1))
            for cls in range(class_num):
                intersection = ((preds == cls) & (masks_batch == cls)).sum().item()
                union = ((preds == cls) | (masks_batch == cls)).sum().item()

                if union != 0:
                    IoUs[cls] += intersection / union
            total_batch += 1
    IoUs = IoUs / total_batch
    mIoUs = IoUs.mean()

    net.train()  # 将模型转化为训练模式

    return IoUs, mIoUs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(4).to(device)
    IoUs, mIoUs = culIOU(net, 16)
    print(IoUs)
    print(mIoUs)
