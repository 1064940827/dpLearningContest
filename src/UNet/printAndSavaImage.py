import torch
import os
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_combined_image(epoch, image, mask, output):
    """
    将原始图像、掩码和模型输出合并并保存。
    参数:
        epoch: 当前轮次，用于生成文件名。
        image: 原始图像。
        mask: 真实掩码。
        output: 模型输出。
        save_dir: 保存合并图像的目录。
    """

    save_dir = os.path.join('preImage', f"epoch_{epoch}.jpg")

    # 转换模型输出为图片格式
    output_img = torch.argmax(output, dim=1)  # 转为最可能的类别
    output_img = output_img.byte().cpu().numpy()

    # 使用 PIL 将tensor转换为图像
    image_pil = TF.to_pil_image(image.cpu().squeeze(0))  # 假设image是单通道的灰度图
    mask_pil = Image.fromarray(np.uint8(mask.cpu().squeeze(0).numpy() * 85))  # 将标签扩展至0-255范围
    output_pil = Image.fromarray(np.uint8(output_img[0] * 85))  # 将输出扩展至0-255范围

    # 合并三张图像
    combined_image = Image.new('RGB', (image_pil.width * 3, image_pil.height))
    combined_image.paste(image_pil.convert('RGB'), (0, 0))
    combined_image.paste(mask_pil.convert('RGB'), (image_pil.width, 0))
    combined_image.paste(output_pil.convert('RGB'), (image_pil.width * 2, 0))

    # 保存图像
    combined_image.save(save_dir)


def print_tensor(tensor, isMask=True):
    """
    期望输入的tensor形状为[N, C, H, W],位于GPU上
    则默认打印图像[0,C,H,W]
    """
    img = tensor.to('cpu')[0].permute(1, 2, 0).numpy()

    if isMask:
        img = img * 85

    plt.imshow(img,cmap='viridis')
    plt.colorbar()
    plt.show()
