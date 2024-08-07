"""
定义了训练数据集
"""


import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class MyDataset(Dataset):

    def __init__(self, image_dir, mask_dir,
                 transform=transforms.Compose([
                     transforms.ToTensor()])):
        """
        :param image_dir:图像存放的路径
        :param mask_dir: 标签存放的路径
        :param transform: 对输入图像所做的转化，默认进行向灰度图像的转化、向tensor的转化
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [os.path.join(image_dir, image) for image in os.listdir(image_dir)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 构建钢材图像和标签图像的路径名
        image_path = self.images[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path).replace('.jpg', '.png'))

        # 官方提供的数据集中，钢材图像位深度24bit，为RGB图，但是置于PS中查看发现RGB通道值全部相等，故转化为灰度图像不丢失任何信息
        image_ori=Image.open(image_path).convert('L').resize((192,192),Image.Resampling.BICUBIC)
        mask_ori = Image.open(mask_path).resize((192, 192), Image.Resampling.NEAREST)


        # 对图像应用变换
        image = self.transform(image_ori)
        mask = self.transform(mask_ori)

        return image, mask


if __name__ == '__main__':
    dataset = MyDataset(image_dir='D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\images\\training',
                        mask_dir='D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\annotations\\training')


