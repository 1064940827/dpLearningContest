import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from data import MyDataset
from net import UNet
from printTest import save_combined_image

"""
超参数，需训练时调整
"""
batch_size = 8
epochs = 200

"""
路径参数，第一次运行时调整
"""
image_path = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\images\\training'  # 用于训练的图像路径
mask_dir = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\annotations\\training'  # 用于训练的标签路径、
weight_save_path = "model/{}.pth".format(datetime.now().strftime("%y%m%d-%H%M%S"))  # 用于保存权重的路径

_f_set = [f for f in os.listdir("model") if f.endswith(".pth")]
weight_read_path = "null"
if len(_f_set) != 0:
    weight_read_path = os.path.join("model", max(_f_set,
                                                 key=lambda x: os.path.getctime(
                                                     os.path.join("model", x))))  # 用于读取权重的路径，默认从最新的权重读取，后期可以根据需要修改

"""
日志打印参数
"""
printLoss_epochs = 5
weight_save_epochs = 20
save_image = True

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(MyDataset(image_path, mask_dir), batch_size=batch_size, shuffle=True)
    net = UNet(4).to(device)
    if os.path.exists(weight_read_path):
        net.load_state_dict(torch.load(weight_read_path))
        print('load weights from {}'.format(weight_read_path))
    else:
        print('no weights from {}'.format(weight_read_path))

    opt = optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    epoch = 0
    for epoch in range(epochs):
        for i, (images, masks) in enumerate(data_loader):

            images = images.to(device)
            masks = masks.to(device).squeeze(1)

            outputs = net(images)
            loss = loss_fn(outputs, masks.long())
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % printLoss_epochs == 0:
                print(f'轮次: {epoch}, 训练集索引: {i}, loss: {loss.item()}')

            if save_image & (epoch % 10 == 1):
                save_combined_image(epoch, images[0], masks[0], outputs[0])
        if epoch % weight_save_epochs == 0:
            torch.save(net.state_dict(), weight_save_path)
            print("模型参数保存成功!")
