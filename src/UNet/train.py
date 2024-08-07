import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datetime import datetime
from data import MyDataset
from UNet import UNet
from printAndSavaImage import save_combined_image
from IoUCalculate import culIOU

"""
超参数，需训练时调整
"""
batch_size = 16
epochs = 200

"""
路径参数，第一次运行时调整
"""
load_weight = False  # 是否读取已经存在的模型，若为true则默认读取最新模型，如需要读取其他模型请修改weight_read_path


image_path = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\images\\training'  # 用于训练的图像路径
mask_dir = 'D:\\ProjectsCollection\\dlContest\\data\\NEU_Seg-main\\annotations\\training'  # 用于训练的标签路径
weight_save_path = "model/{}.pth".format(datetime.now().strftime("%y%m%d-%H%M%S"))  # 用于保存权重的路径
weight_dir = 'src/UNet/model'
weight_read_path = "null"


"""
日志打印参数
"""
printLoss_i = 20 # 每训练printLoss_i个batch后打印损失值
sava_weight = False  # 是否保存模型权重
weight_save_epochs = 20 # 每训练weight_save_epochs个epochs后保存权重
save_image = False # 是否保存预测结果测试图像
calIoUs=True
calIoUs_epochs=5 # 每训练calIoUs_epochs个epochs后计算交并比


if __name__ == '__main__':
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    _f_set = [f for f in os.listdir(weight_dir) if f.endswith(".pth")]
    if len(_f_set) != 0:
        weight_read_path = os.path.join(weight_dir, max(_f_set,
                                                        key=lambda x: os.path.getctime(
                                                            os.path.join(weight_dir,x))))
        # 用于读取权重的路径，默认从最新的权重读取，后期可以根据需要修改

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_loader = DataLoader(MyDataset(image_path, mask_dir), batch_size=batch_size, shuffle=True)
    net = UNet(4).to(device)
    if load_weight & os.path.exists(weight_read_path):
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
            # print(outputs.shape) [8, 4, 192, 192]
            if i % printLoss_i == 0:
                print(f'轮次: {epoch}, 训练集索引: {i}, loss: {loss.item()}')

            # if save_image and (epoch % 10 == 1):
            #     save_combined_image(epoch, images[0], masks[0], outputs[0])
        if epoch % calIoUs_epochs == 0 and calIoUs:
            IoUs, mIoU = culIOU(net, batch_size)
            # print(f'夹杂物IoU:{IoUs[1]},补丁IoU:{IoUs[2]},划痕IoU:{IoUs[3]}, mIoU:{mIoU}')
            print(IoUs, mIoU)
        if epoch % weight_save_epochs == 0 and sava_weight:
            torch.save(net.state_dict(), weight_save_path)
            print("模型参数保存成功!")
