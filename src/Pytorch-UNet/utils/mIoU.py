import torch

def confusion_matrix(pred, target, num_classes):
    # 期望传入的pred,target为[h*w]
    k=(pred>=0)&(pred<num_classes)
    matrix=torch.bincount(num_classes * target[k].to(torch.int32) + pred[k].to(torch.int32),
                          minlength=num_classes**2).reshape(num_classes,num_classes)
    return matrix
def IoUs_calculate(preds,targets,device,num_classes):
    # 期望传入的preds,targets为[batch_size,h,w]

    confused_matrix=torch.zeros(num_classes,num_classes).to(device)
    assert preds.size() == targets.size(),'计算IoU时预测值与真实值数据大小不同'
    for i in range(preds.size(0)):
        confused_matrix+=confusion_matrix(preds[i].flatten(),targets[i].flatten(),num_classes)
    print(confused_matrix)

    ious=(torch.diag(confused_matrix)/
          torch.maximum((confused_matrix.sum(axis=1)+confused_matrix.sum(axis=0)-confused_matrix.diag()),
                            torch.ones(num_classes,dtype=torch.int32,device=device)))
    print(ious)
    mIoU=ious.nanmean()
    print(mIoU)
    return ious,mIoU



def IoUs_Evaluate(net,dataloader,device,num_classes):
    net.eval()
    num_batch=len(dataloader)
    IoUs=torch.zeros(num_classes)
    mIoU=0.0
    for batch in dataloader:
        images, targets = batch['image'],batch['mask']
        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
        targets = targets.to(device=device, dtype=torch.long)

        preds = net(images)
        preds=preds.argmax(dim=1)
        batch_IoUs,batch_mIoUs=IoUs_calculate(preds,targets,device,num_classes)
        IoUs+=batch_IoUs.cpu()
        mIoU+=batch_mIoUs.cpu()
    net.train()
    return IoUs/num_batch,mIoU/num_batch