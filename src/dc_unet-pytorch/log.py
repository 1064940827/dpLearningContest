from datetime import datetime
import csv
import numpy as np


class TrainLog:
    def __init__(self,logPath,opt):
        self.generateTime=datetime.now().strftime("%Y-%m-%d %H:%M")
        self.totalEpoch=opt.epoch
        self.lossLogFileName = '{}/lossRecord{}-lr={}-batchSize={}.txt'.format(logPath,self.generateTime,opt.lr,opt.batchsize)
        self.lossCSVFileName = '{}/lossRecord{}-lr={}-batchSize={}.csv'.format(logPath,self.generateTime,opt.lr,opt.batchsize)
        self.IoULogFileName='{}/IoURecord{}-lr={}-batchSize={}.txt'.format(logPath,self.generateTime,opt.lr,opt.batchsize)
        self.IoUCSVFileName = '{}/IoURecord{}-lr={}-batchSize={}.csv'.format(logPath, self.generateTime, opt.lr,opt.batchsize)
        self.bestLoss=np.inf
        self.bestIoU=0.0
        self.isBestIoUGet=False
        with open(self.lossLogFileName,'w') as lossLogFile:
            lossLogFile.write('------start training!------')
            if opt.augmentation:
                lossLogFile.write('开启随机翻转/裁剪(数据增强)\n')
            if opt.adjust_lr:
                lossLogFile.write('开启学习率调整\n')
        with open(self.lossCSVFileName,'w') as lossCSVFile:
            writer = csv.writer(lossCSVFile)
            writer.writerow(['epoch','totalEpoch','step','totalStep','lr','loss'])
        with open(self.IoUCSVFileName,'w') as IoUCSVFile:
            writer = csv.writer(IoUCSVFile)
            writer.writerow(['epoch','backgroundIoU','InclusionIoU','PatchIoU','ScratchIoU','mIoU'])
    def writeLossLog(self,loss,epochNow,lr,step,totalStep):
        with open(self.lossLogFileName,'a') as lossLogFile:
            lossLogFile.write('Epoch:{}/{} Step:{}/{} Lr:{} Loss:{}\n'.format(self.totalEpoch,epochNow,step,totalStep,lr,loss))
        with open(self.lossCSVFileName,'a') as lossCSVFile:
            writer = csv.writer(lossCSVFile)
            writer.writerow([epochNow,self.totalEpoch,step,totalStep,lr,loss])
        if loss < self.bestLoss:
            self.bestLoss=loss

    def writeIoULog(self,epochNow,IoUs,mIoU):
        with open(self.IoULogFileName, 'a') as IoULogFile:
            IoULogFile.write('Epoch:{}/{} 当前IoU:{},mIoU:{}\n'.format(self.totalEpoch,epochNow,IoUs,mIoU))
        with open(self.IoUCSVFileName, 'a') as IoUCSVFile:
            writer = csv.writer(IoUCSVFile)
            writer.writerow([epochNow,IoUs[0],IoUs[1],IoUs[2],IoUs[3],mIoU])
        if mIoU > self.bestIoU:
            self.bestIoU=mIoU
            self.isBestIoUGet=True
        else :
            self.isBestIoUGet=False
    def getBestIoU(self):
        return self.bestIoU
    def getBestLoss(self):
        return self.bestLoss
