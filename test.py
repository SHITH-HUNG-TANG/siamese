from __future__ import print_function, division
import torch
import numpy as np
from gait_siamese import siamese
import torch.nn as nn
from torch.autograd import Variable
from data_loader import RescaleT
# from data_loader import ToTensofLab
from data_loader import GaitDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms, utils
import torch.nn.functional as F
import cv2
import glob
import os

def test_model(model,test_dataloader,loss_function):
    # 切换模型为预测模型
    model.eval()
    running_loss = []
    label_list = []
    predict_list = []
    # 不记录模型梯度信息
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            left, right, label = data
            if torch.cuda.is_available():
                left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()
            predict = model(left_v, right_v)
            loss = loss_function(predict, label_v)
            running_loss.append(loss.item())
            # label = np.array(label)
            for j in range(len(label)):
                label_list.append(label[j])
                predict_list.append(predict[j])
    loss = np.mean(np.array(running_loss))
    test_acc = np.mean(np.array(label) == (np.array(predict.cpu())>=0.5))
    return loss,test_acc



if __name__ == '__main__':

    epoch_num = 100
    batch_size = 256
    load_path = "./weight/onlyNM128/epoch-9_trainLoss-0.024642440901114625_validLoss-0.208388474466669_validAcc-92.28%.t7"
    # ------- define model --------
    model = siamese(in_ch=1)
    if torch.cuda.is_available():
        model.cuda()
    # -----------------------train and valid -----------------------

    try:
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['state']) # 從字典中依次讀取
        startEpoch = checkpoint['epoch']
        print('===> Load last checkpoint data')
    except FileNotFoundError:
        print('Can\'t found .t7 file')


    # 切换模型为预测模型
    model.eval()
    # -------  define loss function --------
    loss_function = nn.BCELoss(size_average=True)

    # -----------------------set test data-----------------------
    for test_name in sorted(glob.glob("./load_data/averageData/cl/*")):
        test_data_list = []
        test_label_list = []
        with open(test_name, "r") as txt:
            for line in txt.readlines():
                line.strip()
                test_data_list.append([line.split(",")[0],line.split(",")[1]])
                test_label_list.append(int(line.split(",")[2]))


        test_dataset = GaitDataset(
            img_name_list=test_data_list,
            label_list=test_label_list,
            transform=transforms.Compose([
                transforms.Resize(128,2),
                transforms.ToTensor()])
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

        # ----------------------- test -----------------------
        running_loss = []
        correct = 0
        total = len(test_dataloader.dataset)
        # 不记录模型梯度信息
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                left, right, label = data
                if torch.cuda.is_available():
                    left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()
                predict = model(left_v, right_v)
                loss = loss_function(predict, label_v)
                running_loss.append(loss.item())
                correct += torch.eq(predict > 0.5, label_v).sum().float().item()

        test_loss = np.mean(np.array(running_loss))
        # test_acc = np.mean(np.array(label) == (np.array(predict.cpu()) >= 0.5))
        test_acc = correct/total
        print("test_name:{} test loss: {:.3f}, test acc:{:.3f}%".format(os.path.basename(test_name)[:-4],test_loss,test_acc*100))



