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
import os


# ------- 1. define loss function --------
#
# class ContrastiveLoss(torch.nn.Module):
#     """
#     Contrastive loss function.
#     Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
#     """
#
#     def __init__(self, margin=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#
#     # def forward(self, output1, output2, label):
#     #     euclidean_distance = F.pairwise_distance(output1, output2)
#     #     loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
#     #                                   (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#     #
#     #     return loss_contrastive
#     def forward(self, y_true, y_pred):
#         loss_contrastive = torch.mean((1 - y_true) * torch.pow(self.margin - y_pred, 2) + y_true * torch.pow(y_pred,2))
#         return loss_contrastive

def train_model(model,train_dataloader,optimizer,loss_function):

    model.train()
    running_loss = 0
    for i, data in enumerate(train_dataloader):
        left, right, label = data

        # wrap them in Variable
        if torch.cuda.is_available():
            left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()

        # y zero the parameter gradients
        optimizer.zero_grad()
        predict = model(left_v, right_v)
        loss = loss_function(predict, label_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    loss = running_loss / len(train_dataloader)
    return loss

def valid_model(model,valid_dataloader,loss_function):
    # 切换模型为预测模型
    model.eval()
    running_loss = 0
    correct = 0
    # 不记录模型梯度信息
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            left, right, label = data
            if torch.cuda.is_available():
                left_v, right_v, label_v = left.cuda(), right.cuda(), label.cuda()
            predict = model(left_v, right_v)
            loss = loss_function(predict, label_v)
            running_loss += loss.item()
            correct += torch.eq(predict > 0.9, label_v).sum().float().item()
    loss = running_loss / len(valid_dataloader)
    valid_acc = correct / len(valid_dataloader.dataset)
    return loss,valid_acc



if __name__ == '__main__':
    startEpoch = 0
    endEpoch = 500
    batch_size = 256
    save_path = "./weight/onlyNM128"
    # ------- define model --------
    model = siamese(in_ch=1)
    if torch.cuda.is_available():
        model.cuda()


    # ------- define optimizer --------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)
    # -------  define loss function --------
    loss_function = nn.BCELoss(size_average=True)


    # -----------------------set train data-----------------------
    train_data_list = []
    train_label_list = []
    with open("./train_sgei.txt", "r") as txt:
        for line in txt.readlines():
            line.strip()
            train_data_list.append([line.split(",")[0],line.split(",")[1]])
            train_label_list.append(int(line.split(",")[2]))
    print(len(train_label_list))

    train_dataset = GaitDataset(
        img_name_list=train_data_list,
        label_list=train_label_list,
        transform=transforms.Compose([
            transforms.Resize(128,2),
            transforms.ToTensor()])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10)

    # -----------------------set valid data-----------------------
    valid_data_list = []
    valid_label_list = []
    with open("./valid_sgei.txt", "r") as txt:
        for line in txt.readlines():
            line.strip()
            valid_data_list.append([line.split(",")[0],line.split(",")[1]])
            valid_label_list.append(int(line.split(",")[2]))

    valid_dataset = GaitDataset(
        img_name_list=valid_data_list,
        label_list=valid_label_list,
        transform=transforms.Compose([
            transforms.Resize(128,2),
            transforms.ToTensor()])
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=10)


   # -----------------------train and valid -----------------------
    print('===> Try resume from checkpoint')
    if os.path.isdir(save_path):
        try:
            checkpoint = torch.load('%s/autoencoder.t7'% save_path)
            model.load_state_dict(checkpoint['state']) # 從字典中依次讀取
            startEpoch = checkpoint['epoch']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            print('Can\'t found autoencoder.t7')
    else:
        startEpoch = 0
        print('===> Start from scratch')

    # -----------------------train and valid -----------------------
    best_test_acc = 0
    best_test_loss = 100
    for epoch in range(startEpoch, endEpoch):
        train_loss = train_model(model,train_dataloader,optimizer,loss_function)
        valid_loss ,valid_acc = valid_model(model,valid_dataloader,loss_function)
        print("[epoch-{}/{}, train loss-{}, valid loss-{}, valid acc-{:.2f}%".format(epoch+1,endEpoch,train_loss,valid_loss,valid_acc*100))
        if valid_acc >= best_test_acc:
            best_test_acc = valid_acc
            # 保存模型示例代碼
            print('===> Saving models...')
            state = {
                'state': model.state_dict(),
                'epoch': epoch+1 # 將epoch一併保存
            }
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            torch.save(state, '{}/epoch-{}_trainLoss-{}_validLoss-{}_validAcc-{:.2f}%.t7'.format(save_path, epoch+1, train_loss, valid_loss, valid_acc*100))

        scheduler.step(valid_acc)






