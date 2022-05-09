from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2


class RescaleT(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		left, right, label = sample['image_left'], sample['image_right'], sample["label"]
		h, w = left.shape[:2]

		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)
		# img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		# lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		left = cv2.resize(left, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
		right = cv2.resize(right, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
		# img = cv2.resize(image,(self.output_size,self.output_size),interpolation = cv2.INTER_NEAREST )
		# lbl = cv2.resize(label,(self.output_size,self.output_size),interpolation = cv2.INTER_NEAREST )

		return {'image_left': left, 'image_right': right, 'label': label}

# class ToTensofLab(object):
# 	def __init__(self,flag=0):
# 		self.flag = flag
#
# 	def __call__(self, sample):
# 		left, right ,label = sample["image_left"], sample["image_right"], sample["label"]
# 		# tmpleft = np.zeros((left.shape[0], left.shape[1], 3))
# 		# tmpright = np.zeros((right.shape[0], right.shape[1], 3))
# 		left = left / 255
# 		right = right / 255
# 		#
# 		# if len(left.shape) == 2:
# 		# 	# tmpleft[:, :, 0] = (left[:, :] - 0.485) / 0.229
# 		# 	tmpleft[:, :, 0] = left
# 		# else:
# 		# 	tmpleft[:, :, 0] = (left[:, :, 0] - 0.485) / 0.229
# 		# 	tmpleft[:, :, 1] = (left[:, :, 1] - 0.456) / 0.224
# 		# 	tmpleft[:, :, 2] = (left[:, :, 2] - 0.406) / 0.225
# 		#
# 		# # cv2.imshow("xx",left)
# 		# # cv2.waitKey(0)
# 		# if len(right.shape) == 2:
# 		# 	# 为什么变成三通道？ 因为与训练模型大多数都是三通道
# 		# 	# tmpright[:, :, 0] = (right[:, :] - 0.485) / 0.229
# 		# 	tmpright[: ,:, 0] = right
# 		# else:
# 		# 	tmpright[:, :, 0] = (right[:, :, 0] - 0.485) / 0.229
# 		# 	tmpright[:, :, 1] = (right[:, :, 1] - 0.456) / 0.224
# 		# 	tmpright[:, :, 2] = (right[:, :, 2] - 0.406) / 0.225
#
# 		# change the r,g,b to b,r,g from [0,255] to [0,1]
# 		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
# 		left = left.transpose((2, 0, 1))
# 		right = right.transpose((2, 0, 1))
# 		return {"image_left":torch.from_numpy(left),"image_right":torch.from_numpy(right),"label":label}

class GaitDataset(Dataset):
	def __init__(self,img_name_list,label_list,transform=None):
		self.image_name_list = img_name_list
		self.label_list = label_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
		left = Image.open(self.image_name_list[idx][0]).convert("L")
		right = Image.open(self.image_name_list[idx][1]).convert("L")
		label = self.label_list[idx]

		if self.transform:
			left = self.transform(left)
			right = self.transform(right)


		return left,right,torch.from_numpy(np.array([label],dtype=np.float32))