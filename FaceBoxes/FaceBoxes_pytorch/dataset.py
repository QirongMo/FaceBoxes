# coding:utf-8

import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from encoderl import DataEncoder

class MyDataset(data.Dataset):
    def __init__(self, img_dir, file):
        self.image_size = 1024
        self.img_dir = img_dir
        self.img_idx = 0
        self.image_name = [] # list: image name
        self.boxes = []
        self.labels = []
        self.data_encoder = DataEncoder()
        f = open(file,'r')
        self.allfile = f.readlines()
        self.num_file = len(self.allfile) # num of images

    def __getitem__(self, idx):
        file = self.allfile[idx].strip().split()
        file_name = file[0]
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        numboxes = int(file[1])
        boxes = []
        labels = []
        for i in range(numboxes):
            x1 = float(file[2 + 5 * i])
            y1 = float(file[2 + 5 * i + 1])
            x2 = float(file[2 + 5 * i + 2])
            y2 = float(file[2 + 5 * i + 3])
            c = int(file[2 + 5 * i + 4])
            boxes.append([x1, y1, x2, y2])
            labels.append(c)
        boxes = torch.Tensor(boxes)
        labels = torch.Tensor(labels).long()
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)
        img = cv2.resize(img,(self.image_size,self.image_size))
        # 解析box
        loc_target,conf_target = self.data_encoder.encode(boxes,labels)
        img = transforms.ToTensor()(img)
        return img,loc_target,conf_target

    def __len__(self):
        return self.num_file
