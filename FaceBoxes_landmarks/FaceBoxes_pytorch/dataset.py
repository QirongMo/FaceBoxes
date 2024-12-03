# coding:utf-8

import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from encoderl import DataEncoder

class MyDataset(data.Dataset):
    def __init__(self, img_dir, file):
        self.img_dir = img_dir
        self.data_encoder = DataEncoder()
        f = open(file,'r')
        self.allfile = f.readlines()
        self.num_file = len(self.allfile) # num of images

    def __getitem__(self, idx):
        file = self.allfile[idx].strip().split()
        file_name = file[0]
        img_path = os.path.join(self.img_dir, file_name)
        img = cv2.imread(img_path)
        numboxes = int(file[1])
        boxes = []
        landmarks = []
        labels = []
        for i in range(numboxes):
            x1 = float(file[2 + 14 * i])
            y1 = float(file[2 + 14 * i + 1])
            x2 = float(file[2 + 14 * i + 2])
            y2 = float(file[2 + 14 * i + 3])
            boxes.append([x1, y1, x2, y2])
            landmark = []
            for j in range(4,14):
                landmark.append(float(file[2 + 14 * i + j]))
            landmarks.append(landmark)
            labels.append(1)

        boxes = torch.Tensor(boxes)
        landmarks = torch.Tensor(landmarks)
        labels = torch.Tensor(labels).long()

        # 解析box
        loc_target,conf_target,landmarks_target = self.data_encoder.encode(boxes,landmarks, labels)
        img = transforms.ToTensor()(img)
        return img,loc_target,conf_target, landmarks_target

    def __len__(self):
        return self.num_file
