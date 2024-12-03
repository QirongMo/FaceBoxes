# coding:utf-8

import os
import paddle
from paddle.io import Dataset
import cv2
import numpy as np
from encoderl import DataEncoder

class WIDER_Face(Dataset):
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
        maxsize = max(h, w)
        pad1 = ((maxsize-h)//2, (maxsize-h)-(maxsize-h)//2)
        pad2 = ((maxsize - w) // 2, (maxsize-w) - (maxsize - w) // 2)
        pad = (pad1, pad2, (0, 0))
        img = np.pad(img, pad, 'constant', constant_values=128)
        img = cv2.resize(img, (self.image_size,self.image_size))

        numboxes = int(file[1])
        boxes = []
        landmarks = []
        labels = []
        for i in range(numboxes):
            x1 = float(file[2+14*i])
            y1 = float(file[2+14*i + 1])
            x2 = float(file[2+14*i + 2])
            y2 = float(file[2+14*i + 3])
            boxes.append([x1, y1, x2, y2])
            landmark = []
            for j in range(4, 14):
                landmark.append(float(file[2 + 14 * i + j]))
            landmarks.append(landmark)
            labels.append([1])
        boxes = np.array(boxes)
        landmarks = np.array(landmarks)
        labels = np.array(labels)
        # 解析box
        tg_obj, tg_boxes, tg_landmarks = self.data_encoder.encode(boxes, landmarks, labels)
        img = paddle.to_tensor(img).astype('float32')
        return img, tg_obj, tg_boxes, tg_landmarks, boxes.shape[0]

    def __len__(self):
        return self.num_file

from paddle.io import DataLoader
train_dataset = WIDER_Face(img_dir='./dataset/mx/img', file='./dataset/mx/mxface.txt')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
for i,(img, tg_obj, tg_boxes, tg_landmarks, boxes_shape) in enumerate(train_loader):
    # if boxes_shape == 1:
    #     continue
    # break
    1
