# coding:utf-8

import os
import paddle
from paddle.io import Dataset
import cv2
import numpy as np
from encoderl import DataEncoder

class FDDB(Dataset):
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
        img = cv2.resize(img, (self.image_size,self.image_size))/255.0

        numboxes = int(file[1])
        boxes = []
        # labels = []
        for i in range(numboxes):
            x1 = (float(file[2 + 5 * i]) + pad2[0])/maxsize
            y1 = (float(file[2 + 5 * i + 1]) + pad1[0])/maxsize
            x2 = (float(file[2 + 5 * i + 2]) + pad2[0])/maxsize
            y2 = (float(file[2 + 5 * i  + 3]) + pad1[0])/maxsize
            boxes.append([x1, y1, x2, y2])
            # c = int(file[2 + 5 * i + 4]) #其实都是1
            # labels.append([c])
        boxes = np.array(boxes)
        # labels = paddle.to_tensor(labels).astype("int64")
        # 解析box
        tg_obj, tg_boxes = self.data_encoder.encode(boxes)
        img = paddle.to_tensor(img).astype('float32')
        tg_obj = paddle.to_tensor(tg_obj)
        tg_boxes = paddle.to_tensor(tg_boxes)
        return img, tg_obj, tg_boxes

    def __len__(self):
        return self.num_file
