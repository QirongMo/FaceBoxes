
from networks import FaceBoxes
from encoderl import DataEncoder

import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time
from matplotlib import pyplot as plt
from mAP import compute_ap

net_time = []
total_time = []
def detect(im):
    h, w, _ = im.shape
    maxsize = max(h,w)
    pad1 = ((maxsize-h)//2, (maxsize-h)-(maxsize-h)//2)
    pad2 = ((maxsize - w) // 2, (maxsize-w) - (maxsize - w) // 2)
    pad = (pad1, pad2, (0, 0))
    input_img = np.pad(im, pad, 'constant', constant_values=128)
    input_img = cv2.resize(input_img,(1024,1024))/255.
    im_tensor = torch.from_numpy(input_img.transpose((2,0,1))).float()
    start_time = time.time()
    conf, loc= net(im_tensor.unsqueeze(0))
    end_time1 = time.time()
    boxes, labels, probs = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0), dim=1))
    end_time2 = time.time()
    net_time.append(end_time1-start_time)
    total_time.append(end_time2-start_time)

    if probs[0] != 0:
        boxes = boxes.numpy()
        probs = probs.detach().numpy()
        boxes[:,1] = boxes[:,1]*maxsize-pad1[0]
        boxes[:,3] = boxes[:,3]*maxsize-pad1[0]
        boxes[:,0] = boxes[:,0]*maxsize-pad2[0]
        boxes[:,2] = boxes[:,2]*maxsize-pad2[0]
    return boxes, probs

APs = []
def get_list():
    f = open('FDDB/FDDB.txt', 'r')
    allfiles = f.readlines()
    root_path = "FDDB/originalPics/"
    num = 1
    for file in allfiles:
        file=file.strip().split()
        filename = file[0]
        img = cv2.imread(root_path+filename)
        pred_boxes, probs = detect(img)
        if probs[0] == 0:
            pred_boxes = np.array([[0,0,0,0]])
            probs = np.array([0])
        pred_class_ids = np.ones((pred_boxes.shape[0]))
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
        gt_boxes = np.array(boxes)
        gt_class_ids = np.array(labels)
        mAP = compute_ap(gt_boxes, gt_class_ids,
                         pred_boxes, pred_class_ids, probs)
        APs.append(mAP[0])
        print(num)
        num += 1
    print(np.mean(APs))

if __name__ == '__main__':
    net = FaceBoxes()
    net.load_state_dict(torch.load('log/60.pt', map_location=lambda storage, loc:storage), strict=False)
    net.eval()
    data_encoder = DataEncoder()
    get_list()