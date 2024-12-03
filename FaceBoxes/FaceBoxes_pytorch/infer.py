
from networks import FaceBoxes
from encoderl import DataEncoder

import torch
import numpy as np
import torch.nn.functional as F
import cv2
import time
from matplotlib import pyplot as plt

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

def testIm(file):
    im = cv2.imread(file)
    if im is None:
        print("can not open image:", file)
        return
    h,w,_ = im.shape
    boxes, probs = detect(im)

    if probs[0] == 0:
        print('There is no face in the image')
        exit()
    for i, (box) in enumerate(boxes):
        print('i=', i, 'box=', box, 'prob=', probs[i])
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        cv2.rectangle(im,(x1,y1+4),(x2,y2),(255,0,0),2)
        cv2.putText(im, str(probs[i]), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255))
    cv2.imwrite('./result.jpg', im)
    # plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # plt.show()


if __name__ == '__main__':
    net = FaceBoxes()
    net.load_state_dict(torch.load('log/标准卷积/faceboxes.pt', map_location=lambda storage, loc:storage), strict=False)
    # torch.save(net, 'log/facebox_model.pth')
    net.eval()
    data_encoder = DataEncoder()

    # given image path, predict and show
    # root_path = "FDDB/originalPics/"
    # picture = '2002/07/26/big/img_837.jpg'
    # testIm(root_path + picture)
    testIm('vlcsnap-2024-05-22-13h18m21s762.png')