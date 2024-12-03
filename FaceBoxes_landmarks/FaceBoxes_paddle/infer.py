from model import FaceBoxes
from encoderl import DataEncoder

import paddle
import numpy as np
import cv2
import matplotlib.pyplot as plt

def detect(img):
    h, w = img.shape[:2]
    maxsize = max(h, w)
    pad1 = ((maxsize - h) // 2, (maxsize - h) - (maxsize - h) // 2)
    pad2 = ((maxsize - w) // 2, (maxsize - w) - (maxsize - w) // 2)
    pad = (pad1, pad2, (0, 0))
    input_img = np.pad(img, pad, 'constant', constant_values=128)
    input_img = cv2.resize(input_img,(1024,1024))/255.0
    input_img = paddle.to_tensor(input_img.transpose((2,0,1))).astype('float32').unsqueeze(0)

    loc, landmarks, conf = model(input_img)
    conf = paddle.nn.Softmax(axis=1)(paddle.squeeze(conf, axis=0))
    loc = paddle.squeeze(loc, axis=0)
    landmarks = paddle.squeeze(landmarks, axis=0)
    boxes, landmarks, probs = data_encoder.decode(conf.numpy(), loc.numpy(), landmarks.numpy())

    if probs.shape[0] != 0:
        boxes[:,1::2] = boxes[:,1::2]* maxsize - pad1[0]
        boxes[:, ::2] = boxes[:, ::2] * maxsize - pad2[0]

        landmarks[:, 1::2] = landmarks[:, 1::2] * maxsize - pad1[0]
        landmarks[:, ::2] = landmarks[:, ::2] * maxsize - pad2[0]
    return boxes, landmarks, probs

def testIm(file):
    img = cv2.imread(file)
    if img is None:
        print("can not open image:", file)
        return

    boxes, landmarks, probs = detect(img)

    if probs.shape[0] == 0:
        print('There is no face in the image')
        return
    for i, (box) in enumerate(boxes):
        print('i=', i, 'box=', box,  'probs=', probs[i])
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),6)
        cv2.putText(img, str(probs[i]), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0,0,255), 2)

    for i, (landmark) in enumerate(landmarks):
        x = landmark[::2].tolist()
        y = landmark[1::2].tolist()
        for j in range(len(x)):
            cv2.circle(img, (int(x[j]), int(y[j])), 2, (0,0,255), 8)
    cv2.imshow('photo', img)
    cv2.waitKey(0)
    # cv2.imwrite('picture/1111.jpg', img)
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    model = FaceBoxes()
    model_dict = paddle.load('./log/FaceBoxes.pdparams')
    model.set_state_dict(model_dict)
    model.eval()
    data_encoder = DataEncoder()
    # testIm('./dataset/mx/img/45.jpg')
    testIm('vlcsnap-2024-05-21-15h10m47s816.png')