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
    img_size = 1024
    input_img = cv2.resize(input_img,(img_size, img_size))/255.0
    input_img = paddle.to_tensor(input_img.transpose((2,0,1))).astype('float32').unsqueeze(0)

    loc, conf = model(input_img)
    conf = paddle.nn.Softmax(axis=1)(paddle.squeeze(conf, axis=0))
    loc = paddle.squeeze(loc, axis=0)
    boxes, probs = data_encoder.decode(conf.numpy(), loc.numpy())

    if probs.shape[0] != 0:
        boxes[:,1::2] = boxes[:,1::2] * maxsize - pad1[0]
        boxes[:, ::2] = boxes[:, ::2] * maxsize - pad2[0]
    return boxes,  probs

def testIm():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        boxes, probs = detect(frame)

        if probs.shape[0] == 0:
            print('There is no face in the image')
        else:
            for i, (box) in enumerate(boxes):
                print('i=', i, 'box=', box,  'probs=', probs[i])
                x1 = int(box[0])
                x2 = int(box[2])
                y1 = int(box[1])
                y2 = int(box[3])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
                cv2.putText(frame, str(probs[i]), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,255), 1)

        cv2.imshow('photo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    model = FaceBoxes()
    model_dict = paddle.load('./log/FaceBoxes.pdparams')
    model.set_state_dict(model_dict)
    model.eval()
    data_encoder = DataEncoder()
    testIm()