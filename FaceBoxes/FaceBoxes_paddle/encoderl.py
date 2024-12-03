
import paddle
import math
import itertools
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tool_np import get_iou, nms

class DataEncoder:
    def __init__(self):
        self.image_size = 1024
        self.scales = [[32, 64, 128], [256], [512]] # Anchor的尺寸
        self.interval = [32, 64, 128] # Anchor的间隔
        self.feature_maps = [[math.ceil(self.image_size / interval), math.ceil(self.image_size / interval)]
                             for interval in self.interval]  # x轴和y轴的boxes数量，由于都是正方形，所以x轴和y轴的boxes数量相同
        self.init_anchors()

    def init_anchors(self):
        anchors = []
        for k, feature_map in enumerate(self.feature_maps):
            scales = self.scales[k]
            for i, j in itertools.product(range(feature_map[0]), range(feature_map[1])):
                for scale in scales: # Anchor稠密化
                    s_kx = scale / self.image_size
                    s_ky = scale / self.image_size
                    # 第一个尺寸的Anchor（scale=32）数量会变成4*4倍，第二个尺寸的Anchor（scale=64）数量会变成2*2倍，
                    # 第三、四、五个尺寸Anchor不变
                    if scale == 32:
                        dense_cx = [x * (self.interval[k] / self.image_size)
                                    for x in [j + 0, j + 0.25, j + 0.5, j + 0.75]] # x为中心点偏移，数量4倍
                        dense_cy = [y * (self.interval[k] / self.image_size)
                                    for y in [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in itertools.product(dense_cy, dense_cx):
                            anchors.append([cx, cy, s_kx, s_ky])
                    elif scale == 64:
                        dense_cx = [x * (self.interval[k] / self.image_size) for x in [j + 0, j + 0.5]] # 2倍
                        dense_cy = [y * (self.interval[k] / self.image_size) for y in [i + 0, i + 0.5]]
                        for cy, cx in itertools.product(dense_cy, dense_cx):
                            anchors.append([cx, cy, s_kx, s_ky])
                    else:
                        cx = (j + 0.5) * (self.interval[k] / self.image_size)
                        cy = (i + 0.5) * (self.interval[k] / self.image_size)
                        anchors.append([cx, cy, s_kx, s_ky])
        # self.default_boxes = paddle.to_tensor(anchors)
        self.default_boxes = np.array(anchors)

    def encode(self, boxes, threshold=0.5):
        '''
        boxes:[num_obj, 4]
        default_box： [num_anhcors, 4] (x1,y1,w,h)
        return:boxes: [num_anhcors, 4]
        classes:class label [num_anhcors,]，都是1
        '''
        default_boxes = self.default_boxes  # [num_anhcors,4]
        tg_obj = np.zeros(shape=(default_boxes.shape[0], ))

        # 将default_box (cx,cy,w,h) => (x1,y1,x2,y2)
        x1y1 = default_boxes[:, :2] - default_boxes[:, 2:] / 2
        x2y2 = default_boxes[:, :2] + default_boxes[:, 2:] / 2
        iou = get_iou(boxes, np.concatenate([x1y1, x2y2], 1))  # iou_size = (num_obj, num_anhcors)

        # 每个default_box对应于一张脸
        face_iou = np.max(iou, axis=0)
        face_index = np.argmax(iou, axis=0)
        tg_obj[face_iou > threshold] = 1 #将iou>0.5设置为1

        # 每张脸对应于一个default_box（iou最大）
        box_index = np.argmax(iou, axis=1)
        num_obj = box_index.shape[0]

        for i in range(num_obj):
            if tg_obj[box_index[i]] == 0: #face_iou<=0.5的如果有张脸与之对应，也设为1
                face_index[box_index[i]] = i
                tg_obj[box_index[i]] = 1

        boxes = boxes[face_index] #[num_anhcors, 4]

        variances = 0.1
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]
        cxcy /= default_boxes[:, 2:]*variances
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]
        wh = np.log(wh)/variances
        tg_boxes = np.concatenate([cxcy, wh], axis=1)  # [num_anhcors,4]

        return tg_obj, tg_boxes

    def decode(self, conf, loc):
        '''
        將预测出的 loc/conf转换成真实的人脸框
        loc [21842,4]
        conf [num_anhcors,2]
        '''

        variances = 0.1
        cxcy = loc[:, :2] * variances * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = np.exp(loc[:, 2:] * variances) * self.default_boxes[:, 2:]

        boxes = np.concatenate([cxcy - wh / 2, cxcy + wh / 2], 1)  # [num_anhcors,4]
       

        # conf[:,0] means no face
        # conf[:,1] means face
        max_conf = np.max(conf, axis=1)  # [num_anhcors,1]
        labels = np.argmax(conf, axis=1)

        ids = (labels ==1)
        boxes = boxes[ids]
        max_conf = max_conf[ids]

        keep = nms(boxes, max_conf)
        boxes = boxes[keep]

        max_conf = max_conf[keep]
        # return boxes, max_conf
        mask = max_conf > 0.6  # 筛选大于0.9的框
        return boxes[mask],  max_conf[mask]

    def draw_default_boxes(self):
        img = np.ones((1024, 1024, 3)) * 255.
        boxes = self.default_boxes.squeeze(0).detach().numpy()
        for i, (box) in enumerate(boxes):
            cx = int(box[0] * 1024)
            w = int(box[2] * 1024)
            cy = int(box[1] * 1024)
            h = int(box[3] * 1024)
            x1 = cx - w // 2
            x2 = cx + w // 2
            y1 = cy - h // 2
            y2 = cy + h // 2
            if w <= 32:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            elif w <= 64:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 1)
            elif w <= 128:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)
            elif w <= 256:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            elif w <= 512:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # cv2.imwrite('./default_boxes.png', img)
        img = img[:, :, ::-1]  # BGR2RGB
        plt.imshow(img)
        plt.show()