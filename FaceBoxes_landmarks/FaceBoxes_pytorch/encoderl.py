
import torch
import math
import itertools
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
        self.default_boxes = torch.Tensor(anchors)

    def iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].

        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].

        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(  # left top
            box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(  # right bottom
            box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def nms(self, bboxes, scores, threshold=0.5):
        '''
        bboxes(tensor) [N,4]
        scores(tensor) [N,]
        '''
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= threshold).nonzero(as_tuple=False).squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        return torch.LongTensor(keep)

    def encode(self, boxes, landmarks, classes, threshold=0.35):
        '''
        boxes:[num_obj, 4]
        default_box (x1,y1,w,h)
        return:boxes: (tensor) [num_obj,21824,4]
        classes:class label [obj,]
        '''
        boxes_org = boxes

        default_boxes = self.default_boxes  # [21824,4]

        iou = self.iou(
            boxes,
            torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,  # (x0,y0,w,h) => (x1,y1,x2,y2)
                       default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1))  # iou_size = (num_obj, 21824)

        # find max iou of each face in default_box
        max_iou, max_iou_index = iou.max(1)
        # find max iou of each default_box in faces
        iou, max_index = iou.max(0)

        # ensure every face have a default box,I think is no use.
        # max_index[max_iou_index] = torch.LongTensor(range(num_obj))

        boxes = boxes[max_index]  # [21824,4] 是图像label, use conf to control is or not.
        landmarks = landmarks[max_index]

        variances = [0.1, 0.2]
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [21824,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [21824,2]  为什么会出现0宽度？？
        wh = torch.log(wh) / variances[1]  # Variable

        landmarks[:,::2] -= default_boxes[:, 0].unsqueeze(1).expand_as(landmarks[:,::2]) # [21824, 5]
        landmarks[:,::2] /=  variances[0] * default_boxes[:, 2].unsqueeze(1).expand_as(landmarks[:,::2])

        landmarks[:,1::2] -=  default_boxes[:, 1].unsqueeze(1).expand_as(landmarks[:,1::2])
        landmarks[:,1::2] /= variances[0] * default_boxes[:, 3].unsqueeze(1).expand_as(landmarks[:,1::2])

        inf_flag = wh.abs() > 10000
        if (inf_flag.sum().item() is not 0):
            print('inf_flag has true', wh, boxes)
            print('org_boxes', boxes_org)
            print('max_iou', max_iou, 'max_iou_index', max_iou_index)
            raise ['inf error']

        loc = torch.cat([cxcy, wh], 1)  # [21824,4]

        conf = classes[max_index]  # 其实都是1 [21824,]
        conf[iou < threshold] = 0  # iou小的设为背景
        return loc, conf, landmarks

    def decode(self, loc, conf, landmarks):
        '''
        將预测出的 loc/conf转换成真实的人脸框
        loc [21842,4]
        conf [21824,2]
        '''
        variances = [0.1, 0.2]
        cxcy = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + self.default_boxes[:, :2]
        wh = torch.exp(loc[:, 2:] * variances[1]) * self.default_boxes[:, 2:]
        boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)  # [21824,4]

        
        landmarks[:,::2] *=  variances[0] * self.default_boxes[:, 2].unsqueeze(1).expand_as(landmarks[:,::2])
        landmarks[:,::2] += self.default_boxes[:, 0].unsqueeze(1).expand_as(landmarks[:,::2]) # [21824, 5]
        landmarks[:,1::2] *=  variances[0] * self.default_boxes[:, 3].unsqueeze(1).expand_as(landmarks[:,1::2])
        landmarks[:,1::2] += self.default_boxes[:, 1].unsqueeze(1).expand_as(landmarks[:,1::2]) # [21824, 5]

        # conf[:,0] means no face
        # conf[:,1] means face

        max_conf, labels = conf.max(1)  # [21842,1]

        num_points = landmarks.shape[1]//2
        if labels.long().sum().item() is 0:
            # no face in image
            return torch.empty((0, 4)), torch.empty((0, 1)), torch.empty((0, 4)), torch.empty((0, num_points*2)) 

        ids = labels.nonzero(as_tuple=False).squeeze(1)

        keep = self.nms(boxes[ids], max_conf[ids])

        return boxes[ids][keep], labels[ids][keep], max_conf[ids][keep], landmarks[ids][keep]

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
        img = img[:,:,::-1] # BGR2RGB
        plt.imshow(img)
        plt.show()

def main():
    encoder = DataEncoder()
    boxes = torch.tensor([[0.4341981075697211, 0.18634482071713146, 0.6547481075697211, 0.4047846613545817]])
    landmarks = torch.tensor([[0.505511952191235, 0.2636867529880478, 0.596722609561753, 0.26070019920318727, 0.5525667330677291,
    	0.3180229083665338,	0.5169721115537849, 0.35262948207171313, 0.5867509960159362, 0.3480179282868526]])
    classes = torch.tensor([[1]]).long()
    encoder.encode(boxes, landmarks, classes)

if __name__ == '__main__':
    main()