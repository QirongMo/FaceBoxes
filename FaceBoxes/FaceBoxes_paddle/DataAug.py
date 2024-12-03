
import numpy as np
import random

def draw(img, boxes):
    img_copy = img.copy()
    for i, (box) in enumerate(boxes):
        x1 = int(box[0])
        x2 = int(box[2])
        y1 = int(box[1])
        y2 = int(box[3])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('img', img_copy)
    cv2.waitKey(1000)

class DataAug:
    def __init__(self):
        super(DataAug, self).__init__()

    def fliplr(self, img, boxes, threshold=0.1):
        # 左右翻转
        if random.random() < threshold: #随机数判断
            print('水平翻转')
            img = np.fliplr(img)
            boxes[:, ::2] = img.shape[1] - boxes[:, 2::-2] # 只变x，y不变
        return img, boxes

    def flipud(self, img, boxes, threshold=0.5):
        # 上下翻转
        if random.random() > threshold: #随机数判断
            print('垂直翻转')
            img = np.flipud(img)
            boxes[:, 1::2] = img.shape[0] - boxes[:, 3::-2] # 只变y，x不变
        return img, boxes

    def rot90(self, img, boxes, threshold=0.1):
        # 翻转90度
        if random.random() > threshold:  # 随机数判断
            # print('旋转90度')
            h, w = img.shape[:2]
            img  = np.rot90(img)
            x1 = boxes[:, 1] # x1就是原来的y1
            y1 = w - boxes[:, 2] # y1
            x2 = boxes[:, 3] # x1就是原来的y2
            y2 = w - boxes[:, 0]

            boxes[:, 0] = x1
            boxes[:, 1] = y1
            boxes[:, 2] = x2
            boxes[:, 3] = y2

        return img, boxes
    
    def colorbright(self, img):
        alphas=[0.3, 0.5, 1.2, 1.6]
        beta=10
        alpha = random.choice(alphas)
        return np.uint8(np.clip((alpha * img + beta), 0, 255))
    
    def crop(self, img, boxes):
        x1 = int(min(boxes[:, 0]))
        y1 = int(min(boxes[:, 1]))
        x2 = int(max(boxes[:, 2]))
        y2 = int(max(boxes[:, 3]))

        # 可以裁剪的范围（保留所有的box）
        h, w = img.shape[:2]

        x1 = max(x1, 0)
        x2 = max(w-x2, 0)
        y1 = max(y1, 0)
        y2 = max(h-y2, 0)

        x1 = random.randint(0, x1)
        x2 = random.randint(0, x2)
        x2 = w-x2
        y1 = random.randint(0, y1)
        y2 = random.randint(0, y2)
        y2 = h-y2
        img = img[y1:y2+1, x1:x2+1, :]
        boxes[:, 0::2] -= x1 
        boxes[:, 1::2] -= y1

        return img, boxes


if __name__ == '__main__':

    dataaug = DataAug()

    import cv2
    # img = cv2.imread('img_591.jpg')
    # boxes = np.array([[184, 38, 355, 285]])
    img = cv2.imread('img_422.jpg')
    boxes = np.array([[272, 31, 415, 240],[89, 9, 143, 81]])
    draw(img, boxes)
    img = dataaug.colorbright(img)
    draw(img, boxes)
