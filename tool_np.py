
import paddle
import numpy as np

def get_iou(boxes1, boxes2):
    '''
    :param boxes1: [N, 4], (x1, y1, x2, y2)
    :param boxes2: [M, 4]
    :return: iou, [N, M]
    '''
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    # 将boxes的每个box复制M遍，对应于与boxes2的每个box求iou，相当于将boxes复制M遍
    boxes1 = np.repeat(np.expand_dims(boxes1, axis=1), M, axis=1) # [N, M, 4]
    # boxes2同理
    boxes2 = np.repeat(np.expand_dims(boxes2, axis=0), N, axis=0) # [N, M, 4]
    # 求取boxes1和boxes2的每个box相互之间的交点
    # 左上取最大, 右下取最小
    x1y1 = np.maximum(boxes1[:, :, :2], boxes2[:, :, :2])
    x2y2 = np.minimum(boxes1[:, :, 2:], boxes2[:, :, 2:])
    # 算宽和高，由于可能某两个box没有重叠，此时通过用x2y2-x1y1可能会出现负数，这时需要截取
    wh = np.clip(x2y2-x1y1, a_min=0, a_max=None) # shape：[N, M, 2]
    # 计算交集的面积
    area_inter = wh[:, :, 0] * wh[:, :, 1]  # shape：[N, M]
    # 计算每个box的面积，再求并集的面积
    area_boxes1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * ( boxes1[:, :, 3]-boxes1[:, :, 1])  # shape：[N, M]
    area_boxes2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])  # shape：[N, M]
    area_union = area_boxes1 + area_boxes2 - area_inter
    return area_inter/area_union

def nms(boxes, scores, thresh=0.5):
    '''
    :param boxes: [N, 4], (x1, y1, x2, y2)
    :param scores: [N]
    :param thresh: 按分数排序后，由分数高的开始保留，而与这些框的iou > thresh的框将被舍弃
    :return:
    '''

    index = np.argsort(scores)[::-1] #排序，分数由大到小的序号
    # 遍历index
    keep = [] #要保存的box的序号
    while index.shape[0] > 0:
        idx = index[0]
        keep.append(idx) #当前最高分的一定保留
        if index.shape[0] == 1: #如果只剩下这一个，就不用在继续了，否则要计算iou进行筛选
            break
        # 当前序号的box和后续序号的box的iou,返回结果为[k]
        iou = get_iou(np.expand_dims(boxes[index[0], :], axis=0), boxes[index[1:], :]).squeeze(0)
        new_index = iou < thresh #iou<thresh的保留，其余舍弃
        index = index[1:][new_index] #更新index.index[0]代表当前序号，所以要从index[1:]选取iou满足要求的来更新
    return keep


def main():
    boxes = [[150.78372, 225.33029, 287.48123, 420.98615], [143.59653, 216.02008, 283.73373, 414.75256],
        [637.2773,   36.99518, 788.42596, 279.09363], [631.57214,   31.707092, 774.79175,  272.07147 ],
        [285.6257,   71.52667, 457.4387,  370.05774], [613.1212,   -10.935471, 771.29944,  259.64886 ],
        [626.7298,     -7.7216797, 788.17084,   249.33057  ], [308.02524, 117.75873, 451.37473, 369.77448],
        [148.34291, 227.28085, 290.04865, 447.86176], [160.35065, 227.9816,  290.08368, 448.8723 ],
        [184.75714, 203.83899, 327.9107,  441.81476], [142.44617, 208.67328, 284.47797, 430.06445]]

    scores = [0.5508736, 0.6155565, 0.958624, 0.92724186, 0.58953863, 0.85838556, 0.6919359, 0.50508595,
        0.7110466, 0.6979322, 0.53987175, 0.59540105]

    boxes = np.array(boxes)
    scores = np.array(scores)
    keep = nms(boxes, scores, 0.5)
    print(keep)
if __name__ == '__main__':
    main()