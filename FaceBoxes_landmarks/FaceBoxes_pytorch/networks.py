
import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d,BatchNorm2d,Sequential,ReLU


class CReLu(nn.Module):
    def __init__(self):
        super(CReLu, self).__init__()
        self.relu = ReLU()
    def forward(self, inputs):
        x = torch.cat([inputs, -inputs], axis=1)
        x = self.relu(x)
        return x

class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=True):
        super(conv_bn_relu, self).__init__()
        # self.conv = Sequential(
        #     Conv2d(in_channels=in_channels, out_channels=in_channels,
        #            kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
        #     Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        # )
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm2d(num_features=out_channels)
        if act:
            self.relu = ReLU()
        else:
            self.relu = None

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class RDCL(nn.Module):
    def __init__(self):
        super(RDCL, self).__init__()
        self.crelu = CReLu()
        self.conv1 = conv_bn_relu(in_channels=3, out_channels=24, kernel_size=7, stride=4, padding=3, act=False)
        self.pool1 = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_bn_relu(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2, act=False)
        self.pool2 = MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.crelu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.crelu(x)
        x = self.pool2(x)
        return x

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1 = conv_bn_relu(in_channels=128, out_channels=32, kernel_size=1)

        self.branch2 = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_bn_relu(in_channels=128, out_channels=32, kernel_size=1)
        )
        self.branch3 = Sequential(
            conv_bn_relu(in_channels=128, out_channels=24, kernel_size=1),
            conv_bn_relu(in_channels=24, out_channels=32, kernel_size=3, padding=1)
        )
        self.branch4 = Sequential(
            conv_bn_relu(in_channels=128, out_channels=24, kernel_size=1),
            conv_bn_relu(in_channels=24, out_channels=32, kernel_size=3, padding=1),
            conv_bn_relu(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        )

    def forward(self, inputs):
        x1 = self.branch1(inputs)
        x2 = self.branch2(inputs)
        x3 = self.branch3(inputs)
        x4 = self.branch4(inputs)
        return torch.cat([x1, x2, x3, x4], axis=1)

class MSCL(nn.Module):
    def __init__(self):
        super(MSCL, self).__init__()
        self.Inception1 = Inception()
        self.Inception2 = Inception()
        self.Inception3 = Inception()
        self.conv3_1 = conv_bn_relu(in_channels=128, out_channels=128, kernel_size=1, stride=1)
        self.conv3_2 = conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = conv_bn_relu(in_channels=256, out_channels=128, kernel_size=1, stride=1)
        self.conv4_2 = conv_bn_relu(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
    def forward(self, inputs):
        x = self.Inception1(inputs)
        x = self.Inception2(x)
        x = self.Inception3(x)
        x1 = x
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x2 = x
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        return [x1,x2,x]

class FaceBoxes(nn.Module):
    def __init__(self):
        super(FaceBoxes, self).__init__()
        self.rdcl = RDCL()
        self.mscl = MSCL()

        self.regression1 = Conv2d(in_channels=128, out_channels=21*4, kernel_size=3, stride=1, padding=1)
        self.classifier1 = Conv2d(in_channels=128, out_channels=21*2, kernel_size=3, stride=1, padding=1)
        self.landmark1 = Conv2d(in_channels=128, out_channels=21*10, kernel_size=3, stride=1, padding=1)

        self.regression2 = Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.classifier2 = Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.landmark2 = Conv2d(in_channels=256, out_channels=10, kernel_size=3, stride=1, padding=1)

        self.regression3 = Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.classifier3 = Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.landmark3 = Conv2d(in_channels=256, out_channels=10, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        x = self.rdcl(inputs)
        x1,x2,x = self.mscl(x)
        classifier1 = self.classifier1(x1)
        classifier1 = classifier1.permute(0, 2, 3, 1).contiguous()
        classifier1 = classifier1.view(classifier1.size(0), -1, 2)

        regression1 = self.regression1(x1)
        regression1 = regression1.permute(0, 2, 3, 1).contiguous()
        regression1 = regression1.view(regression1.size(0), -1, 4)

        landmark1 = self.landmark1(x1)
        landmark1 = landmark1.permute(0, 2, 3, 1).contiguous()
        landmark1 = landmark1.view(landmark1.size(0), -1, 10)

        classifier2 = self.classifier2(x2)
        classifier2 = classifier2.permute(0, 2, 3, 1).contiguous()
        classifier2 = classifier2.view(classifier2.size(0), -1, 2)

        regression2 = self.regression2(x2)
        regression2 = regression2.permute(0, 2, 3, 1).contiguous()
        regression2 = regression2.view(regression2.size(0), -1, 4)

        landmark2 = self.landmark2(x2)
        landmark2 = landmark2.permute(0, 2, 3, 1).contiguous()
        landmark2 = landmark2.view(landmark2.size(0), -1, 10)

        classifier3 = self.classifier3(x)
        classifier3 = classifier3.permute(0, 2, 3, 1).contiguous()
        classifier3 = classifier3.view(classifier3.size(0), -1, 2)

        regression3 = self.regression3(x)
        regression3 = regression3.permute(0, 2, 3, 1).contiguous()
        regression3 = regression3.view(regression3.size(0), -1, 4)

        landmark3 = self.landmark3(x)
        landmark3 = landmark3.permute(0, 2, 3, 1).contiguous()
        landmark3 = landmark3.view(landmark3.size(0), -1, 10)

        classifier = torch.cat([classifier1, classifier2, classifier3], axis=1)

        regression = torch.cat([regression1, regression2, regression3], axis=1)

        landmark = torch.cat([landmark1, landmark2, landmark3], axis=1)

        return [classifier, regression, landmark]
def main():
    import torch
    import numpy as np
    x = np.random.randn(1,3,1024,1024).astype(np.float32)
    x = torch.from_numpy(x)
    model = FaceBoxes()
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)


if __name__ == '__main__':
    main()