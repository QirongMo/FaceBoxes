
import paddle
from paddle.nn import Conv2D, MaxPool2D, BatchNorm2D, Sequential, ReLU
from paddle.nn import Layer

class CReLu(Layer):
    def __init__(self):
        super(CReLu, self).__init__()
        self.relu = ReLU()
    def forward(self, inputs):
        x = paddle.concat([inputs, -inputs], axis=1)
        x = self.relu(x)
        return x

class conv_bn_relu(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, act=True):
        super(conv_bn_relu, self).__init__()
        # self.conv = Sequential(Conv2D(in_channels=in_channels, out_channels=in_channels,
        #             kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels),
        #             Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        self.conv = Conv2D(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = BatchNorm2D(num_features=out_channels)
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

class RDCL(Layer):
    def __init__(self):
        super(RDCL, self).__init__()
        self.crelu = CReLu()
        self.conv1 = conv_bn_relu(in_channels=3, out_channels=24, kernel_size=7, stride=4, padding=3, act=False)
        self.pool1 = MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_bn_relu(in_channels=48, out_channels=64, kernel_size=5, stride=2, padding=2, act=False)
        self.pool2 = MaxPool2D(kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.crelu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.crelu(x)
        x = self.pool2(x)
        return x

class Inception(Layer):
    def __init__(self):
        super(Inception, self).__init__()
        self.branch1 = conv_bn_relu(in_channels=128, out_channels=32, kernel_size=1)

        self.branch2 = Sequential(
            MaxPool2D(kernel_size=3, stride=1, padding=1),
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
        return paddle.concat([x1, x2, x3, x4], axis=1)

class MSCL(Layer):
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

class FaceBoxes(Layer):
    def __init__(self):
        super(FaceBoxes, self).__init__()
        self.rdcl = RDCL()
        self.mscl = MSCL()

        self.regression1 = Conv2D(in_channels=128, out_channels=21 * 4, kernel_size=3, stride=1, padding=1)
        self.classifier1 = Conv2D(in_channels=128, out_channels=21 * 2, kernel_size=3, stride=1, padding=1)

        self.regression2 = Conv2D(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.classifier2 = Conv2D(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)

        self.regression3 = Conv2D(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.classifier3 = Conv2D(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        N = inputs.shape[0]
        x = self.rdcl(inputs)
        x1,x2,x = self.mscl(x)
        
        classifier1 = self.classifier1(x1)
        classifier1 = paddle.transpose(classifier1,(0, 2, 3, 1) )
        classifier1 = paddle.reshape(classifier1,(N, -1, 2))

        regression1 = self.regression1(x1)
        regression1 = paddle.transpose(regression1, (0, 2, 3, 1))
        regression1 = paddle.reshape(regression1, (N, -1, 4))

        classifier2 = self.classifier2(x2)
        classifier2 = paddle.transpose(classifier2, (0, 2, 3, 1))
        classifier2 = paddle.reshape(classifier2, (N, -1, 2))

        regression2 = self.regression2(x2)
        regression2 = paddle.transpose(regression2, (0, 2, 3, 1))
        regression2 = paddle.reshape(regression2, (N, -1, 4))

        classifier3 = self.classifier3(x)
        classifier3 = paddle.transpose(classifier3, (0, 2, 3, 1))
        classifier3 = paddle.reshape(classifier3, (N, -1, 2))

        regression3 = self.regression3(x)
        regression3 = paddle.transpose(regression3, (0, 2, 3, 1))
        regression3 = paddle.reshape(regression3, (N, -1, 4))

        classifier = paddle.concat([classifier1, classifier2, classifier3], axis=1)
        regression = paddle.concat([regression1, regression2, regression3], axis=1)

        return [regression, classifier]
        
def main():
    import paddle
    import numpy as np
    x = np.random.randn(1,3,1024, 1024).astype(np.float32)
    x = paddle.to_tensor(x)
    model = FaceBoxes()
    out = model(x)
    print(out[0].shape)
    print(out[1].shape)


if __name__ == '__main__':
    main()