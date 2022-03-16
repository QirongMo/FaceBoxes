
# FaceBoxes
paddlepaddle复现[《FaceBoxes: A CPU Real-time Face Detector with High Accuracy》](https://arxiv.org/abs/1708.05234)进行人脸检测，其中部分层用深度可分离卷积替代标准卷积。

数据集采用[FDDB数据集](http://vis-www.cs.umass.edu/fddb/index.html#download)来做训练，需要下载原图“Original, unannotated set of images“和人脸标注”Face annotations“两个文件解压到FDDB文件夹，然后运行getFDDBtxt.py获取FDDB.txt。
