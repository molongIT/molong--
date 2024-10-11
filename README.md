# molong--深度插值

## molong-resource
* 测试的图像
* 处理后的结果图像

## molong-utils
* 指标metric已经慢慢补充，主要用来测试算法效果。
包括：dice、clDice、0betti值、accracy指标
* test连通量去除孔洞，是主要的代码，用于实验拓扑失败点获取。

## molong-utils
* DeepClosing-main是对比的方法，后续要复现成功
* molong-DeepTopoInterpolation是主要的项目入口，包括模型训练和测试
* simple-unet-2d是Unet+topoloss的项目，已经复现成功，效果比Unet好（调参好），但是创新点作用一般。
* Unet是最原始的医学分割模型，已经复现，效果很差。

## 实验
* 两台电脑一起跑!都利用起来!

## task:
到底要不要做!

## tda的cubical complex,是有cube矩形一步步组成.
* Cubical Complex生成Barcode图像了. 需要概率图.
* Filtration: 在获取二值化的图像后，我们继续通过filtration进行预处理。当然也可以直接使用二值化的图像作为输入进行TDA，这种为binary filtration。:通过逐步增加“尺度”（如从低值到高值），我们可以观察到在不同尺度下数据形状（如连通分量、环、空腔等）的生成和消失。这种分析提供了一个对数据内部结构的全景视图，有助于发现和理解数据的多尺度特征。
* barcode或者diagram，那么如何把他变成一个数值、数组之类的，这样才能作为特征放入机器学习呀？
  将persistent barcode/diagram矢量化（vectorization）。
