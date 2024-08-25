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