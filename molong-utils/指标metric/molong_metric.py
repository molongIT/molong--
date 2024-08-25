import numpy as np
import SimpleITK as sitk
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.morphology import skeletonize, skeletonize_3d
import cc3d


def dice_score(groundtruth, predict):
    """
    计算 Dice 系数，用于评估两个二值图像的重叠程度。

    参数:
    groundtruth (numpy.ndarray): Ground truth 二值图像数组。
    predict (numpy.ndarray): 预测结果的二值图像数组。

    返回:
    float: Dice 系数，值域为 [0, 1]，越接近 1 表示重叠程度越高。
    """
    # 将图像转换为二值形式，确保值为 0 或 1
    groundtruth = np.asarray(groundtruth).astype(np.bool_)
    predict = np.asarray(predict).astype(np.bool_)

    # 计算交集和两个图像的像素数之和
    intersection = np.sum(groundtruth & predict)
    sum_pixels = np.sum(groundtruth) + np.sum(predict)

    # 计算并返回 Dice 系数
    if sum_pixels == 0:
        return 1.0  # 如果两个图像都是空的，Dice 系数为 1.0
    else:
        dice = 2 * intersection / sum_pixels
        return dice

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

def clDice(v_p, v_l):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

    
# return : 表示输入图像中找到的连通分量的数量，即0维贝蒂数。
def conn_comp(arr):
    ## 疑问：这里明明是二维图像，用的确实三维的连通性分析。
    labels_out, numcomp = cc3d.connected_components(arr, connectivity=8, return_N=True) # 26-connected
    return numcomp

def get_betti_error(arr1, arr2, patchsize=[64,64], stepsize=[64,64]):
    arrsize = arr1.shape
    all_betti = []
    
    for x in range(0,arrsize[0],stepsize[0]):
        for y in range(0,arrsize[1],stepsize[1]):
            newidx = [x+patchsize[0],y+patchsize[1]]
            if(check_bounds([x,y],arrsize) and check_bounds(newidx,arrsize)):
                minivol1 = arr1[x:newidx[0],y:newidx[1]]
                minians1 = conn_comp(minivol1)

                minivol2 = arr2[x:newidx[0],y:newidx[1]]
                minians2 = conn_comp(minivol2)

                all_betti.append(np.abs(minians1-minians2))

    avg_betti = np.asarray(all_betti).mean()
    return avg_betti

def check_bounds(idx, volsize):
    if idx[0] < 0 or idx[0] > volsize[0]:
        return False
    if idx[1] < 0 or idx[1] > volsize[1]:
        return False
    return True

# accuracy可以衡量预测分割结果与真实标签的匹配程度，较高的accuracy表示模型的整体分类效果较好。
def accuracy(y_true, y_pred):
    """
    计算准确率，用于评估预测结果与真实标签的匹配程度。

    参数:
    y_true (numpy.ndarray): 真实标签的二值图像数组。
    y_pred (numpy.ndarray): 预测结果的二值图像数组。

    返回:
    float: 准确率，值域为 [0, 1]，越接近 1 表示匹配程度越高。
    """
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

# # Create dummy ground truth and prediction tensors
# y_true = torch.tensor([[[[0, 1], [1, 1]], [[1, 0], [0, 0]]]], dtype=torch.float32)
# y_pred = torch.tensor([[[[0, 1], [1, 1]], [[1, 1], [0, 0]]]], dtype=torch.float32)

# # Set a smoothing factor
# smooth = 1.0

# # Call the soft_cldice function
# cl_dice_score = soft_cldice(smooth, y_true, y_pred)

# # Print the result
# print(f"Soft clDice score: {cl_dice_score.item()}")
