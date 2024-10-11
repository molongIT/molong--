import numpy as np
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt

### —————————————————————计算连通方向
def calculate_direction_pixel_count(image, x, y, maxPixel,max_check=5, debug=False):
    """
    计算图像中某个像素在四个方向上的连续像素数量。
    
    :param image: 二值化图像矩阵
    :param x: 当前像素的x坐标
    :param y: 当前像素的y坐标
    :param max_check: 每个方向最大检查的步数
    :return: 包含四个方向的像素数量的数组 [水平, 垂直, 45度对角线, 135度对角线]
    """
    directions = {
        'horizontal': (0, 1),  # 水平方向
        'diagonal_45': (-1, 1), # 45度方向
        'vertical': (-1, 0),    # 垂直方向
        'diagonal_135': (-1, -1) # 135度方向
    }
    counts = [(0,0), (0,0), (0,0), (0,0)]  # 用于存储每个方向的计数

    for i, (dx, dy) in enumerate(directions.values()):
        positive_count = 0
        negative_count = 0        # 正方向检查
        for step in range(1, max_check + 1):
            new_x = x + step * dx
            new_y = y + step * dy
            if debug: print(f"检查正方向: ({new_x}, {new_y})")
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0] and image[new_x, new_y] == maxPixel:
                positive_count += 1
            else:
                continue
        
        # 负方向检查
        for step in range(1, max_check + 1):
            new_x = x - step * dx
            new_y = y - step * dy
            if debug: print(f"检查负方向: ({new_x}, {new_y})")
            if 0 <= new_x < image.shape[1] and 0 <= new_y < image.shape[0] and image[new_x, new_y] == maxPixel:
                negative_count += 1
            else:
                continue
        
        counts[i] = (positive_count, negative_count)  # 更新计数为 (正方向, 负方向)
        if debug: print(f"方向 {i} 的累计值: {counts[i]}")
    return counts


### —————————————————————计算差分loss

def calculate_non_connectivity_error(image, direction_counts, x, y,debug=False):
    """
    计算非连通误差，基于相邻像素差值。
    
    :param image: 图像像素矩阵
    :param direction_counts: 四个方向上正负像素数量统计 [(正, 负), ...]
    :param x: 当前像素的x坐标
    :param y: 当前像素的y坐标
    :return: 误差值
    """
    # 定义四个方向对应的增量 (dx, dy)
    directions = {
        'horizontal': (0, 1),  # 水平方向
        'diagonal_45': (-1, 1),  # 45度方向
        'vertical': (-1, 0),  # 垂直方向
        'diagonal_135': (-1, -1)  # 135度方向
    }
    
    # 找到最大方向索引（正负数目和最大的）
    max_index = max(range(len(direction_counts)), key=lambda i: sum(direction_counts[i]))

    # 获取最大方向的正负数
    positive_count, negative_count = direction_counts[max_index]
    if debug:print(f"方向: {positive_count},{negative_count}")
    dx, dy = list(directions.values())[max_index]

    # 如果正负数目相同，使用策略1
    if positive_count == negative_count:
        errors = 0
        for step in [1, 2]:
            # 检查正负方向的两个位置
            pos_x, pos_y = x + step * dx, y + step * dy
            neg_x, neg_y = x - step * dx, y - step * dy
            if (0 <= pos_x < image.shape[1] and 0 <= pos_y < image.shape[0] and
                0 <= neg_x < image.shape[1] and 0 <= neg_y < image.shape[0]):
                if image[pos_y, pos_x] != image[neg_y, neg_x]:
                    errors += 1  # 如果不相等，则 errors + 1
        return errors

    # 使用策略2，侧重方向少的
    else:
        errors = 0
        if positive_count < negative_count:
            # 偏重正方向
            if debug:print("偏重正方向")
            for step in [1, 2, 3]:
                pos_x, pos_y = x + step * dx, y + step * dy
                neg_x, neg_y = x - 1 * dx, y - 1 * dy
                if (0 <= pos_x < image.shape[1] and 0 <= pos_y < image.shape[0] and
                    0 <= neg_x < image.shape[1] and 0 <= neg_y < image.shape[0]):
                    if image[pos_x, pos_y] != image[neg_x, neg_y]:
                        errors += 1  # 如果不相等，则 errors + 1
        else:
            # 偏重负方向
            if debug:print("偏重负方向")
            for step in [1, 2, 3]:
                pos_x, pos_y = x + 1 * dx, y + 1 * dy
                neg_x, neg_y = x - step * dx, y - step * dy
                if (0 <= pos_x < image.shape[1] and 0 <= pos_y < image.shape[0] and
                    0 <= neg_x < image.shape[1] and 0 <= neg_y < image.shape[0]):
                    if image[pos_x, pos_y] != image[neg_x, neg_y]:
                        errors += 1  # 如果不相等，则 errors + 1
        return errors

#### ————————————————————增强边缘
def extract_edges(matrix):
    """
    提取矩阵中的边缘像素，边缘像素是那些值为255且至少有一个邻居是0的像素。
    
    :param matrix: 原始图像的二值矩阵
    :return: 同样大小的矩阵，边缘像素为1，其他像素为0
    """
    rows, cols = matrix.shape
    # 创建一个与原始矩阵大小相同的全零矩阵
    edge_matrix = np.zeros((rows, cols), dtype=int)
    
    # 遍历矩阵中每个像素
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 检查当前像素是否为255
            if matrix[i, j] == 255:
                # 检查上下左右是否有任意一个是0
                if (matrix[i-1, j] == 0 or matrix[i+1, j] == 0 or 
                    matrix[i, j-1] == 0 or matrix[i, j+1] == 0):
                    # 将边缘像素位置设置为1
                    edge_matrix[i, j] = 1
    
    return edge_matrix

def bfs_label(matrix, edge_result):
    """
    使用 BFS 对每个连通区域进行标记，生成一个与 edge_result 相同大小的标签矩阵。
    8邻域连通的像素点算作一个连通区域。
    
    :param matrix: 原始图像的二值矩阵
    :param edge_result: 边缘检测结果矩阵，边缘位置为 1，其余为 0
    :return: 标签矩阵，每个连通区域有唯一的标签
    """
    rows, cols = edge_result.shape
    label_matrix = np.zeros_like(edge_result, dtype=int)
    label = 1

    # 用于八个方向的移动
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 遍历每个像素，找到未标记的边缘像素
    for i in range(rows):
        for j in range(cols):
            if edge_result[i, j] == 1 and label_matrix[i, j] == 0:
                # 使用 BFS 对连通区域进行标记
                queue = deque([(i, j)])
                label_matrix[i, j] = label

                while queue:
                    x, y = queue.popleft()

                    # 检查八个方向的邻接像素
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if edge_result[nx, ny] == 1 and label_matrix[nx, ny] == 0:
                                label_matrix[nx, ny] = label
                                queue.append((nx, ny))

                label += 1

    return label_matrix

def enhance_edges(matrix, edge_result, label_matrix, max_value=2):
    """
    在原始图像中遍历每一个 edge_result 为 1 的位置，将8邻域相邻的像素值进行增强，
    并确保同一个连通区域仅贡献一次。
    
    :param matrix: 原始图像的二值矩阵
    :param edge_result: 边缘检测结果矩阵，边缘位置为 1，其余为 0
    :param label_matrix: 连通区域的标签矩阵
    :param max_value: 最大允许的增强值
    :return: 增强后的原始图像矩阵
    """
    rows, cols = matrix.shape
    contributions = {}  # map：记录每个像素的贡献情况，key：元组（坐标x，坐标y），value：set(label_No,...)

    # 遍历 edge_result 矩阵
    for i in range(rows):
        for j in range(cols):
            if edge_result[i, j] == 1:
                label = label_matrix[i, j]  # 当前边缘像素的连通区域标签

                # 检查八个方向的邻接像素
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if matrix[ni, nj] != 255:  # 只有背景像素才能被增强
                            # 初始化贡献记录
                            if (ni, nj) not in contributions:
                                contributions[(ni, nj)] = set()
                            # 检查是否已经被当前连通区域贡献过
                            if label not in contributions[(ni, nj)]:
                                contributions[(ni, nj)].add(label)
                                if matrix[ni, nj] < max_value:
                                    matrix[ni, nj] += 1
    
    return matrix, contributions

def set_values_to_max(matrix, min_value=1, max_value=255):
    """
    将矩阵中处于 min_value 和 max_value 之间的像素值设置为 max_value。
    
    :param matrix: 原始图像矩阵
    :param min_value: 最小阈值，默认值为 1
    :param max_value: 设置为的最大值，默认值为 255
    :return: 处理后的矩阵
    """
    # 创建一个布尔掩码，查找所有大于 min_value 小于 max_value 的值
    mask = (matrix > min_value) & (matrix < max_value)
    # 将这些值设置为 max_value
    matrix[mask] = max_value
    return matrix

def set_values_by_flag(matrix, flags, target_value):
    """
    根据标记数组 flags，将 matrix 中 flags 为 1 的位置设置为 target_value。
    
    :param matrix: 原始图像的二值矩阵
    :param flags: 标记数组，与 matrix 尺寸相同，1 表示需要修改的像素位置
    :param target_value: 需要设置的目标值
    :return: 修改后的矩阵
    """
    # 将 flags 为 1 的位置设置为 target_value
    matrix[flags == 1] = target_value
    return matrix


# 绘制原始矩阵和处理后的矩阵在一张图中
def save_combined_image(original_matrix, processed_matrix1, processed_matrix2, save_path, save=False):
    """
    绘制原始矩阵和处理后的矩阵在一张图中并保存到指定路径。
    
    :param original_matrix: 原始图像矩阵
    :param processed_matrix1: 处理后的图像矩阵1
    :param processed_matrix2: 处理后的图像矩阵2
    :param save_path: 保存图像的路径
    :param save: 是否保存图像
    """
    
    # 确保所有输入矩阵为numpy数组且为数值类型
    original_matrix = np.array(original_matrix, dtype=np.float32)
    processed_matrix1 = np.array(processed_matrix1, dtype=np.float32)
    processed_matrix2 = np.array(processed_matrix2, dtype=np.float32)
    
    # 创建绘图窗口，设置大小
    plt.figure(figsize=(40, 20))
    
    # 绘制原始矩阵
    plt.subplot(1, 3, 1)
    plt.imshow(original_matrix, cmap='gray', interpolation='none')
    plt.title("Original Matrix")
    plt.axis('off')
    
    # 绘制处理后的矩阵1
    plt.subplot(1, 3, 2)
    plt.imshow(processed_matrix1, cmap='gray', interpolation='none')
    plt.title("Processed Matrix1")
    plt.axis('off')
    
    # 绘制处理后的矩阵2
    plt.subplot(1, 3, 3)
    plt.imshow(processed_matrix2, cmap='gray', interpolation='none')
    plt.title("Processed Matrix2")
    plt.axis('off')
    
    # 保存图像到指定路径或显示
    if save:
        plt.savefig(save_path)
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
        
    # 关闭绘图窗口
    plt.close()

def mark_combined_contributions(contributions, matrix):
    """
    根据 contributions 中的记录，标记相邻像素由不同区域贡献的地方。
    
    :param contributions: 字典记录每个像素的贡献情况，key: (x, y), value: set(label_No,...)
    :param matrix_shape: 原始图像的尺寸 (rows, cols)
    :return: 新的标注数组，标记位置值为 1
    """
    # 初始化标记数组，大小和原始图像相同
    marked_array = np.zeros(matrix.shape, dtype=int)
    
    # 方向列表用于检查8邻域
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # 遍历 contributions 中的每一个坐标点
    for (x, y), labels in contributions.items():
        # 如果当前坐标的 labels 集合中包含多个 label，说明已找到交集，跳过不参与邻近增强
        if len(labels) > 1:
            continue
        
        # 检查当前坐标的8邻域
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # 判断邻域坐标是否在 contributions 中
            if (nx, ny) in contributions:
                # 检查邻域是否属于不同的 label
                if contributions[(nx, ny)].isdisjoint(labels):  # 确保没有交集
                    # 标记当前坐标和邻域坐标
                    marked_array[x, y] = 1
                    marked_array[nx, ny] = 1

    return marked_array




# 原始图像矩阵
matrix_example = np.array([
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255, 255,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [ 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [ 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
    [ 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0 ],
])

edges = extract_edges(matrix_example)
labels = bfs_label(matrix_example,edges)

newMatrix = matrix_example.copy() # 复制原始矩阵
enhance,contributions =  enhance_edges(newMatrix, edges, labels)
ConnResult = set_values_to_max(enhance)

# 迭代相邻增强

ConnResult1 = set_values_to_max(enhance.copy())

flag = mark_combined_contributions(contributions, enhance)
enhance = set_values_by_flag(enhance,flag,3)
ConnResult2 = set_values_to_max(enhance.copy())

save_path = "/home/pxl/myProject/血管分割/molong-深度插值/molong-utils/combined_image.png"  # 指定保存路径
save_combined_image(matrix_example,ConnResult1,ConnResult2, save_path,save=True)