import numpy as np
"""
  这个类用于融合拓扑先验，找到拓扑失败点，实现长程拓扑关联探索模块。
"""
class topologyClass:
  
  def __init__(self):
    pass
  
  def test(self):
    print('这个类用于融合拓扑先验，找到拓扑失败点，已经实现长程拓扑关联探索模块。')
    
  """
  根据给定的比例系数分割图像成多个块。参数:
  mg (np.ndarray): 输入图像。
  cale_factor (float): 块相对于图像尺寸的比例。
  返回:
  list: 包含所有块的列表。
  """
  def split_image_into_tiles(img, scale_factor):
      height, width = img.shape[:2]
      # 计算基于最小边的块大小
      tile_size = int(min(height, width) * scale_factor)

      tiles = []
      for y in range(0, height, tile_size):
          for x in range(0, width, tile_size):
              tiles.append(img[y:y+tile_size, x:x+tile_size])
      return tiles

  def topologicalPriorAssistance(self,image) -> None:
      print("执行拓扑先验辅助处理...")
      # 1. 要求连通条件。
      # 2. 团的话，是否是需要连接。
      # 3. 

  def findCriticalPointSet(self, image: np.ndarray) -> np.ndarray:
    source = image
    self.topologicalPriorAssistance(source)

    
class DeepTopoInterpolation:
# 主程序入口
t = topologyClass()
t.test()
