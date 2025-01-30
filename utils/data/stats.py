import numpy as np


def calculate_gini_like_index(class_proportions: dict):
    """
    计算类似基尼系数的指标来评估类不平衡度
    :param class_proportions: 每个类别的样本数量占比字典
    :return: 类似基尼系数的指标值
    """
    sorted_props = sorted(class_proportions.values(), reverse=True)
    cumulative_props = np.cumsum(sorted_props)
    lorenz_curve = cumulative_props / cumulative_props[-1]
    perfect_equality_line = np.arange(1, len(lorenz_curve) + 1) / len(lorenz_curve)
    gini_like_area = np.sum((lorenz_curve - perfect_equality_line))
    gini_like_index = gini_like_area / (len(lorenz_curve) - 1)
    return gini_like_index

def calculate_cross_entropy(class_proportions: dict):
    """
    计算交叉熵作为类不平衡度的度量
    :param class_proportions: 每个类别的样本数量占比字典
    :return: 交叉熵值
    """



