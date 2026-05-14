import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {
    "maxdepth": 100, 
    "purity_bound": 1e-1, 
    "gainfunc": "gainratio" 
    }

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: (n,), 标签向量
    @return: 熵
    """
    # TODO: YOUR CODE HERE
    if Y.shape[0] == 0:
        return 0
    uY, Ycnt = np.unique(Y, return_counts=True)
    Ycnt = Ycnt[Ycnt > 0]
    Yp = Ycnt / np.sum(Ycnt)
    en_y = np.sum(-Yp * (np.log2(Yp)))
    return en_y

def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True) 
    # ufeat是np.array([所有不同的元素])，featcnt是np.array([元素出现次数])
    featp = featcnt / feat.shape[0] # featcnt是np.array([元素出现概率])
    ret = 0
    # TODO: YOUR CODE HERE
    hY = entropy(Y)
    hY_if_idx = 0
    for i in range(ufeat.shape[0]):
        val = ufeat[i]
        valp = featp[i]
        Y_idx_val = np.array([Y[indice] for indice in np.where(feat==val)[0]])
        hY_if_idx += valp * entropy(Y_idx_val)
    ret = hY - hY_if_idx
    return ret

def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: (n,), 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask]) # 提取所有feat值为u对应的索引的Y的点
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点列表，其中key是特征的取值，value是子节点（Node）
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return: bool, 是否为叶节点
        """
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, maxdepth: int, purity_bound: float, gainfunc: Callable):
    """
    递归构建决策树。
    @params X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @params Y: (n,), 样本的label
    @params unused: List of int, 未使用的特征索引
    @params depth: int, 树的当前深度
    @params purity_bound: float, 熵的阈值
    @params gainfunc: Callable, 信息增益函数
    @return: Node, 决策树的根节点
    """  

    root = Node()
    u, ucnt = np.unique(Y, return_counts=True)
    root.label = u[np.argmax(ucnt)]
    # 当达到终止条件时，返回叶节点
    # TODO: YOUR CODE HERE
    if entropy(Y) <= purity_bound or not unused or depth >= maxdepth:
        return root
    
    gains = [gainfunc(X, Y, i) for i in unused]
    idx = np.argmax(gains)
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)
    feat = X[:, root.featidx]
    ufeat = np.unique(feat)
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    # TODO: YOUR CODE HERE
    for val in ufeat:
        mask = (feat == val)
        X_val = X[mask]
        Y_val = Y[mask] 
        root.children[val] = buildTree(X_val, Y_val, unused, depth + 1, maxdepth, purity_bound, gainfunc)
    return root

def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

"""X = np.array([[1,2,3],[1,3,2],[2,3,4],[3,2,1],[3,2,4]])
Y = np.array([1,2,2,3,1])
buildTree(X, Y, [0,1,2], 0, 0, gainratio)"""