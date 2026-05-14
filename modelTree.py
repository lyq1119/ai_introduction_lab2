import numpy as np
# you only need matplotlib if you want to create some plots of the data
import matplotlib.pyplot as plt
import mnist
from answerTree import *
import pickle
from util import setseed

setseed(2) # 固定随机数种子以提高可复现性

save_path = "model/tree.npy"

def discretize(x, bin=64):
    return np.divmod(x, bin)[0] # x//bin

trn_X = discretize(mnist.trn_X) # shape:(1000,784) 训练集图像
trn_Y = mnist.trn_Y # shape:(1000,) 训练集标签
val_X = discretize(mnist.val_X) # shape:(69000,784) 验证集图像
val_Y = mnist.val_Y # shape:(69000,) 验证集标签

if __name__ == "__main__":
    hyperparams["gainfunc"] = eval(hyperparams["gainfunc"]) # 超参数，分别为树的最大深度、熵的阈值、信息增益函数
    root = buildTree(trn_X, trn_Y, list(range(mnist.num_feat)), 0, **hyperparams)
    with open(save_path, "wb") as f:
        pickle.dump(root, f)
    pred = np.array([inferTree(root, val_X[i]) for i in range(val_X.shape[0])])
    print("valid acc", np.average(pred==val_Y))