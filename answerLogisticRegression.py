import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 6.1e-2  # 学习率
wd = 5e-2  # l2正则化项系数
eps = 1e-8


def predict(X, weight, bias):
    """
    使用输入的weight和bias,预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出,即wx+b
    """
    # TODO: YOUR CODE HERE
    y = np.dot(X, weight) + bias
    return y

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    def forward(X, weight, bias):
        haty = predict(X, weight, bias)
        return haty
    
    def backward(haty, weight, bias, Y):
        z = haty * Y
        loss = np.mean(np.where(z > 10, -z, np.where(z > -5, -np.log(sigmoid(z) + eps), -z + np.log(1 + np.exp(z))))) + 0.5 * wd * np.sum(weight ** 2)

        threshold = 80
        t = np.where(z < threshold, sigmoid(-haty * Y), sigmoid(-threshold))
        dweight = np.mean(t[:, np.newaxis] * (-X) * Y[:, np.newaxis], axis=0) + wd * weight
        dbias = np.mean(t * (-Y))

        weight = weight - dweight * lr
        bias = bias - dbias * lr

        return (loss, weight, bias)
    haty = forward(X, weight, bias)
    loss, weight, bias = backward(haty, weight, bias, Y)
    return (haty, loss, weight, bias)

# 测试代码
'''
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 5
    n_features = 3
    X = np.random.randn(n_samples,n_features)
    Y = np.random.choice([1,-1],size=n_samples)
    weight = np.zeros(n_features)
    bias = np.zeros(1)
    
    for epoch in range(1,11):
        haty, loss, weight, bias = step(X, weight, bias, Y)
        print(f"Epoch {epoch}: loss =", loss)
'''    