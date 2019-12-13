import numpy as np
from utils import function

# 各レイヤで共通な機能を実装


class Layer(object):
    def __init__(self, lr=0.01):
        self.params = {}
        self.grads = {}
        self.lr = lr

    def update(self):
        for k in self.params.keys():
            self.params[k] = self.params[k] - self.lr * self.grads[k]

    def zerograd(self):
        for k in self.params.keys():
            self.grads[k] = np.zeros(
                shape=self.params[k].shape, dtype=self.params[k].dtype)

# 順伝播、逆伝播を行う機能を実装


class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers
    # レイヤの追加

    def addlayer(self, layer):
        self.layers.append(layer)
    # 順伝播の計算

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    # 逆伝播の計算

    def backward(self, y):
        for l in reversed(self.layers):
            y = l.backward(y)
        return y
    # 結合の重みを更新

    def update(self):
        for l in self.layers:
            l.update()
    # 勾配をゼロに

    def zerograd(self):
        for l in self.layers:
            l.zerograd()

# 全結合線形レイヤの実装


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        # 正規分布に従った乱数による重みの初期化（Xivierの初期値）
        self.params['W'] = np.random.normal(loc=0.0, scale=np.sqrt(
            1.0/input_dim), size=(input_dim, output_dim)).astype(np.float32)
        # バイアスの初期値をゼロに設定
        self.params['b'] = np.zeros(shape=(1, output_dim), dtype=np.float32)

    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, y):
        self.grads['W'] = np.dot(self.x.T, y)
        self.grads['b'] = y
        return y*self.params['W']

# シグモイド関数のレイヤの実装


class SigmoidLayer(Layer):
    def __init__(self):
        super(SigmoidLayer, self).__init__()

    def forward(self, x):
        z = function.sigmoid(x)
        self.z = z
        return z

    def backward(self, y):
        return np.array([np.sum(y*function.sigmoid_dash(self.z).T, axis=1)])


class Classfier:
    def __init__(self, model):
        self.model = model

    def update(self, x, t):
        self.model.zerograd()
        y = self.model.forward(x)
        prob = function.softmax(y)
        loss = function.cross_entropy(prob, t)
        dout = prob - t
        dout = self.model.backward(dout)
        self.model.update()

    def test(self, x):
        y = function.softmax(self.model.forward(x))
        prob = np.array([y[0]])
        return prob
