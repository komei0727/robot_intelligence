import numpy as np
from numpy.random import *
from utils import mnist_reader
import Layer
import random

# 画像データの圧縮ファイル
dl_list = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

# 画像データのあるディレクトリ(mnist)
dataset_dir = '/Users/komei0727/workspace/robot_intelligence/data/mnist'
# 画像データのあるディレクトリ(fashion_mnist)
# dataset_dir = '/Users/komei0727/workspace/robot_intelligence/data/fashion_mnist'

# 画像データの読み込み
dataset = mnist_reader.load_mnist(dl_list, dataset_dir)
train_img, train_label, test_img, test_label = dataset

# 画素値を0~1の間の値に変換
train_img = train_img / 255
test_img = test_img / 255
train_img = train_img.astype(np.float32)
# ラベルをone-hot表現で表す
train_label = [int(x) for x in train_label]
train_label_one_hot = np.identity(10)[train_label].astype(np.float32)
test_label = [int(x) for x in test_label]
test_label_one_hot = np.identity(10)[test_label]

# ノイズの設定
noise_rate = 10
for i in range(len(train_img)):
    for j in range(len(train_img[0])):
        random_number = random.uniform(0, 100)
        if random_number < noise_rate:
            train_img[i][j] = random.random()

model = Layer.Sequential()
model.addlayer(Layer.LinearLayer(784, 1000))
model.addlayer(Layer.SigmoidLayer())
model.addlayer(Layer.LinearLayer(1000, 10))
classifier = Layer.Classfier(model)

for i in range(100):
    rand1 = np.arange(60000)
    shuffle(rand1)
    rand2 = np.arange(10000)
    shuffle(rand2)
    train_img = train_img[rand1, :]
    train_label_one_hot = train_label_one_hot[rand1, :]
    test_img = test_img[rand2]
    test_label_one_hot = np.array(test_label_one_hot[rand2, :])
    for j in range(200):
        x = np.array([train_img[j]])
        t = np.array([train_label_one_hot[j]])
        classifier.update(x, t)
    count = 0
    for j in range(100):
        x = np.array([test_img[j]])
        prob = classifier.test(x)
        if np.argmax(prob[0]) == np.argmax(test_label_one_hot[j]):
            count += 1
    print(i+1)
    print(count/100)
