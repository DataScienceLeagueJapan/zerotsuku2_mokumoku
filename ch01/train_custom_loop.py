from sys import path
from pathlib import Path
path.append('/Users/yudaikawano/Library/Mobile Documents/com~apple~CloudDocs/dev/zerotsuku2_mokumoku/')

import numpy as np
import matplotlib.pyplot as plt

from two_layer_net import TwoLayerNet
from dataset import spiral
from common.optimizer import SDG

# 1. ハイパーパラメータの設定
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 2. データの読み込み・モデルとオプティマイザの設定
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SDG(learning_rate)

# 学習で使用する変数
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # データのシャッフル
    idx = np.random.permutation(data_size)
    x = x[idx]; t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size: (iters+1)*batch_size]
        batch_t = t[iters*batch_size: (iters+1)*batch_size]

        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) // 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d | iter %d / %d | loss %.2f'
                  % (epoch+1, iters+1, max_iters, avg_loss))
    loss_list.append(avg_loss)
    total_loss, loss_count = 0, 0

# プロット
x = np.linspace(1, max_epoch, len(loss_list))
y = loss_list
plt.figure(figsize=(7, 7))
axe = plt.subplot(1, 1, 1)
axe.plot(x, y)
plt.show()

