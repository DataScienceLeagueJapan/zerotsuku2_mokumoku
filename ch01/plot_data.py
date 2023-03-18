from sys import path
path.append('/Users/yudaikawano/Library/Mobile Documents/com~apple~CloudDocs/dev/zerotsuku2_mokumoku/')

import numpy as np
import matplotlib.pyplot as plt
from dataset import spiral

x, t = spiral.load_data()

# 各行で1になる要素のインデックスを取得する
labels = np.array([np.where(row == 1)[0] for row in t]).flatten()
colors = ['red', 'blue', 'green']

for i in range(3):
    plt.scatter(x[labels==i, 0], x[labels==i, 1], color=colors[i])
plt.show()





