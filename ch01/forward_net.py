import numpy as np

# Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.params = []  # 学習するパラメータは無いので空のまま
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
# Affineレイヤ
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]  # 学習すべきパラメータは、重みWとバイアスb

    def forward(self, x):
        W, b = self.params
        return np.dot(x, W) + b
    
# 単純なNN
class TwoLayerNet:
    def __init__(self,input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 重みWとバイアスbを初期化(標準正規分布による初期化)
        W1 = np.random.randn(I,H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H,O)
        b2 = np.random.randn(O)

        # レイヤ
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b1)
        ]

        # 全ての重みパラメータをリストにまとめる
        # ↑ パラメータ更新・保存を容易にするため
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)  # 各層に対して順伝播を行う
        return x