import numpy as np

class Sigmoid:
    """
    シグモイド関数
    """
    def __init__(self):
        self.params = [] # シグモイド関数は学習パラメータがないので空
        
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    
class Affine:
    """
    全結合層
    """
    def __init__(self, W, b):
        self.params = [W, b] # 重みとバイアスを学習する
        
    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out
    
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        # 重みとバイアスの初期化
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        # レイヤの生成
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        
        # 全ての重みをリストにまとめる
        self.params = []
        for layer in self.layers:
            self.params += layer.params
            
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          