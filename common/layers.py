import numpy as np

# MatMulレイヤ
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    
    def forward(self, x):
        W, = self.params  # コンマ付ければリストの要素だけをWに代入できる!!
        out = np.dot(x, W)
        self.x = x
        return out
    
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
    
# Sigmoidレイヤ
class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None
    
    def forward(self, x):
        out = 1. / (1. + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1. - self.out)
        return dx
    
# Affineレイヤ(MatMulレイヤを使うバージョン)
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        self.matmul = MatMul(W)
    
    def forward(self,x):
        W, b = self.params
        out = self.matmul.forward(x) + b
        self.x = x
        return out
    
    def backward(self, dout):
        W, b = self.params
        dx = self.matmul.backward(dout)
        dW, = self.matmul.grads
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
    
# softmax関数
def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)  # オーバーフローを防ぐため
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x)
        x /= np.sum(x)
    return x

# クロスエントロピー誤差関数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# Softmaxレイヤ
class Softmax:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None

    def forward(self,x):
        self.out = softmax(x)
        return self.out
    
    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx
    
# SoftmaxwithLossレイヤ
class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx
    



