import sys
import os
sys.path.append(os.getcwd())

from numpy.random import randn

from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW():
    def __init__(self, vocab_size, hidden_size) -> None:
        V = vocab_size; H = hidden_size

        W_in  = 0.01 * randn(V, H)
        W_out = 0.01 * randn(V, H)

        self.in_layer0  = MatMul(W_in)
        self.in_layer1  = MatMul(W_in)
        self.out_layer  = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        self.layers = [self.in_layer0, self.in_layer1, self.out_layer]

        self.params = []; self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])

        h = (h0 + h1) / 2
        score = self.out_layer.forward(h)
        loss = self.loss_layer(score, target)
        return loss

    def backward(self,  dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer0.backward(da)
        self.in_layer1.backward(da)
        return None
