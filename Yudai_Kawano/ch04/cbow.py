import os, sys
sys.path.append(os.getcwd())

import numpy as np

from Yudai_Kawano.common.layers import Embedding
from Yudai_Kawano.ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW():
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V = vocab_size; H = hidden_size

        # 重みの初期化
        Win  = 0.01 * np.random.randn(V, H).astype('f')
        Wout = 0.01 * np.random.randn(V, H).astype('f')

        self.in_layers = []
        for _ in range(2 * window_size):
            layer = Embedding(Win)
            self.in_layers.append(layer)
        self.ns_layer = NegativeSamplingLoss(Wout, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_layer]
        self.params = []; self.grads = []
        for layer in layers:
            self.params += layer.params
            self.grads  += layer.grads

        self.word_vecs = Win

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h /= len(self.in_layers)
        loss = self.ns_layer.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_layer.backward(dout)
        dout /= len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None


