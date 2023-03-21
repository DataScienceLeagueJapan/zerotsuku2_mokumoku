import sys
sys.path.append("..")
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

max_epoch = 300
batch_size = 30
hidden_size = 10
learning_late = 1.

x,t = spiral.load_data()
model = TwoLayerNet(2, hidden_size, 3)
optimizer = SGD(learning_late)

max_iters = len(x) // batch_size
total_loss = 0.
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    idx = np.random.permutation(len(x))
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(f"epoch:{epoch+1}  iter:{iters+1}/{max_iters}  loss:{avg_loss:.2f}")
            total_loss, loss_count = 0, 0
            loss_list.append(avg_loss)

plt.plot(loss_list)
plt.savefig("LOSS_train_custom_loop.png")