import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, most_similar, ppmi

np.set_printoptions(precision=3)

text = "you say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

#most_similar('say', word_to_id, id_to_word, C) #　関連度の高い順に表示

W = ppmi(C)

# 共起行列 と PPMIの表示
# print('covariance matrix')
# print(C)
# print('-' * 50)
# print('PPMI')
# print(W)


# SVD
U, S, V = np.linalg.svd(W)
# print(C[0])
# print(W[0])
# print(U[0])

for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
plt.show()
