import sys
import os
sys.path.append(os.getcwd())

import numpy as np

from common.util import preprocess, create_co_matrix, most_similar, ppmi

text = "you say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
most_similar('say', word_to_id, id_to_word, C)

W = ppmi(C)

np.set_printoptions(precision=3)
print('covariance matrix')
print(C)
print('-' * 50)
print('PPMI')
print(W)

