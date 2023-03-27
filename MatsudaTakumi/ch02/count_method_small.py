# 特異値分解
import sys
sys.path.append("..")
from common.utils import *
import numpy as np
import matplotlib.pyplot as plt

with open("sample_byChatGPT.txt","r") as f:
    text = f.read()

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

U,S,V = np.linalg.svd(W)

# 語数が多いのでランダムに10個選ぶ
random_choice = np.random.permutation(U.shape[0])[:10]

for word_id in random_choice:
    word = id_to_word[word_id]
    plt.annotate(word, (U[word_id,0], U[word_id,1]))
plt.scatter(U[random_choice,0], U[random_choice,1], alpha=0.5)
plt.savefig("SVD_map.png")
plt.show()