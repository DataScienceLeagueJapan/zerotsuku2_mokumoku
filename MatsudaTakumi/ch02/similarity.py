# you と I の類似度を求めるプログラム

import sys
sys.path.append("..")
from common.utils import *

with open("sample_byChatGPT.txt","r") as f:
    text = f.read()

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
print(cos_similarity(c0,c1))

most_similar("you", word_to_id, id_to_word, C, 5)