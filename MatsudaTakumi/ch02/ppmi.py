import sys
sys.path.append("..")
from common.utils import *
import seaborn as sns
import matplotlib.pyplot as plt

with open("sample_byChatGPT.txt","r") as f:
    text = f.read()

corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C, verbose=True)

sns.heatmap(C)
plt.title("co matrix")
plt.savefig("co_matrix.png")
plt.show()

sns.heatmap(W)
plt.title("PPMI")
plt.savefig("ppmi.png")
plt.show()