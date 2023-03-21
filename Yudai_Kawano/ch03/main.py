import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from common.util import preprocess, create_contexts_target, convert_one_hot

np.set_printoptions(precision=3)

text = "you say goodbye and I say hello."
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size=7)
contexts = convert_one_hot(contexts, vocab_size=7)


