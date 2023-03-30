import sys
sys.path.append("..")
import numpy as np
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from common.utils import create_contexts_target
from dataset import ptb

window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

corpus, word_to_id, id_to_word = ptb.load_data("train")
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size, eval_interval=100)
trainer.plot("cbow.png")

word_vecs = model.word_vecs
params = dict()
params["word_vecs"] = word_vecs
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
with open("cbow_params.pkl", "wb") as f:
    pickle.dump(params, f, -1)