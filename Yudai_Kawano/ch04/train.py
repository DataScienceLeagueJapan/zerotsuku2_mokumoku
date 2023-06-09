import sys, os
sys.path.append(os.getcwd())

import numpy as np
import pickle

from Yudai_Kawano.common.trainer import Trainer
from Yudai_Kawano.common.optimizer import Adam
from cbow import CBOW
from Yudai_Kawano.common.util import create_contexts_target
from Yudai_Kawano.dataset import ptb

# ハイパーパラメータ
window_size = 5
hidden_size = 100
batch_size  = 100
max_epoch   = 10

# データの読み込み
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

# モデルなどの作成
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 学習開始
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# 分散表現を保存
word_vecs = model.word_vecs
params = {}
params['word_vecs']  = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)


