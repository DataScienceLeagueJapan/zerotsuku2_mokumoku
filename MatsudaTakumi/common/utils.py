import numpy as np
from tqdm import tqdm

def preprocess(text):
    """
    小文字化
    --> 単語分割
    --> 単語id割当て
    
    Returns:
    
    corpus: array
        単語idの列
    word_to_id: dict
        単語 --> 単語id
    id_to_word: dict
        単語id --> 単語
    """
    text = text.lower()
    text = text.replace(".", " .")  # ピリオドを単語として分割するためにピリオド直前に空白を入れる
    text = text.replace(",", " ,")  # カンマを単語として分割するためにピリオド直前に空白を入れる
    text = text.replace('"', ' " ')  # クオーテーションマークを単語として分割するためにピリオド直前に空白を入れる
    words = text.split(" ")

    word_to_id = dict()
    id_to_word = dict()
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

def create_co_matrix(corpus, vocab_size, window_size=1):
    """
    共起行列を作成する
    """
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    """2つの単語ベクトルx,yのcos類似度を計算"""
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx,ny)


def most_similar(query, word_to_id, id_to_word, word_matrix,top):
    """単語queryに最も近い単語上位を表示"""
    if query not in word_to_id.keys():
        raise Exception(f"{query} is not found.")
    
    print(f"\n[query] {query}")
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    count = 0
    for i in (-similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(f"{id_to_word[i]}: {similarity[i]:.3f}")

        count += 1
        if count >= top:
            return None

def ppmi(C, verbose=False, eps=1e-8):
    """正の相互情報量(Positive PMI)を計算"""
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)  # 行ごとの合計値
    
    if verbose:
        for i in tqdm(range(C.shape[0])):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i,j] * N / (S[i]*S[j]) + eps)
                M[i,j] = max(0, pmi)
    else:
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                pmi = np.log2(C[i,j] * N / (S[i]*S[j]) + eps)
                M[i,j] = max(0, pmi)
    
    return M
