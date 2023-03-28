import sys
sys.path.append("..")
from common.utils import most_similar
import pickle

with open("cbow_params.pkl", "rb") as f:
    params = pickle.load(f)
    word_vecs = params["word_vecs"]
    word_to_id = params["word_to_id"]
    id_to_word = params["id_to_word"]

queries = ["you", "year", "car", "toyota", "happy", "important"]
for query in queries:
    most_similar(query, word_to_id, id_to_word, word_vecs, 5)