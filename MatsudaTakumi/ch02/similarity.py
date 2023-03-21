# you と I の類似度を求めるプログラム

import sys
sys.path.append("..")
from common.utils import *

# ChatGPTに生成してもらったテキスト
text = """
As I sit here, contemplating my life, I can't help but think about you. You have been such an important part of my journey, and I am grateful for every moment we have shared together. I remember the first time we met, and how I felt an instant connection to you. It was like we had known each other for years, even though we had just met.
As I reflect on our friendship, I realize how much I rely on you. You are always there for me, through thick and thin, and I know that I can count on you no matter what. Whether I need someone to vent to or just a shoulder to cry on, you are always there to listen and offer support.
At the same time, I know that I have been there for you too. I have seen you through some tough times, and I hope that I have been able to offer you the same level of support that you have given me. That's what true friendship is all about, after all - being there for each other through the ups and downs of life.
As I continue on my journey, I know that I will always have you by my side. I am grateful for your friendship and everything that you bring to my life. I can't wait to see where our journey takes us next, and I am excited to continue growing and learning alongside you.
"""
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id["you"]]
c1 = C[word_to_id["i"]]
print(cos_similarity(c0,c1))