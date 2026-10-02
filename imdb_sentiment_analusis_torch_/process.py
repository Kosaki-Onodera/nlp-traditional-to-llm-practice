import logging
import os
import re
import sys
import numpy as np
from itertools import chain
from gensim.models import KeyedVectors
import gensim
import pandas as pd
import torch
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import pickle

# =================== 超参（和你本地保持一致） ===================
embed_size = 300
max_len = 512

# =================== Kaggle路径【自行核对修改】 ===================
TRAIN_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/labeledTrainData.tsv.zip"
TEST_PATH = "/kaggle/input/competitions/word2vec-nlp-tutorial/testData.tsv.zip"
GLOVE_PATH = "/kaggle/input/datasets/gongbaoxin/common-crawl-840b/glove.840B.300d.txt"

# =================== 文本清洗函数（原版不动） ===================
def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "lxml").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    return words

def encode_samples(tokenized_samples, word_to_idx):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

def pad_samples(features, maxlen=max_len, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature.copy()
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features

# =================== 主流程 ===================
os.makedirs("/kaggle/working/pickle", exist_ok=True)

train = pd.read_csv(TRAIN_PATH, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(TEST_PATH, header=0, delimiter="\t", quoting=3)

clean_train_reviews, train_labels = [], []
for i, review in enumerate(train["review"]):
    clean_train_reviews.append(review_to_wordlist(review))
    train_labels.append(train["sentiment"][i])

clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append(review_to_wordlist(review))

vocab = set(chain(*clean_train_reviews)) | set(chain(*clean_test_reviews))
vocab_size = len(vocab)

train_reviews, val_reviews, train_labels, val_labels = train_test_split(
    clean_train_reviews, train_labels, test_size=0.2, random_state=0)

# ===================【重点修改】适配Common Crawl 840B（glove-gensim分割逻辑） ===================
wvmodel = KeyedVectors(embed_size)
word_dict = {}
with open(GLOVE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        if len(tokens) <= embed_size:
            continue
        try:
            vec = np.array(tokens[-embed_size:], dtype=np.float32)
            word = " ".join(tokens[:-embed_size])
            word_dict[word] = vec
        except ValueError:
            continue
wvmodel.add_vectors(list(word_dict.keys()), list(word_dict.values()))
print(f"GloVe加载完成，载入词总数：{len(wvmodel)}")
# =========================================================================================

word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'

train_features = torch.tensor(pad_samples(encode_samples(train_reviews, word_to_idx)))
val_features = torch.tensor(pad_samples(encode_samples(val_reviews, word_to_idx)))
test_features = torch.tensor(pad_samples(encode_samples(clean_test_reviews, word_to_idx)))

train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

# 构建Embedding权重矩阵
weight = torch.zeros(vocab_size + 1, embed_size)
hit = 0
for word, idx in word_to_idx.items():
    if word in wvmodel:
        weight[idx, :] = torch.from_numpy(wvmodel.get_vector(word))
        hit += 1
print(f"词表匹配成功向量：{hit}/{len(word_to_idx)}")

pickle_file = "/kaggle/working/pickle/imdb_glove.pickle3"
pickle.dump(
    [train_features, train_labels, val_features, val_labels, test_features, weight, word_to_idx, idx_to_word, vocab],
    open(pickle_file, 'wb'))
print('pickle文件生成完成！')