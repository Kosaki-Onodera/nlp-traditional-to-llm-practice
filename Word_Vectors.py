import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from Bags_of_Words import review_to_words


def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences=[]
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_to_wordlist(raw_sentence,remove_stopwords))
    return sentences


train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,delimiter="\t",quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'unlabeledTrainData.tsv'), header=0, delimiter="\t", quoting=3)

nltk.download('punkt')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences=[]
print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review,tokenizer)  
    
print ("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review,tokenizer)

num_features = 300    # 词向量维度               
min_word_count = 40   # 词的最小出现频次                  
num_workers = 12      # 并行运行线程数
context = 10          # 上下文窗口大小
downsampling = 1e-3   # 高频词下采样阈值

# 初始化和训练模型
from gensim.models import word2vec
print ("Training model")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

model.init_sims(replace=True)

model_name = "300features_40minwords_10context"
model.save(model_name)