import os
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import nltk

#***********将单条评论转换为单词列表的函数***********
def review_to_words(raw_review):
    # 1.去除HTML标签
    review = BeautifulSoup(raw_review).get_text()
    # 2.去除非字母字符
    letters_only = re.sub("[^a-zA-Z]", " ", review)
    # 3.转换为小写并拆分为单词
    lower_case = letters_only.lower()
    words = lower_case.split()
    # 4.去除停用词
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    # 5.将单词列表重新组合为字符串
    return " ".join(meaningful_words)


#***********读取并处理训练集和测试集数据***********
train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0,delimiter="\t",quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )

clean_train_reviews = []
for i in range(0, len(train["review"])):
    clean_train_reviews.append(review_to_words(train["review"][i]))
clean_test_reviews = []
for i in range(0, len(test["review"])):
    clean_test_reviews.append(review_to_words(test["review"][i]))


#***********开始创建词袋模型***********
print("Creating the bag of words\n")

#analyzer:指定特征的类型，word表示单词，char表示字符
#tokenizer:指定分词器，None表示使用默认的分词器
#preprocessor:指定预处理器，None表示使用默认的预处理器
#stop_words:指定停用词，None表示不使用停用词

vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)

#fit_transform:将文本数据转换为特征向量
train_data_features = vectorizer.fit_transform(clean_train_reviews)

#将稀疏矩阵转换为数组
train_data_features = train_data_features.toarray()

#************利用词袋模型训练随机森林***********
print("Training the random forest")

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, train["sentiment"])

test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

results = forest.predict(test_data_features)

output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )